"""Explicit, hash-bound weight edits. A patch is data, never executable code."""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import torch

from .artifact import load_artifact, sha256_bytes, sha256_file, write_safetensors
from .syntax import WgtError


def patch_template(path: str | Path, author: str) -> dict:
    artifact = load_artifact(path, warn_untrained=False)
    if not author.strip():
        raise WgtError("author must not be empty (self-reported attribution)")
    return {
        "format": "weightlang.patch",
        "format_version": 1,
        "baseline_weights_sha256": artifact.manifest["weights"]["sha256"],
        "baseline_manifest_sha256": sha256_file(artifact.path / "manifest.json"),
        "author": author,
        "reason": "Describe the intended behavior change",
        "edits": [],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WgtError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _scalar(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and abs(value) <= torch.finfo(torch.float32).max
    except OverflowError:
        valid = False
    if not valid:
        raise WgtError(f"{label} must be a finite float32 number")
    return torch.tensor(value, dtype=torch.float32).item()


def apply_patch(path: str | Path, patch_path: str | Path, output: str | Path) -> dict:
    """Create a new artifact directory, refusing stale patches and existing outputs."""
    try:
        return _apply_patch(Path(path), Path(patch_path), Path(output))
    except (OSError, ValueError, OverflowError) as exc:
        raise WgtError(f"cannot apply weight patch: {exc}") from None


def _apply_patch(path: Path, patch_path: Path, output: Path) -> dict:
    raw = patch_path.read_bytes()
    patch = json.loads(raw, object_pairs_hook=_unique_object)
    fields = {
        "format",
        "format_version",
        "baseline_weights_sha256",
        "baseline_manifest_sha256",
        "author",
        "reason",
        "edits",
    }
    if not isinstance(patch, dict) or set(patch) != fields:
        raise WgtError("patch must contain exactly the documented fields; use patch-template")
    if (
        patch["format"] != "weightlang.patch"
        or type(patch["format_version"]) is not int
        or patch["format_version"] != 1
    ):
        raise WgtError("unsupported weight patch format")
    for label in ("author", "reason"):
        if not isinstance(patch[label], str) or not patch[label].strip():
            raise WgtError(f"patch {label} must be a nonempty string")
    edits = patch["edits"]
    if not isinstance(edits, list) or not edits:
        raise WgtError("patch edits must be a nonempty array")
    if output.exists() or output.is_symlink():
        raise WgtError("output already exists; choose a new artifact directory")
    if output.resolve().is_relative_to(path.resolve()):
        raise WgtError("output must be outside the baseline artifact directory")
    baseline_manifest = (path / "manifest.json").read_bytes()
    if sha256_bytes(baseline_manifest) != patch["baseline_manifest_sha256"]:
        raise WgtError("baseline manifest sha256 mismatch: stale patch")
    artifact = load_artifact(path, warn_untrained=False)
    if artifact.manifest != json.loads(baseline_manifest):
        raise WgtError("baseline manifest changed while reading")
    if artifact.manifest["weights"]["sha256"] != patch["baseline_weights_sha256"]:
        raise WgtError("baseline weights sha256 mismatch: stale patch")
    tensors = artifact.module.state_dict()
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise WgtError("baseline contains non-finite weights")
    seen = set()
    changes = []
    for edit in edits:
        if (
            not isinstance(edit, dict)
            or not {"tensor", "index", "value"} <= set(edit)
            or set(edit) - {"tensor", "index", "value", "expected"}
        ):
            raise WgtError("each edit needs tensor, index, value; only expected is optional")
        name, index = edit["tensor"], edit["index"]
        if not isinstance(name, str) or name not in tensors:
            raise WgtError(f"unknown tensor {name!r}")
        if type(index) is not int or not 0 <= index < tensors[name].numel():
            raise WgtError(f"index out of bounds for tensor {name!r}")
        if (name, index) in seen:
            raise WgtError(f"duplicate edit for {name}[{index}]")
        seen.add((name, index))
        values = tensors[name].view(-1)
        before = values[index].item()
        after = _scalar(edit["value"], "value")
        if "expected" in edit and _scalar(edit["expected"], "expected") != before:
            raise WgtError(f"expected value mismatch for {name}[{index}]")
        if after == before:
            continue
        values[index] = after
        changes.append({"tensor": name, "index": index, "before": before, "after": after})
    if not changes:
        raise WgtError("patch changes no float32 values")
    manifest = copy.deepcopy(artifact.manifest)
    touched = sorted({change["tensor"] for change in changes})
    previous = {name: copy.deepcopy(manifest["weights"]["tensors"][name]) for name in touched}
    for name in touched:
        manifest["weights"]["tensors"][name]["origin"] = "edited"
        # Prior origins belong in history, not in the current imported-origin shortcut.
        manifest["weights"]["tensors"][name].pop("upstream_origin", None)
    record = {
        "patch": patch,
        "patch_sha256": sha256_bytes(raw),
        "changes": changes,
        "previous_tensor_metadata": previous,
        "previous_training": manifest.get("training"),
    }
    manifest.setdefault("edit_history", []).append(record)
    manifest["training"] = None
    # Edited provenance is a v3 extension; old readers must refuse it.
    manifest["format_version"] = 3
    manifest["rebuild"] = (
        "source describes the base architecture; replay edit_history patches after building the exact baseline"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # exclusive reservation; never overwrite even an empty directory
    try:
        shutil.copyfile(artifact.path / "source.wgt", output / "source.wgt")
        weights = output / "weights.safetensors"
        write_safetensors(weights, tensors, {"format": "weightlang.edited"})
        manifest["weights"]["sha256"] = sha256_file(weights)
        # Publish the manifest last; partially written directories are not valid artifacts.
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        load_artifact(output, warn_untrained=False)
    except BaseException:
        shutil.rmtree(output)
        raise
    return {
        "format": "weightlang.patch-result",
        "format_version": 1,
        "artifact": str(output),
        "baseline_weights_sha256": patch["baseline_weights_sha256"],
        "weights_sha256": manifest["weights"]["sha256"],
        "patch_sha256": record["patch_sha256"],
        "changed_elements": len(changes),
        "edited_tensors": touched,
        "note": "Directly edited, not trained or validated. Evaluate against the baseline before claiming improvement.",
    }
