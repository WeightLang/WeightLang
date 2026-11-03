"""Persist Python weight updates without losing the declarative model contract."""

from __future__ import annotations

import copy
import json
import platform
import shutil
from pathlib import Path

import torch

from .artifact import Artifact, load_artifact, sha256_file, write_safetensors
from .syntax import WgtError


def _architecture(module):
    result = []
    for name, child in module.named_modules():
        forward = child.__dict__.get("forward")
        if any(getattr(child, key) for key in ("_forward_hooks", "_forward_pre_hooks")):
            raise WgtError("cannot save Python forward hooks")
        config = {key: value for key, value in vars(child).items()
                  if not key.startswith("_") and key not in ("training", "forward")}
        result.append((name, type(child), child.extra_repr(), config, getattr(forward, "__func__", forward)))
    return result


def save_artifact(
    artifact: Artifact, path: str | Path, *, training: dict | None = None
) -> Artifact:
    """Save weights to a NEW exact directory and return its verified CPU artifact.

    With no training record, changed tensors are marked edited. A JSON training
    record with a nonempty ``method`` explicitly attests that changed parameters
    were trained by the caller; it is not independent quality verification.
    The source architecture, frozen parameters and baseline must remain intact.
    """
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise WgtError("output already exists; choose a new artifact directory")
    if output.resolve().is_relative_to(artifact.path.resolve()):
        raise WgtError("output must be outside the baseline artifact directory")
    if training is not None:
        if not isinstance(training, dict) or not isinstance(training.get("method"), str) or not training["method"].strip():
            raise WgtError("training must be a JSON dictionary with a nonempty method")
        try:
            training = json.loads(json.dumps(training, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise WgtError(f"invalid training record: {exc}") from exc
    baseline = load_artifact(artifact.path, warn_untrained=False)
    if baseline.manifest != artifact.manifest:
        raise WgtError("baseline manifest changed; reload before saving")
    if _architecture(artifact.module) != _architecture(baseline.module):
        raise WgtError("model structure or forward changed; only weight updates can be saved")
    before = baseline.module.state_dict()
    state = artifact.module.state_dict()
    if state.keys() != before.keys():
        raise WgtError("model tensor names changed")
    parameters = dict(artifact.module.named_parameters())
    frozen = {name for name, p in baseline.module.named_parameters() if not p.requires_grad}
    changed = []
    for name, value in state.items():
        if value.dtype != before[name].dtype or value.shape != before[name].shape:
            raise WgtError(f"tensor {name} shape or dtype changed")
        if not torch.isfinite(value).all():
            raise WgtError(f"tensor {name} contains non-finite values")
        if not torch.equal(value.detach().cpu(), before[name]):
            if name in frozen:
                raise WgtError(f"frozen parameter {name} was modified")
            changed.append(name)
    manifest = copy.deepcopy(baseline.manifest)
    previous = {name: copy.deepcopy(manifest["weights"]["tensors"][name]) for name in changed}
    for name in changed:
        meta = manifest["weights"]["tensors"][name]
        if training is not None and name in parameters:
            meta["origin"] = "imported+trained" if meta["origin"] in ("imported", "imported+trained") else "trained"
        else:
            meta["origin"] = "edited"
            meta.pop("upstream_origin", None)
    record = {
        "baseline_manifest_sha256": sha256_file(artifact.path / "manifest.json"),
        "baseline_weights_sha256": baseline.manifest["weights"]["sha256"],
        "changed_tensors": changed,
        "previous_tensor_metadata": previous,
        "previous_training": manifest.get("training"),
        "training": training,
        "provenance": "caller-reported" if training is not None else "direct-edit",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    }
    manifest.setdefault("python_history", []).append(record)
    if changed:
        manifest["training"] = None if training is None else {**training, "backend": "python", "provenance": "caller-reported"}
    manifest["format_version"] = 3
    manifest["rebuild"] = "Source describes the base model; Python weight updates require the external training code and data."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation: never replace a retained model.
    try:
        shutil.copyfile(artifact.path / "source.wgt", output / "source.wgt")
        manifest["source"] = {"file": "source.wgt", "sha256": sha256_file(output / "source.wgt")}
        write_safetensors(output / "weights.safetensors", state, {"format": manifest["format"], "model": artifact.name})
        manifest["weights"]["file"] = "weights.safetensors"
        manifest["weights"]["sha256"] = sha256_file(output / "weights.safetensors")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return load_artifact(output, warn_untrained=False)
    except BaseException:
        shutil.rmtree(output)
        raise
