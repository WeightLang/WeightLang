"""Independent inference-time interventions against one immutable baseline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

from .artifact import load_artifact, sha256_bytes, sha256_file
from .patches import _unique_object
from .syntax import WgtError
from .weights import _evaluate_loaded


def _zero_output(module, inputs, output):
    if not isinstance(output, torch.Tensor):
        raise WgtError("zero_output requires a tensor-valued layer")
    return torch.zeros_like(output)


def run_ablation(artifact_path, suite_path):
    """Evaluate declared variants without training or writing model files."""
    path = Path(suite_path)
    try:
        raw = path.read_bytes()
        suite = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise WgtError(f"cannot read ablation suite: {exc}") from None
    if (
        not isinstance(suite, dict)
        or set(suite) != {"format", "format_version", "data", "loss", "variants"}
        or suite["format"] != "weightlang.ablation"
        or type(suite["format_version"]) is not int
        or suite["format_version"] != 1
        or not isinstance(suite["data"], str)
        or not suite["data"]
        or not isinstance(suite["variants"], list)
        or not suite["variants"]
    ):
        raise WgtError("expected weightlang.ablation v1 with data, loss and nonempty variants")
    baseline = load_artifact(artifact_path, warn_untrained=False)
    modules = dict(baseline.module.named_modules())
    tensors = baseline.module.state_dict()
    names = set()
    for variant in suite["variants"]:
        if not isinstance(variant, dict) or set(variant) != {"name", "interventions"}:
            raise WgtError("each variant needs name and interventions")
        name = variant["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise WgtError("variant names must be unique nonempty strings")
        names.add(name)
        edits = variant["interventions"]
        if not isinstance(edits, list) or not edits:
            raise WgtError("interventions must be a nonempty array")
        targets = []
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {"op", "target"}:
                raise WgtError("each intervention needs exactly op and target")
            op, target = edit["op"], edit["target"]
            if op not in ("zero_tensor", "zero_output", "skip"):
                raise WgtError(f"unknown ablation operation {op!r}")
            available = tensors if op == "zero_tensor" else modules
            if not isinstance(target, str) or not target or target not in available:
                raise WgtError(f"unknown ablation target {target!r}")
            if any(target == t or target.startswith(t + ".") or t.startswith(target + ".") for t in targets):
                raise WgtError("overlapping interventions in a variant are ambiguous")
            targets.append(target)
    data_path = path.parent / suite["data"]
    try:
        data = data_path.read_bytes()
    except OSError as exc:
        raise WgtError(f"cannot read evaluation data: {exc}") from None
    baseline.module.eval()
    reference = _evaluate_loaded(baseline, data_path, suite["loss"], raw=data)
    metrics = {k: v for k, v in reference["candidate"].items() if k != "artifact"}
    variants = []
    for variant in suite["variants"]:
        candidate = copy.deepcopy(baseline)
        for edit in variant["interventions"]:
            target = edit["target"]
            if edit["op"] == "zero_tensor":
                with torch.no_grad():
                    candidate.module.state_dict()[target].zero_()
            elif edit["op"] == "zero_output":
                candidate.module.get_submodule(target).register_forward_hook(_zero_output)
            else:
                parent, _, leaf = target.rpartition(".")
                setattr(candidate.module.get_submodule(parent), leaf, nn.Identity())
        try:
            result = _evaluate_loaded(candidate, data_path, suite["loss"], raw=data)
        except WgtError as exc:
            raise WgtError(f"ablation variant {variant['name']!r}: {exc}") from None
        measured = {k: v for k, v in result["candidate"].items() if k != "artifact"}
        variants.append(
            {
                **variant,
                "metrics": measured,
                "delta": {k: measured[k] - metrics[k] for k in metrics},
            }
        )
        del candidate
    return {
        "format": "weightlang.ablation-report",
        "format_version": 1,
        "suite": {"file": str(path), "sha256": sha256_bytes(raw)},
        "baseline": {
            **reference["candidate"]["artifact"],
            "manifest_sha256": sha256_file(Path(artifact_path) / "manifest.json"),
            "metrics": metrics,
        },
        "data": reference["data"],
        "loss": suite["loss"],
        "variants": variants,
        "delta_direction": "variant minus baseline; positive loss delta means degradation",
        "note": "Inference-only interventions; no retraining. Each variant starts from the same baseline.",
    }
