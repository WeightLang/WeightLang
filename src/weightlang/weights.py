"""Read-only tools for inspecting weights and measuring candidate changes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from .artifact import Artifact, load_artifact, sha256_bytes, structure_signature
from .syntax import WgtError


def _identity(artifact: Artifact) -> dict:
    return {
        "model": artifact.name,
        "weights_sha256": artifact.manifest["weights"]["sha256"],
        "untrained_tensors": artifact.untrained_tensors,
    }


def _tensors(artifact: Artifact) -> dict[str, torch.Tensor]:
    tensors = artifact.module.state_dict()
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise WgtError(f"tensor {name!r} contains non-finite values")
    return tensors


def export_weights(path: str | Path, *, layer: str | None = None, limit: int = 256) -> dict:
    """Export row-major float32 values, with a global scalar budget (0 = all)."""
    if limit < 0:
        raise WgtError("limit must be non-negative (0 exports all values)")
    artifact = load_artifact(path, warn_untrained=False)
    tensors = _tensors(artifact)
    selected = {n: t for n, t in sorted(tensors.items()) if layer is None or n.startswith(layer + ".")}
    if layer is not None and not selected:
        raise WgtError(f"no parameter tensors under layer {layer!r}")
    remaining = limit
    records = {}
    for name, tensor in selected.items():
        count = tensor.numel() if limit == 0 else min(remaining, tensor.numel())
        remaining -= count
        records[name] = {
            **artifact.manifest["weights"]["tensors"][name],
            "shape": list(tensor.shape),
            "dtype": "float32",
            "values": tensor.flatten()[:count].tolist(),
            "omitted": tensor.numel() - count,
        }
    return {
        "format": "weightlang.weights",
        "format_version": 1,
        "artifact": _identity(artifact),
        "interface": artifact.manifest["interface"],
        "layers": artifact.manifest["layers"],
        "layout": "flat row-major; values start at offset 0 in each tensor",
        "tensors": records,
    }


def _compatible(before: Artifact, after: Artifact) -> None:
    def signature(artifact):
        return [(layer["name"], structure_signature(layer)) for layer in artifact.manifest["layers"]]

    def interface(artifact):
        return [
            (artifact.manifest["interface"][k]["dtype"], artifact.manifest["interface"][k]["shape"])
            for k in ("input", "output")
        ]

    if signature(before) != signature(after) or interface(before) != interface(after):
        raise WgtError("incompatible artifacts: architecture or input/output contract differs")


def diff_weights(before_path: str | Path, after_path: str | Path) -> dict:
    before = load_artifact(before_path, warn_untrained=False)
    after = load_artifact(after_path, warn_untrained=False)
    _compatible(before, after)
    old, new = _tensors(before), _tensors(after)
    records = {}
    total = changed = 0
    for name in sorted(old):
        a, b = old[name], new[name]
        delta = b.to(torch.float64) - a.to(torch.float64)
        n = int((a != b).sum())
        total += a.numel()
        changed += n
        records[name] = {
            "shape": list(a.shape),
            "elements": a.numel(),
            "changed_elements": n,
            "max_abs_delta": float(delta.abs().max()),
            "l2_delta": float(torch.linalg.vector_norm(delta)),
            "before_origin": before.manifest["weights"]["tensors"][name]["origin"],
            "after_origin": after.manifest["weights"]["tensors"][name]["origin"],
        }
    return {
        "format": "weightlang.weight-diff",
        "format_version": 1,
        "before": _identity(before),
        "after": _identity(after),
        "elements": total,
        "changed_elements": changed,
        "changed_percent": 100 * changed / total if total else 0.0,
        "comparison": "exact numeric float32 inequality; percentage counts scalar parameters, not tensors",
        "tensors": records,
    }


def _number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and abs(value) <= torch.finfo(torch.float32).max
    except OverflowError:
        return False


def _array(values, shape, label, *, integer=False, bound=None):
    def valid(value, dims):
        if not dims:
            return _number(value) and (not integer or value == int(value) and 0 <= value < bound)
        return isinstance(value, list) and len(value) == dims[0] and all(valid(v, dims[1:]) for v in value)

    if not valid(values, shape):
        kind = "integer class index / token" if integer else "numeric"
        raise WgtError(f"{label} must contain finite {kind} values with shape {shape} and matching rows")
    return torch.tensor(values, dtype=torch.int64 if integer else torch.float32)


def evaluate_artifact(path: str | Path, data_path: str | Path, loss: str, *, compare: str | Path | None = None) -> dict:
    """Evaluate exactly the same file on candidate and optional baseline; never train."""
    try:
        raw = Path(data_path).read_bytes()
    except OSError as exc:
        raise WgtError(f"cannot read evaluation data: {exc}") from None
    candidate = load_artifact(path, warn_untrained=False)
    report = _evaluate_loaded(candidate, data_path, loss, raw=raw)
    if compare is not None:
        baseline = load_artifact(compare, warn_untrained=False)
        _compatible(baseline, candidate)
        report["baseline"] = _evaluate_loaded(baseline, data_path, loss, raw=raw)["candidate"]
        report["delta"] = {
            key: report["candidate"][key] - report["baseline"][key]
            for key in ("loss", "accuracy")
            if key in report["candidate"]
        }
        report["delta_direction"] = "candidate minus baseline; lower loss and higher accuracy are better"
    return report


def _evaluate_loaded(candidate: Artifact, data_path, loss: str, *, raw: bytes | None = None) -> dict:
    """Shared evaluator for a loaded model, optionally using a fixed dataset snapshot."""
    if loss not in ("mse", "bce", "cross_entropy"):
        raise WgtError(f"unsupported evaluation loss {loss!r}")
    try:
        if raw is None:
            raw = Path(data_path).read_bytes()
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise WgtError(f"cannot read evaluation data: {exc}") from None
    if not isinstance(data, dict) or set(data) != {"inputs", "targets"}:
        raise WgtError("evaluation data must contain exactly inputs and targets")
    inp = candidate.manifest["interface"]["input"]
    outshape = candidate.manifest["interface"]["output"]["shape"]
    width = outshape[-1]
    if not isinstance(data["inputs"], list) or not data["inputs"]:
        raise WgtError("inputs must be a nonempty numeric batch")
    rows = len(data["inputs"])
    x = _array(data["inputs"], [rows, *inp["shape"]], "inputs", integer=inp["dtype"] == "i64", bound=2**63)
    if loss == "cross_entropy":
        y = _array(data["targets"], [rows, *outshape[:-1]], "cross_entropy targets", integer=True, bound=width)
    else:
        y = _array(data["targets"], [rows, *outshape], "targets")
        if loss == "bce" and not torch.all((y == 0) | (y == 1)):
            raise WgtError("bce targets must be 0 or 1")

    def metrics(artifact):
        _tensors(artifact)
        prediction = artifact.predict(x)
        if list(prediction.shape) != [len(x), *outshape] or not torch.isfinite(prediction).all():
            raise WgtError("model produced invalid evaluation outputs")
        if loss == "cross_entropy":
            last = artifact.module
            while type(last) is nn.Sequential:
                last = last[-1]
            if isinstance(last, (nn.Softmax, nn.Sigmoid)):
                raise WgtError("cross_entropy requires raw logits, not final softmax/sigmoid")
            value = nn.functional.cross_entropy(prediction.reshape(-1, width), y.reshape(-1))
            accuracy = (prediction.argmax(-1) == y).double().mean().item()
        elif loss == "bce":
            if not torch.all((prediction >= 0) & (prediction <= 1)):
                raise WgtError("bce requires probabilities in [0, 1]")
            value = nn.functional.binary_cross_entropy(prediction, y)
            accuracy = ((prediction >= 0.5) == y.bool()).double().mean().item()
        else:
            value = nn.functional.mse_loss(prediction, y)
            accuracy = None
        if not torch.isfinite(value):
            raise WgtError("evaluation loss is non-finite")
        result = {"artifact": _identity(artifact), "loss": value.item()}
        if accuracy is not None:
            result["accuracy"] = accuracy
        return result

    report = {
        "format": "weightlang.evaluation",
        "format_version": 1,
        "data": {"file": str(data_path), "sha256": sha256_bytes(raw), "rows": len(x)},
        "loss": loss,
        "candidate": metrics(candidate),
        "note": "Read-only evaluation. Dataset independence and model quality are not inferred.",
    }
    return report
