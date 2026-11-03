"""Auditable tensor transfer across architecture revisions, before training."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .artifact import build_module, load_artifact, sha256_file
from .syntax import WgtError, parse_file


class ReuseError(WgtError):
    def __init__(self, message, report):
        super().__init__(message)
        self.reuse_report = report


def _under(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def _owners(layers, prefix=""):
    result = {}
    for layer in layers:
        name = prefix + layer["name"]
        result[name] = layer
        result.update(_owners(layer.get("layers", []), name + "."))
    return result


def _owner(name, owners):
    matches = [key for key in owners if _under(name, key)]
    return owners[max(matches, key=len)] if matches else None


def plan(program, module, source=None, *, policy="exact", mapping=None, strict=False, revision=None):
    if policy not in ("exact", "overlap", "none"):
        raise WgtError("reuse policy must be exact, overlap or none")
    if policy == "none" and (strict or mapping or source is not None):
        raise WgtError("reuse=none cannot be combined with a source, mappings or strict mode")
    mapping = mapping or {}
    target = module.state_dict()
    for name, original in mapping.items():
        if not isinstance(name, str) or not isinstance(original, str) or not name or not original:
            raise WgtError("reuse mappings must have nonempty target and source names")
        if not any(_under(tensor, name) for tensor in target):
            raise WgtError(f"reuse mapping target {name!r} matches no tensors")
        if any(_under(name, item.target) or _under(item.target, name) for item in program.imports):
            raise WgtError(f"reuse mapping {name!r} conflicts with an explicit import/inherit")
    layers = [layer.to_dict() for layer in program.layers]
    target_owners = _owners(layers)
    previous = None
    if source is not None:
        try:
            with torch.random.fork_rng(devices=[]):
                previous = load_artifact(source, warn_untrained=False)
        except (WgtError, OSError) as exc:
            raise baseline_error(str(exc), policy) from None
    original = previous.module.state_dict() if previous else {}
    source_owners = _owners(previous.manifest["layers"]) if previous else {}
    rows = []
    used = set()
    for name, tensor in target.items():
        count = tensor.numel()
        row = {
            "target": name,
            "target_shape": list(tensor.shape),
            "target_values": count,
            "status": "initialized",
            "copied_values": 0,
            "initialized_values": count,
        }
        candidates = [key for key in mapping if _under(name, key)]
        mapped = max(candidates, key=len) if candidates else None
        old_name = mapping[mapped] + name[len(mapped) :] if mapped else name
        row["source"] = old_name
        if any(_under(name, item.target) for item in program.imports):
            row.update(status="explicit_import", reason="source_directive_takes_precedence", initialized_values=0)
        elif policy == "none":
            row["reason"] = "reuse_disabled"
        elif previous is None:
            row["reason"] = "no_baseline"
        elif old_name not in original:
            row["reason"] = "source_tensor_missing"
        else:
            old = original[old_name]
            row.update(source_shape=list(old.shape), source_origin=previous.manifest["weights"]["tensors"][old_name])
            dst_owner, src_owner = _owner(name, target_owners), _owner(old_name, source_owners)
            same_op = dst_owner is not None and src_owner is not None and dst_owner["op"] == src_owner["op"]
            same_config = same_op and dst_owner.get("args", {}) == src_owner.get("args", {})
            if same_op and dst_owner["op"] == "linear" and name.endswith(".bias"):
                # Linear bias is indexed by outputs; changing input width does
                # not change the bias coordinate system.
                same_config = {k: v for k, v in dst_owner.get("args", {}).items() if k != "in"} == {
                    k: v for k, v in src_owner.get("args", {}).items() if k != "in"
                }
            # An explicit mapping cannot reinterpret a bias as a weight, even if square.
            same_role = name.rsplit(".", 1)[-1] == old_name.rsplit(".", 1)[-1]
            if not same_role:
                row["reason"] = "parameter_role_mismatch"
            elif not same_op:
                row["reason"] = "operator_mismatch"
            elif tensor.dtype != old.dtype:
                row["reason"] = "dtype_mismatch"
            elif tuple(tensor.shape) == tuple(old.shape) and same_config:
                row.update(status="reused", reason="exact_match", copied_values=count, initialized_values=0)
            elif policy == "overlap" and dst_owner["op"] in ("linear", "embedding", "position_embedding"):
                if tensor.ndim != old.ndim or tensor.ndim not in (1, 2):
                    row["reason"] = "rank_not_supported"
                else:
                    extents = [min(a, b) for a, b in zip(tensor.shape, old.shape)]
                    copied = math.prod(extents)
                    row.update(
                        status="partial",
                        reason="explicit_leading_overlap",
                        extents=extents,
                        copied_values=copied,
                        initialized_values=count - copied,
                        source_unused_values=old.numel() - copied,
                    )
            elif tuple(tensor.shape) != tuple(old.shape):
                row["reason"] = "shape_mismatch" if policy == "exact" else "overlap_not_supported_for_operator"
            else:
                row["reason"] = "operator_configuration_changed"
            if row["copied_values"]:
                used.add(old_name)
        rows.append(row)
    total = sum(row["target_values"] for row in rows)
    copied = sum(row["copied_values"] for row in rows)
    fresh = sum(row["initialized_values"] for row in rows)
    explicit = sum(row["target_values"] for row in rows if row["status"] == "explicit_import")
    automatic = [row for row in rows if row["status"] != "explicit_import"]
    if policy == "none":
        status = "disabled"
    elif not automatic:
        status = "explicit_only"
    elif previous is None:
        status = "no_baseline"
    elif all(row["status"] == "reused" for row in automatic):
        status = "complete"
    elif copied:
        status = "partial"
    else:
        status = "no_reuse"
    interface_changed = previous is not None and any(
        previous.manifest["interface"][role]["dtype"] != port.dtype
        or previous.manifest["interface"][role]["shape"] != list(port.shape)
        for role, port in (("input", program.input), ("output", program.output))
    )
    report = {
        "format": "weightlang.reuse",
        "format_version": 1,
        "status": status,
        "policy": policy,
        "strict": strict,
        "stage": "initialization_before_training",
        "execution": "planned",
        "quality": "not_evaluated",
        "architecture_changed": interface_changed or (previous is not None and previous.manifest["layers"] != layers),
        "interface_changed": interface_changed,
        "source": None,
        "mapping": mapping,
        "summary": {
            "target_tensors": len(rows),
            "target_values": total,
            "copied_values": copied,
            "initialized_values": fresh,
            "explicit_import_values": explicit,
            "copied_fraction": copied / total if total else 0.0,
            "reused_tensors": sum(r["status"] == "reused" for r in rows),
            "partial_tensors": sum(r["status"] == "partial" for r in rows),
            "initialized_tensors": sum(r["status"] == "initialized" for r in rows),
        },
        "tensors": rows,
        "unused_source_tensors": sorted(set(original) - used),
    }
    if previous:
        report["source"] = {
            "model": previous.name,
            "weights_sha256": previous.manifest["weights"]["sha256"],
            "manifest_sha256": sha256_file(Path(source) / "manifest.json"),
            "revision": revision,
        }
    if strict and any(row["status"] not in ("reused", "explicit_import") for row in rows):
        report.update(status="failed", failure="strict_requires_exact_reuse")
        raise ReuseError("strict reuse failed; inspect per-tensor reasons (no training was run)", report)
    return report, original, previous


def apply(program, module, source=None, **options):
    report, original, previous = plan(program, module, source, **options)
    target = module.state_dict()
    upstream = {}
    with torch.no_grad():
        for row in report["tensors"]:
            if row["status"] not in ("reused", "partial"):
                continue
            dst, src = target[row["target"]], original[row["source"]]
            if row["status"] == "partial":
                slices = tuple(slice(0, size) for size in row["extents"])
                dst[slices].copy_(src[slices])
            else:
                dst.copy_(src)
            if row["copied_values"] == row["target_values"]:
                meta = previous.manifest["weights"]["tensors"][row["source"]]
                upstream[row["target"]] = (
                    meta.get("upstream_origin", "imported") if meta["origin"] == "imported" else meta["origin"]
                )
    report["execution"] = "applied"
    return report, upstream


def preview(source_file, baseline=None, **options):
    program = parse_file(source_file)
    with torch.device("meta"):
        module = build_module([layer.to_dict() for layer in program.layers])
    return plan(program, module, baseline, **options)[0]


def baseline_error(message, policy="exact"):
    return ReuseError(
        message,
        {
            "format": "weightlang.reuse",
            "format_version": 1,
            "status": "failed",
            "policy": policy,
            "stage": "initialization_before_training",
            "quality": "not_evaluated",
            "execution": "planned",
            "failure": "baseline_validation_failed",
            "error": message,
            "tensors": [],
        },
    )
