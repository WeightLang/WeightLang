"""Architecture loading and portable model persistence for PyTorch hosts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn

from .compiler import compile, compile_source
from .runtime import build_module

MODEL_FORMAT = "weightlang.model.v1"


def select_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve auto to an available CUDA, XPU, MPS, or CPU device, in that order."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def from_ir(
    definition: dict[str, Any],
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Initialize a PyTorch module from a compiled definition; no compiler needed.

    Parameters start from declared initializers. Training and checkpoint restore
    are separate operations. Inputs must be tensors on the model's device.
    """
    return build_module(definition).to(device=select_device(device), dtype=dtype)


def loads(
    source: str,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    compiler: str | os.PathLike[str] | None = None,
    timeout: float = 30,
) -> nn.Module:
    """Compile inline WGT and initialize an ordinary, trainable PyTorch module."""
    return from_ir(compile_source(source, compiler=compiler, timeout=timeout), device=device, dtype=dtype)


def load(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    compiler: str | os.PathLike[str] | None = None,
    timeout: float = 30,
) -> nn.Module:
    """Load .wgt source, compiled .json, or a .pt/.pth model saved by save().

    Only .wgt requires the native compiler. Saved models contain architecture and
    state_dict, and use torch.load(weights_only=True). Optimizer and data-loader
    state remain the host application's responsibility. Loaded modules start in
    training mode, following torch.nn.Module; call eval() for inference.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".wgt":
        return from_ir(compile(path, compiler=compiler, timeout=timeout), device=device, dtype=dtype)
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            return from_ir(json.load(stream), device=device, dtype=dtype)
    if suffix not in {".pt", ".pth"}:
        raise ValueError("Expected .wgt source, compiled .json, or an SDK .pt/.pth model")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
        raise ValueError("Expected a model saved by weightlang.save; load plain state_dict files with PyTorch")
    if not isinstance(bundle.get("definition"), dict) or not isinstance(bundle.get("state_dict"), dict):
        raise ValueError("Saved WeightLang model requires a definition and state_dict")
    # Reuse the exact registration layout, but allocate no parameter/state data
    # and execute no declared initializer before binding the saved values.
    model = build_module(bundle["definition"], _schema_only=True)
    # assign preserves saved tensor precision; create optimizers after loading.
    model.load_state_dict(bundle["state_dict"], strict=True, assign=True)
    from .initialization import parameter_specs
    from .program import validate_parameter
    for spec in parameter_specs(bundle["definition"]):
        validate_parameter(model.get_parameter(model.parameter_names[spec["name"]]), spec)
    for spec in bundle["definition"].get("operations", {}).get("states", []):
        validate_parameter(model.get_buffer(model.state_names[spec["name"]]), spec)
    if any(not bool(torch.isfinite(value).all()) for value in model.buffers() if value.is_floating_point()):
        raise ValueError("saved model contains nonfinite state or constants")
    return model.to(device=select_device(device), dtype=dtype)


def save(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Save architecture and state_dict for compiler-free loading with load().

    The destination must end in .pt or .pth. Parent directories are created.
    Existing files are preserved unless overwrite=True is explicitly supplied.
    Use torch.save directly when also saving optimizer or application state.
    """
    # torch.compile wraps the original module and prefixes state_dict keys.
    # Persist the architecture's own parameters, independent of that wrapper.
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, nn.Module):
        model = original
    definition = getattr(model, "definition", None)
    if not isinstance(definition, dict):
        raise TypeError("save expects an unmodified WeightLang module with its compiled definition")
    path = Path(path).expanduser()
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Saved SDK models use a .pt or .pth extension")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Model already exists: {path}; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"format": MODEL_FORMAT, "definition": definition, "state_dict": model.state_dict()}
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(bundle, temporary_path)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # Linking publishes the complete file and refuses an existing path.
            os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
