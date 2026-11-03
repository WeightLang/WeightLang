"""Algorithm-independent parameter snapshots, functional binding and explicit writes."""
from collections.abc import Mapping

import torch

from .syntax import WgtError


def read_parameters(module, names=None):
    """Detached, independent snapshots. Defaults to one path per shared Parameter."""
    available = dict(module.named_parameters(remove_duplicate=False))
    selected = list(dict(module.named_parameters()) if names is None else names)
    if len(set(selected)) != len(selected) or any(n not in available for n in selected):
        raise WgtError("parameter names must be unique existing paths")
    return {name: available[name].detach().clone() for name in selected}


def _bindings(module, values, *, writing):
    if not isinstance(values, Mapping):
        raise WgtError("parameters must be a mapping of paths to tensors")
    available = dict(module.named_parameters(remove_duplicate=False))
    seen, result = set(), {}
    for name, value in values.items():
        if name not in available:
            raise WgtError(f"unknown parameter: {name}")
        parameter = available[name]
        if id(parameter) in seen:
            raise WgtError(f"multiple paths address the same shared parameter: {name}")
        seen.add(id(parameter))
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise WgtError(f"parameter {name} requires a dense tensor")
        if (value.shape, value.dtype, value.device) != (parameter.shape, parameter.dtype, parameter.device):
            raise WgtError(f"parameter {name} shape, dtype and device must match exactly")
        if writing and not parameter.requires_grad:
            raise WgtError(f"frozen parameter: {name}")
        if not torch.isfinite(value).all():
            raise WgtError(f"nonfinite parameter: {name}")
        result[name] = value
    return available, result


def call_with_parameters(module, parameters, *args, **kwargs):
    """Run with temporary tensor bindings; unspecified weights retain their values.

    Shared aliases follow the supplied binding. Gradients to supplied tensors are
    preserved; this API does not require or disable autograd. Registered buffers
    are cloned to isolate forward updates. This is not a sandbox for Python side
    effects or a concurrency primitive; serialize access to the module.
    """
    _, bindings = _bindings(module, parameters, writing=False)
    # Clone with gradients intact: even an in-place forward must not mutate the
    # caller's bindings or the model's original parameters.
    replacements = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        replacements.setdefault(id(parameter), (name, parameter))
    for name, value in bindings.items():
        replacements[id(module.get_parameter(name))] = (name, value)
    weights = {name: value.clone() for name, value in replacements.values()}
    buffers = {name: value.clone() for name, value in module.named_buffers()}
    return torch.func.functional_call(module, (weights, buffers), args, kwargs, tie_weights=True)


def write_parameters(module, parameters):
    """Validate then copy a partial mapping in place, preserving Parameter identity.

    Frozen weights are rejected. Values are snapshotted before any copy, so swaps
    and views of current weights have simultaneous-assignment semantics. Writes
    are detached, do not reset gradients/optimizer state, and do not save files.
    Validation errors leave all parameters unchanged. Device failures are not a
    transactional rollback guarantee. Do not write while a backward pass is pending.
    """
    available, bindings = _bindings(module, parameters, writing=True)
    snapshots = {name: value.detach().clone() for name, value in bindings.items()}
    with torch.no_grad():
        for name, value in snapshots.items():
            available[name].copy_(value)
