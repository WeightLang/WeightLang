"""Explicit execution devices and independently resident parameter tensors."""

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn

from .syntax import WgtError


def resolve_device(value: str | torch.device) -> torch.device:
    text = str(value)
    if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", text):
        raise WgtError(f"unsupported device {text!r}; expected cpu, mps, cuda, or cuda:N")
    if text == "cpu":
        return torch.device("cpu")
    if text == "mps":
        if not torch.backends.mps.is_available():
            raise WgtError("device 'mps' requires an available macOS MPS PyTorch runtime")
        return torch.device("mps")
    if not torch.cuda.is_available():
        raise WgtError(f"device {text!r} requires an available CUDA-enabled PyTorch runtime")
    index = torch.cuda.current_device() if text == "cuda" else int(text.split(":")[1])
    if index >= torch.cuda.device_count():
        raise WgtError(f"CUDA device index {index} out of range ({torch.cuda.device_count()} available)")
    return torch.device("cuda", index)


def placement_plan(module: nn.Module, device="cpu", placements=None) -> dict:
    default = resolve_device(device)
    if placements is None:
        placements = {}
    if not isinstance(placements, dict):
        raise WgtError("placements must be a mapping of layer/parameter names to devices")
    modules = dict(module.named_modules())
    parameters = dict(module.named_parameters(remove_duplicate=False))
    rules = {}
    for name, value in placements.items():
        if not isinstance(name, str) or not name or name not in modules and name not in parameters:
            raise WgtError(f"unknown placement target {name!r}; use exact layer or parameter names")
        rules[name] = str(resolve_device(value))

    def assigned(name):
        parts = name.split(".")
        for length in range(len(parts), 0, -1):
            key = ".".join(parts[:length])
            if key in rules:
                return rules[key]
        return str(default)

    aliases = {}
    for name, value in parameters.items():
        previous = aliases.setdefault(id(value), name)
        if assigned(name) != assigned(previous):
            raise WgtError(f"tied parameters must share placement: {previous}, {name}")
    return {
        "device": str(default),
        "placements": dict(sorted(rules.items())),
        "layers": {name: assigned(name) for name, child in modules.items() if name and not list(child.children())},
        "parameters": {name: assigned(name) for name in parameters},
    }


def _placed_forward(module, x):
    """Copies remain in autograd, so gradients reach the resident parameters."""
    device = module._weightlang_device
    x = [value.to(device) for value in x] if isinstance(x, list) else x.to(device)
    state = {**dict(module.named_parameters(recurse=False)), **dict(module.named_buffers(recurse=False))}
    if all(value.device == device for value in state.values()):
        return type(module).forward(module, x)
    # A private shell avoids swapping state on the live module during forward.
    shell = copy.copy(module)
    shell.__dict__ = module.__dict__.copy()
    shell.__dict__.pop("forward", None)
    shell._parameters = module._parameters.copy()
    shell._buffers = module._buffers.copy()
    shell._forward_pre_hooks = {}
    shell._forward_hooks = {}
    shell._backward_hooks = {}
    shell._backward_pre_hooks = {}
    return torch.func.functional_call(shell, {name: value.to(device) for name, value in state.items()}, (x,))


def _resident_forward(module, x):
    """Uniform placement follows live parameters, including later Module.to calls."""
    device = next(module.parameters(recurse=False)).device
    x = [value.to(device) for value in x] if isinstance(x, list) else x.to(device)
    return type(module).forward(module, x)


def apply_placement(module: nn.Module, plan: dict) -> None:
    """Apply before constructing an optimizer; names and artifact layout stay intact."""
    aliases = {}
    for name, value in list(module.named_parameters(remove_duplicate=False)):
        parent, _, leaf = name.rpartition(".")
        owner = module.get_submodule(parent)
        if id(value) not in aliases:
            aliases[id(value)] = nn.Parameter(value.detach().to(plan["parameters"][name]), value.requires_grad)
        owner.register_parameter(leaf, aliases[id(value)])
    uniform = len(set(plan["layers"].values()) | set(plan["parameters"].values())) <= 1
    for name, device in plan["layers"].items():
        child = module.get_submodule(name)
        for key, value in list(child.named_buffers(recurse=False)):
            child.register_buffer(key, value.to(device), persistent=key not in child._non_persistent_buffers_set)
        if uniform and any(True for _ in child.parameters(recurse=False)):
            child.forward = MethodType(_resident_forward, child)
        elif not uniform:
            child._weightlang_device = torch.device(device)
            child.forward = MethodType(_placed_forward, child)


def cuda_devices(plan: dict) -> list[int]:
    values = [plan["device"], *plan["layers"].values(), *plan["parameters"].values()]
    return sorted({torch.device(value).index for value in values if value.startswith("cuda:")})


@contextmanager
def execution_rng(plan: dict, seed: int):
    """Preserve random streams for all participating accelerator backends."""
    use_mps = "mps" in [plan["device"], *plan["layers"].values(), *plan["parameters"].values()]
    with torch.random.fork_rng(devices=cuda_devices(plan)):
        state = torch.mps.get_rng_state() if use_mps else None
        try:
            if use_mps:
                torch.mps.manual_seed(seed)
            yield
        finally:
            if state is not None:
                torch.mps.set_rng_state(state)
