"""Scoped Python hooks for named model modules; no WGT-side callbacks."""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Iterator

from torch import nn


@contextmanager
def forward_hooks(model: nn.Module, hooks: Mapping[str, Callable]) -> Iterator[None]:
    """Temporarily attach PyTorch forward hooks to exact module paths.

    Each callback receives (module, inputs, output). Return None to observe;
    returning a value replaces the output, as with register_forward_hook.
    Paths come from model.named_modules(); "" selects the whole model.
    Hooks run on every invocation, including repeated shared loop cells.
    Only handles installed by this context are removed, even on exceptions.
    """
    targets = []
    for name, callback in hooks.items():
        if not isinstance(name, str):
            raise TypeError("hook target must be an exact module path string")
        if not callable(callback):
            raise TypeError(f"hook for {name!r} must be callable")
        # Validate every target before installing any hooks.
        targets.append((model.get_submodule(name), callback))
    handles = []
    try:
        for module, callback in targets:
            handles.append(module.register_forward_hook(callback))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
