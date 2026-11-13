"""Shape-specialized shared-IR graphs with one canonical set of parameters.

Inputs have logical shapes (no implicit batch axes). Ordinary single-input
models return flat native-ABI outputs; explicit port models return tuples. ``state_dict`` is parameter-only: recreate the dynamic
module from source and load it before constructing an optimizer. Full SDK save
is deliberately unsupported. Source files and imports must remain available for
new shapes. This module never parses WGT or executes candidate initializers.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch import nn

from .compiler import CompilationError, _compiler, compile, compile_source
from .initialization import parameter_specs
from .runtime import build_module
from .ports import ForwardModule, unwrap_raw_module, wrap_forward

# Bound retained compiler metadata as well as graph count. A single oversized
# specialization is rejected before building its PyTorch graph.
_METADATA_BUDGET = 16 * 1024 * 1024


def _shape(shape):
    if (not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 8
            or any(type(n) is not int or not 1 <= n <= 16384 for n in shape)
            or math.prod(shape) > 16384):
        raise ValueError("input_shapes require 1..8 integer extents and at most 16384 elements")
    return tuple(shape)


def _schema(definition):
    if definition.get("kind") not in {"forward", "sequential"}:
        raise ValueError("dynamic models require ordinary shared-IR forward, not typed operations or torch_module")
    # Walk all nested IR programs, including loss and branch bodies.
    def check(value):
        if isinstance(value, dict):
            if value.get("op") == "write":
                raise ValueError("dynamic models reject persistent writes")
            if value.get("states"):
                raise ValueError("dynamic models reject module state buffers")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(definition)
    # Data is a materialized value, not metadata. Initializer recipes, domains,
    # frozen flags, logical names, shapes and all other metadata remain exact.
    return {spec["name"]: {k: v for k, v in spec.items() if k != "data"}
            for spec in parameter_specs(definition)}


def _fingerprint(definition):
    value = definition.get("template_fingerprint")
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise CompilationError(
            "dynamic models require a compiler exporting a SHA256 template_fingerprint; "
            "update the compiler and reload the dynamic model")
    return value


def _metadata_size(definition):
    size = len(json.dumps(definition, separators=(",", ":")).encode())
    if size > _METADATA_BUDGET:
        raise ValueError("dynamic specialization exceeds metadata budget")
    return size


class DynamicModule(nn.Module):
    """Compile unseen tensor shapes in Rust, sharing the original Parameters.

    ``max_specializations`` includes the permanent base graph. Remaining graphs
    use an LRU cache with a 16 MiB serialized IR budget (including the base).
    Cache eviction does not invalidate autograd graphs already held by callers.
    ``train``/``eval`` and device/dtype conversions discard candidate graphs;
    subsequent candidates bind the current base Parameters. As with ordinary
    PyTorch, move the model before creating an optimizer. Concurrent forwards
    and module mutation require host synchronization.

    ``bind_optimizer`` projects only through the registered base module's
    immutable learn metadata, once per step. Cached graphs are not registered
    children and never add parameters to state_dict or optimizer enumeration.
    """

    def __init__(self, compile_shape, *, input_shapes, device="auto", dtype=None,
                 max_specializations=8):
        super().__init__()
        if type(max_specializations) is not int or max_specializations < 1:
            raise ValueError("max_specializations must be a positive integer")
        if not isinstance(input_shapes, dict) or not input_shapes:
            raise ValueError("input_shapes must specify all forward inputs")
        if any(not isinstance(name, str) or not name for name in input_shapes):
            raise ValueError("input_shapes requires nonempty input names")
        input_shapes = {name: _shape(shape) for name, shape in input_shapes.items()}
        if sum(math.prod(shape) for shape in input_shapes.values()) > 16384:
            raise ValueError("input_shapes packed input exceeds 16384 elements")
        self._compile_shape = compile_shape
        self._max_specializations = max_specializations
        self._cache = OrderedDict()
        definition = compile_shape(input_shapes)
        self._has_ports = definition.get("forward_interface") is not None
        if self._has_ports:
            self._input_names = tuple(p["name"] for p in definition["forward_interface"]["inputs"])
            if set(input_shapes) != set(self._input_names):
                raise ValueError("input_shapes must exactly cover all forward inputs")
        else:
            if len(input_shapes) != 1:
                raise ValueError("input_shapes must specify exactly one input for a flat model")
            self._input_names = tuple(input_shapes)
        self._base_shape = self._key(input_shapes)
        self._template_fingerprint = _fingerprint(definition)
        self._schema = deepcopy(_schema(definition))
        self._base_bytes = _metadata_size(definition)
        self._cache_bytes = 0
        from .api import from_ir
        self.base = from_ir(definition, device=device, dtype=dtype)
        self._check_state(self.base)
        self._check_interface(self.base, input_shapes)
        self.parameter_names = {name: "base." + attr for name, attr in self.base.parameter_names.items()}

    @staticmethod
    def _check_state(module):
        if getattr(unwrap_raw_module(module), "state_names", None):
            raise ValueError("dynamic models reject module state buffers")

    def _key(self, shapes):
        ordered = tuple(shapes[name] for name in self._input_names)
        return ordered if self._has_ports else ordered[0]

    def _check_interface(self, module, shapes):
        if self._has_ports:
            if not isinstance(module, ForwardModule) or tuple(module.input_shapes) != self._input_names:
                raise ValueError("dynamic forward input interface changed; reload required")
            if module.input_shapes != shapes:
                raise ValueError("compiler forward input shapes do not match input_shapes")
        elif isinstance(module, ForwardModule):
            raise ValueError("dynamic forward interface changed; reload required")

    @property
    def template_fingerprint(self):
        """Rust fingerprint of the root source and frozen imported contents."""
        return self._template_fingerprint

    @property
    def specialization_shapes(self):
        """Retained shapes, including base; port keys follow input declaration order."""
        return (self._base_shape, *self._cache.keys())

    def _clear_cache(self):
        self._cache.clear()
        self._cache_bytes = 0

    def _apply(self, fn, recurse=True):
        self._clear_cache()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        result = super().train(mode)
        self._clear_cache()
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        result = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        parameters = {prefix + name for name in self.parameter_names.values()}
        for key in tuple(result):
            if key.startswith(prefix + "base.") and key not in parameters:
                del result[key]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Constants are regenerated from the verified source, not checkpointed.
        # Supply their current values to the ordinary recursive PyTorch loader.
        # Keep the base's own persistence layout intact for save(model.base).
        parameter_attrs = set(self.base.parameter_names.values())
        for key, value in self.base.state_dict().items():
            if key not in parameter_attrs:
                state_dict.setdefault(prefix + "base." + key, value)
        # assign=True may replace Parameters; discard graphs holding old ones.
        self._clear_cache()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def bind_optimizer(self, optimizer):
        from .training import bind_optimizer
        return bind_optimizer(self, optimizer)

    def forward(self, *inputs, **kwargs):
        if inputs and kwargs:
            raise TypeError("pass dynamic inputs positionally or by name, not both")
        if kwargs:
            if not self._has_ports and set(kwargs) == {"value"}:
                kwargs = {self._input_names[0]: kwargs["value"]}
            if set(kwargs) != set(self._input_names):
                raise TypeError(f"dynamic inputs must exactly match {self._input_names}")
            inputs = tuple(kwargs[name] for name in self._input_names)
        if len(inputs) != len(self._input_names):
            raise TypeError(f"dynamic forward requires {len(self._input_names)} inputs")
        anchor = unwrap_raw_module(self.base).get_buffer("_wgt_gradient_anchor")
        shapes = {}
        for name, value in zip(self._input_names, inputs):
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise TypeError("dynamic input must be a dense torch.Tensor")
            shapes[name] = _shape(tuple(value.shape))
            if value.device != anchor.device or value.dtype != anchor.dtype:
                raise ValueError("dynamic input dtype/device must match the model")
        if sum(math.prod(s) for s in shapes.values()) > 16384:
            raise ValueError("input_shapes packed input exceeds 16384 elements")
        shape = self._key(shapes)
        if any(p.device != anchor.device or p.dtype != anchor.dtype for p in self.base.parameters()):
            raise ValueError("dynamic model parameters must share dtype/device")
        if shape == self._base_shape:
            module = self.base
        elif shape in self._cache:
            module, _ = self._cache[shape]
            self._cache.move_to_end(shape)
        else:
            definition = self._compile_shape(shapes)
            if _fingerprint(definition) != self._template_fingerprint:
                raise ValueError("dynamic model source or imports changed; reload required")
            if _schema(definition) != self._schema:
                raise ValueError("dynamic shape changes parameter metadata schema or initializer recipes")
            size = _metadata_size(definition)
            module = wrap_forward(build_module(definition, _schema_only=True))
            self._check_state(module)
            self._check_interface(module, shapes)
            # Move only graph constants/anchors: applying .to to the candidate
            # Parameters could replace canonical objects under PyTorch future
            # overwrite flags. Meta placeholders are never materialized.
            for child in module.modules():
                for name, buffer in child._buffers.items():
                    if buffer is not None:
                        child._buffers[name] = buffer.to(device=anchor.device, dtype=anchor.dtype)
            from .ops import reset_gradient_anchor
            reset_gradient_anchor(unwrap_raw_module(module))
            for name, attr in module.parameter_names.items():
                owner, _, local = attr.rpartition(".")
                parent = module.get_submodule(owner) if owner else module
                setattr(parent, local, self.base.get_parameter(self.base.parameter_names[name]))
            module.train(self.training)
            if self._max_specializations > 1 and self._base_bytes + size <= _METADATA_BUDGET:
                while self._cache and (len(self._cache) >= self._max_specializations - 1
                        or self._base_bytes + self._cache_bytes + size > _METADATA_BUDGET):
                    _, (_, old_size) = self._cache.popitem(last=False)
                    self._cache_bytes -= old_size
                self._cache[shape] = (module, size)
                self._cache_bytes += size
        return module(*inputs) if self._has_ports else module(inputs[0].reshape(-1)).reshape(-1)


def load_dynamic(path, *, input_shapes, dimensions=None, device="auto", dtype=None,
                 compiler=None, timeout=30, max_specializations=8):
    """Load WGT from a file; infer each call's logical input shape in Rust.

    Files/imports are re-read for unseen shapes; parameter schemas and recipes
    must remain identical. Ordinary models return flat outputs; explicit ports
    return tuples of logical tensors. Use state_dict for parameter-only
    checkpoints; weightlang.save rejects DynamicModule. Explicitly saving
    model.base exports only the initial-shape static model.

    input_shapes is mandatory and maps all root input names to positive logical
    shapes. Calls use declaration order or all named inputs; tensor.shape
    replaces each binding. Packed inputs and outputs remain limited to 16384;
    dimensions are fixed explicit Rust dimension bindings, not Python inference.
    device defaults to auto, dtype to the ordinary SDK default (float32),
    compiler to SDK compiler discovery, timeout to 30 seconds and
    max_specializations to 8 (including the permanent base). Schema-changing
    shapes raise ValueError without changing parameters or the retained cache;
    Rust diagnostics propagate as CompilationError. Parameter metadata compares
    every field except materialized data, including initializer recipes.
    The compiler path is resolved once. Each compilation must export the same
    SHA256 template_fingerprint for root source and frozen imports. Changed
    source/imports are rejected before building or caching a graph with a
    "reload required" diagnostic, even when parameter schemas match. Cached
    graphs continue using their original compiled contents.
    """
    path = Path(path).expanduser().resolve(strict=True)
    dimensions = deepcopy(dimensions)
    compiler = str(Path(_compiler(compiler)).resolve(strict=True))
    return DynamicModule(lambda shapes: compile(path, dimensions=dimensions,
                         input_shapes=shapes, compiler=compiler, timeout=timeout),
                         input_shapes=input_shapes, device=device, dtype=dtype,
                         max_specializations=max_specializations)


def loads_dynamic(source, *, input_shapes, dimensions=None, device="auto", dtype=None,
                  compiler=None, timeout=30, max_specializations=8):
    """Inline equivalent of load_dynamic, using compile_source without parsing.

    Ordinary single-input models return flat native-ABI tensors; explicit
    forward ports accept positional or named tensors and return tuples. Create
    standard optimizers once from the returned canonical parameters.
    """
    dimensions = deepcopy(dimensions)
    compiler = str(Path(_compiler(compiler)).resolve(strict=True))
    return DynamicModule(lambda shapes: compile_source(source, dimensions=dimensions,
                         input_shapes=shapes, compiler=compiler, timeout=timeout),
                         input_shapes=input_shapes, device=device, dtype=dtype,
                         max_specializations=max_specializations)
