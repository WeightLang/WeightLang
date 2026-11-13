"""Logical forward ports over the shared Rust compiler's packed vector ABI.

Only compiler-exported forward_interface metadata is consumed. There is no WGT
source parser. Output fields always form a tuple, including a single field.
Source-defined loss is unsupported; compute loss in the host using the returned
live tensors. For packed inspection/research use ``model.base`` with a packed
input and output seed; research helpers retain their existing vector contract.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn


def _ports(items, *, inputs, width):
    label = "inputs" if inputs else "outputs"
    if not isinstance(items, list) or not items:
        raise ValueError(f"forward_interface {label} must be a nonempty list")
    result, names, offset = [], set(), 0
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"invalid forward_interface {label} port")
        shape = item.get("shape")
        if (not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 8
                or any(type(n) is not int or not 1 <= n <= 16384 for n in shape)):
            raise ValueError("forward_interface requires fixed positive logical shapes")
        count = math.prod(shape)
        if (type(item.get("offset")) is not int or item["offset"] != offset
                or type(item.get("width")) is not int or item["width"] != count
                or offset + count > 16384):
            raise ValueError("forward_interface offsets/widths must cover the packed vector exactly")
        if inputs:
            name = item.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("forward_interface input names must be unique and nonempty")
            names.add(name)
        result.append(deepcopy(item))
        offset += count
    if type(width) is not int or offset != width:
        raise ValueError(f"forward_interface {label} width does not match packed IR")
    return tuple(result)


class ForwardModule(nn.Module):
    """Fixed logical inputs and tuple outputs with one registered packed base.

    Pass all inputs positionally or all by name, with exact logical shapes and
    the base model's dtype/device. Implicit batches and mixed argument styles
    are rejected. Flatten/cat and slice/reshape preserve autograd, including
    repeated output fields. Parameters register only under ``base.`` and keep
    their identity across calls. Ordinary optimizers and bind_optimizer work.
    SDK save/load preserves the wrapper and this state_dict layout.
    """

    def __init__(self, base):
        super().__init__()
        definition = base.definition
        interface = definition.get("forward_interface")
        if not isinstance(interface, dict) or definition.get("kind") != "forward":
            raise ValueError("forward_interface requires an ordinary forward IR model")
        packed = definition["interface"]
        if (interface.get("tuple") is not True
                or type(interface.get("input")) is not int
                or type(interface.get("output")) is not int
                or interface["input"] != packed["input"]
                or interface["output"] != packed["output"]):
            raise ValueError("forward_interface tuple marker and total widths must match packed IR")
        if definition["program"].get("loss") is not None:
            raise ValueError("tuple forward does not support source fn loss; compute host loss from outputs")
        self._inputs = _ports(interface.get("inputs"), inputs=True, width=definition["interface"]["input"])
        self._outputs = _ports(interface.get("outputs"), inputs=False, width=definition["interface"]["output"])
        self.base = base
        self.parameter_names = {name: "base." + attr for name, attr in base.parameter_names.items()}

    @property
    def definition(self):
        """The complete compiler export, shared with base for SDK persistence."""
        return self.base.definition

    @property
    def input_shapes(self):
        return {port["name"]: tuple(port["shape"]) for port in self._inputs}

    @property
    def output_shapes(self):
        return tuple(tuple(port["shape"]) for port in self._outputs)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Checkpoint assign=True restores saved precision but the diagnostic
        # anchor is nonpersistent. Align it with restored storage, not the
        # schema-only constructor's default f32 placeholder.
        raw = unwrap_raw_module(self)
        anchor = raw.get_buffer("_wgt_gradient_anchor")
        storage = next(raw.parameters(), None)
        if storage is None:
            storage = next((b for name, b in raw.named_buffers()
                            if name != "_wgt_gradient_anchor" and b.is_floating_point()), anchor)
        raw._buffers["_wgt_gradient_anchor"] = anchor.to(
            device=storage.device, dtype=storage.dtype).detach().requires_grad_(True)
        return self

    def bind_optimizer(self, optimizer):
        from .training import bind_optimizer
        return bind_optimizer(self, optimizer)

    def forward(self, *inputs, **kwargs):
        if inputs and kwargs:
            raise TypeError("pass forward inputs either positionally or by name, not both")
        if kwargs:
            expected = {port["name"] for port in self._inputs}
            if set(kwargs) != expected:
                raise TypeError(f"forward inputs must exactly match {sorted(expected)}")
            inputs = tuple(kwargs[port["name"]] for port in self._inputs)
        if len(inputs) != len(self._inputs):
            raise TypeError(f"forward requires {len(self._inputs)} inputs")
        anchor = self.base.get_buffer("_wgt_gradient_anchor")
        flattened = []
        for value, port in zip(inputs, self._inputs):
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise TypeError(f"forward input {port['name']} must be a dense torch.Tensor")
            if tuple(value.shape) != tuple(port["shape"]):
                raise ValueError(f"forward input {port['name']} requires logical shape {tuple(port['shape'])}")
            if value.device != anchor.device or value.dtype != anchor.dtype:
                raise ValueError(f"forward input {port['name']} dtype/device must match the base model")
            flattened.append(value.reshape(-1))
        output = self.base(torch.cat(flattened))
        return tuple(output.narrow(0, port["offset"], port["width"]).reshape(tuple(port["shape"]))
                     for port in self._outputs)


def wrap_forward(base):
    """Apply exported logical ports, preserving ordinary models without metadata."""
    if isinstance(base, ForwardModule) or base.definition.get("forward_interface") is None:
        return base
    return ForwardModule(base)


def unwrap_raw_module(module):
    """Return the original packed module below a logical port adapter."""
    return module.base if isinstance(module, ForwardModule) else module


# The port adapter and ordinary pure-forward wrapper are the same module type.
PortModule = ForwardModule
