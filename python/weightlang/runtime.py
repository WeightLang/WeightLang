"""Lower compiler-produced model plans to ordinary PyTorch FX modules.

The Rust compiler owns WGT syntax and shape inference. This module translates
its explicit tensor instructions; model execution and differentiation use torch.
"""

from __future__ import annotations

from copy import deepcopy
import keyword
import math
import operator
from typing import Any

import torch
from torch import fx, nn
from torch.nn import functional as F

from . import ops


_MAX_ELEMENTS = 16_777_216
_UNARY = {
    "detach": torch.detach,
    "exp": torch.exp,
    "log": torch.log,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "neg": torch.neg,
    "validate_probabilities": ops.probabilities,
}
_BINARY = {"add": torch.add, "mul": torch.mul, "minimum": torch.minimum}
_UNSUPPORTED = {
    "write", "memory_alloc", "memory_free", "memory_read", "memory_write",
    "resource_state", "branch", "jump", "label", "inference_skip", "gate", "route",
}


def _positive(value: Any, label: str, maximum: int = _MAX_ELEMENTS) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in 1..{maximum}")
    return value


def _shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 8:
        raise ValueError(f"{label} must have 1..8 dimensions")
    shape = tuple(_positive(item, label) for item in value)
    if math.prod(shape) > _MAX_ELEMENTS:
        raise ValueError(f"{label} exceeds {_MAX_ELEMENTS} elements")
    return shape


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _tensor(data: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if not isinstance(data, list) or len(data) != math.prod(shape):
        raise ValueError(f"{label} initializer does not match shape {shape}")
    # Check before allocation, including float64 values that overflow float32.
    limit = torch.finfo(torch.float32).max
    if any(abs(_finite(value, label)) > limit for value in data):
        raise ValueError(f"{label} initializer exceeds float32 range")
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def _attribute(root: nn.Module, name: str, prefix: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("model names must be nonempty strings")
    candidate = name
    reserved = {"definition", "input_shape", "output_shape", "parameter_names", "meta"}
    if (not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_")
            or name in reserved or hasattr(root, name) or hasattr(fx.GraphModule, name)):
        candidate = f"{prefix}_{len(root._parameters) + len(root._buffers) + len(root._modules)}"
    while hasattr(root, candidate):
        candidate += "_"
    return candidate


class _Builder:
    def __init__(self, input_shape: tuple[int, ...]):
        self.root = nn.Module()
        self.graph = fx.Graph()
        self.parameter_names: dict[str, str] = {}
        self.input = self.graph.placeholder("x")
        self.input.meta["wgt_shape"] = input_shape
        self.input = self.graph.call_function(ops.check_input, (self.input, input_shape))
        self.input.meta["wgt_shape"] = input_shape
        self._batch_shape: fx.Node | None = None
        self.input_rank = len(input_shape)
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.values: dict[str, fx.Node] = {}
        self.batched: dict[str, bool] = {}
        self.parameters: dict[str, tuple[dict[str, Any], fx.Node]] = {}

    @property
    def batch_shape(self) -> fx.Node:
        if self._batch_shape is None:
            self._batch_shape = self.graph.call_function(operator.getitem, (
                self.graph.call_function(getattr, (self.input, "shape")), slice(None, -self.input_rank),
            ))
        return self._batch_shape

    def reshape(self, node: fx.Node, shape: tuple[int, ...], batched: bool) -> fx.Node:
        dimensions = self.graph.call_function(operator.add, (self.batch_shape, shape)) if batched else shape
        return self.graph.call_method("reshape", (node, dimensions))

    def finish(self, output: fx.Node, output_shape: tuple[int, ...], batched: bool, definition: dict) -> fx.GraphModule:
        if not batched:
            dimensions = self.graph.call_function(operator.add, (self.batch_shape, output_shape))
            output = self.graph.call_method("expand", (output, dimensions))
        if definition["interface"].get("output_type", "f32") == "probabilities":
            output = self.graph.call_function(ops.probabilities, (output,))
        self.graph.output(output)
        for node in self.graph.nodes:
            if node.op == "call_function" and node.target in (ops.check_input, ops.clip, ops.probabilities):
                # Emit fx.wrap in the generated module so a surrounding Python
                # model can itself be symbolically traced without executing checks.
                node.meta["is_wrapped"] = True
        self.graph.lint()
        name = definition.get("name", "WeightLangModel")
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            name = "WeightLangModel"
        model = fx.GraphModule(self.root, self.graph, name)
        # FX normally retains only attributes used in the graph. Keep declared
        # unused weights, too, so state_dict has the complete model contract.
        for name, parameter in self.root.named_parameters(recurse=False):
            if name not in model._parameters:
                model.register_parameter(name, parameter)
        model.definition = deepcopy(definition)
        model.input_shape = (definition["interface"]["input"],)
        model.output_shape = output_shape
        model.parameter_names = dict(self.parameter_names)
        return model

    def sequential(self, definition: dict) -> fx.GraphModule:
        plan = definition["graph"]
        width = _positive(plan["input"], "graph input", 16384)
        if width != definition["interface"]["input"]:
            raise ValueError("graph input does not match model interface")
        seed = plan["seed"]
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("graph seed must be an unsigned 64-bit integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if not isinstance(plan["layers"], list) or not plan["layers"]:
            raise ValueError("sequential model requires layers")
        value, seen, count = self.input, set(), 0
        for layer in plan["layers"]:
            name, op = layer["name"], layer["op"]
            if name in seen:
                raise ValueError(f"duplicate layer: {name}")
            seen.add(name)
            if layer["input"] != width:
                raise ValueError(f"layer {name} input width mismatch")
            output = _positive(layer["output"], f"layer {name} output", 16384)
            if type(layer["bias"]) is not bool:
                raise ValueError(f"layer {name} bias must be boolean")
            if op == "linear":
                count += width * output + (output if layer["bias"] else 0)
                if count > _MAX_ELEMENTS:
                    raise ValueError("model parameter element limit exceeded")
                # Meta construction avoids consuming the caller's RNG. An
                # isolated torch generator initializes Xavier weights on CPU.
                module = nn.Linear(width, output, bias=layer["bias"], device="meta")
                module.weight = nn.Parameter(torch.empty(output, width, dtype=torch.float32))
                nn.init.xavier_uniform_(module.weight, generator=generator)
                if layer["bias"]:
                    module.bias = nn.Parameter(torch.zeros(output, dtype=torch.float32))
            else:
                activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "identity": nn.Identity}
                if op not in activations:
                    raise NotImplementedError(f"PyTorch backend does not support layer {op!r}")
                if output != width or layer["bias"]:
                    raise ValueError(f"activation {name} must preserve width and have no bias")
                module = activations[op]()
            attribute = _attribute(self.root, name, "layer")
            self.root.add_module(attribute, module)
            for parameter, _ in module.named_parameters():
                self.parameter_names[f"{name}.{parameter}"] = f"{attribute}.{parameter}"
            value = self.graph.call_module(attribute, (value,))
            value.meta.update(wgt_shape=(output,), wgt_name=name)
            width = output
        if width != definition["interface"]["output"]:
            raise ValueError("graph output does not match model interface")
        return self.finish(value, (width,), True, definition)

    def forward(self, definition: dict) -> fx.GraphModule:
        program = definition["program"]
        if type(program["version"]) is not int or program["version"] not in (1, 2, 3):
            raise ValueError("unsupported adaptive IR version; recompile with wgt")
        for register in program["registers"]:
            name = register["name"]
            if not isinstance(name, str) or not name or name in self.shapes:
                raise ValueError(f"invalid or duplicate register {name!r}")
            shape = _shape(register["shape"], f"register {name}")
            if type(register["len"]) is not int or math.prod(shape) != register["len"]:
                raise ValueError(f"register {name} length does not match shape")
            self.shapes[name] = shape
        input_name = program["input"]
        if self.shapes.get(input_name) != (definition["interface"]["input"],):
            raise ValueError("forward input does not match model interface")
        self.values[input_name], self.batched[input_name] = self.input, True
        count = 0
        for parameter in program["parameters"]:
            name = parameter["name"]
            if name in self.parameters:
                raise ValueError(f"duplicate parameter {name}")
            domain = parameter.get("learn")
            if domain is not None and domain != {"kind": "float"}:
                raise NotImplementedError(f"parameter {name}: constrained learn domains require native optimizer projection")
            shape = _shape(parameter["shape"], f"parameter {name}")
            count += math.prod(shape)
            if count > _MAX_ELEMENTS:
                raise ValueError("model parameter element limit exceeded")
            frozen = parameter.get("frozen", False)
            if type(frozen) is not bool:
                raise ValueError(f"parameter {name} frozen must be boolean")
            attribute = _attribute(self.root, name, "parameter")
            tensor = _tensor(parameter["data"], shape, f"parameter {name}")
            self.root.register_parameter(attribute, nn.Parameter(tensor, requires_grad=not frozen))
            self.parameter_names[name] = attribute
            self.parameters[name] = (parameter, self.graph.get_attr(attribute))
        instructions = program["instructions"]
        if not isinstance(instructions, list) or not instructions:
            raise ValueError("forward model requires instructions")
        for index, instruction in enumerate(instructions):
            op = instruction["op"]
            if op in _UNSUPPORTED:
                raise NotImplementedError(f"PyTorch backend does not support {op!r}; use a pure tensor forward definition")
            if op == "return":
                if index != len(instructions) - 1:
                    raise ValueError("PyTorch forward requires one final return")
                name = instruction["value"]
                if self.shapes[name] != (definition["interface"]["output"],):
                    raise ValueError("forward output does not match model interface")
                return self.finish(self.values[name], self.shapes[name], self.batched[name], definition)
            self.instruction(instruction)
        raise ValueError("forward model is missing a return instruction")

    def instruction(self, item: dict) -> None:
        op, out = item["op"], item["out"]
        shape = self.shapes[out]
        if out == next(iter(self.values)):
            raise ValueError("forward instructions cannot overwrite model input")
        input_keys = ["input", "left", "right", "gamma", "beta"]
        if op == "conv":
            input_keys += ["kernel", "bias"]
        inputs = [item[key] for key in input_keys if key in item]
        batched = any(self.batched[name] for name in inputs)
        value = lambda key: self.values[item[key]]
        source_shape = lambda key: self.shapes[item[key]]
        if op == "read":
            parameter, node = self.parameters[item["parameter"]]
            if math.prod(parameter["shape"]) != math.prod(shape):
                raise ValueError("parameter read shape mismatch")
            node = self.reshape(node, shape, False)
        elif op == "constant":
            attribute = _attribute(self.root, out, "constant")
            self.root.register_buffer(attribute, _tensor(item["data"], shape, f"constant {out}"))
            node = self.graph.get_attr(attribute)
        elif op == "linear":
            weight, weight_node = self.parameters[item["weight"]]
            if tuple(weight["shape"]) != (math.prod(shape), math.prod(source_shape("input"))):
                raise ValueError("linear weight must have shape [output, input]")
            if len(shape) != 1 or len(source_shape("input")) != 1:
                raise ValueError("linear requires vector registers")
            bias = item.get("bias")
            bias_node = None
            if bias is not None:
                bias_spec, bias_node = self.parameters[bias]
                if tuple(bias_spec["shape"]) != shape:
                    raise ValueError("linear bias shape mismatch")
            node = self.graph.call_function(F.linear, (value("input"), weight_node, bias_node))
        elif op in _UNARY:
            self.equal(shape, source_shape("input"), op)
            node = self.graph.call_function(_UNARY[op], (value("input"),))
        elif op in _BINARY:
            self.equal(shape, source_shape("left"), op)
            self.equal(shape, source_shape("right"), op)
            node = self.graph.call_function(_BINARY[op], (value("left"), value("right")))
        elif op == "reshape":
            if math.prod(shape) != math.prod(source_shape("input")):
                raise ValueError("reshape must preserve element count")
            node = self.reshape(value("input"), shape, batched)
        elif op in ("transpose", "permute"):
            previous = source_shape("input")
            axes = (1, 0) if op == "transpose" else tuple(item["axes"])
            if len(axes) != len(previous) or sorted(axes) != list(range(len(previous))):
                raise ValueError(f"{op} requires a valid permutation of model axes")
            self.equal(shape, tuple(previous[axis] for axis in axes), op)
            rank = len(axes)
            node = self.graph.call_function(torch.movedim, (
                value("input"), tuple(axis - rank for axis in axes), tuple(range(-rank, 0)),
            ))
        elif op in ("matmul", "batch_matmul"):
            left, right = source_shape("left"), source_shape("right")
            rank = 3 if op == "batch_matmul" else 2
            if len(left) != rank or len(right) != rank or left[-1] != right[-2] or left[:-2] != right[:-2]:
                raise ValueError(f"{op} shape mismatch")
            self.equal(shape, (*left[:-2], left[-2], right[-1]), op)
            node = self.graph.call_function(torch.matmul, (value("left"), value("right")))
        elif op in ("softmax", "log_softmax", "softmax_axis"):
            self.equal(shape, source_shape("input"), op)
            axis = self.axis(item["axis"], shape) if op == "softmax_axis" else -1
            if op != "softmax_axis" and len(shape) != 1:
                raise ValueError(f"{op} requires a vector")
            function = torch.log_softmax if op == "log_softmax" else torch.softmax
            node = self.graph.call_function(function, (value("input"), axis))
        elif op == "layer_norm":
            self.equal(shape, source_shape("input"), op)
            axis = self.axis(item["axis"], shape)
            extent = shape[axis]
            self.equal(source_shape("gamma"), (extent,), op)
            self.equal(source_shape("beta"), (extent,), op)
            epsilon = _finite(item["epsilon"], "layer_norm epsilon")
            if epsilon <= 0:
                raise ValueError("layer_norm epsilon must be positive")
            moved = self.graph.call_function(torch.movedim, (value("input"), axis, -1))
            normalized = self.graph.call_function(F.layer_norm, (moved, (extent,), None, None, epsilon))
            normalized = self.graph.call_function(torch.movedim, (normalized, -1, axis))
            affine_shape = tuple(extent if i == item["axis"] else 1 for i in range(len(shape)))
            gamma = self.reshape(value("gamma"), affine_shape, self.batched[item["gamma"]])
            beta = self.reshape(value("beta"), affine_shape, self.batched[item["beta"]])
            node = self.graph.call_function(torch.add, (
                self.graph.call_function(torch.mul, (normalized, gamma)), beta,
            ))
        elif op in ("scale", "clip"):
            self.equal(shape, source_shape("input"), op)
            if op == "scale":
                node = self.graph.call_function(torch.mul, (value("input"), _finite(item["factor"], "scale factor")))
            else:
                low, high = _finite(item["low"], "clip low"), _finite(item["high"], "clip high")
                if low > high:
                    raise ValueError("clip low must not exceed high")
                node = self.graph.call_function(ops.clip, (value("input"), low, high))
        elif op in ("sum", "mean"):
            self.equal(shape, (1,), op)
            axes = tuple(range(-len(source_shape("input")), 0))
            node = self.graph.call_function(torch.sum if op == "sum" else torch.mean, (value("input"), axes))
            node = self.graph.call_method("unsqueeze", (node, -1))
        elif op == "slice":
            previous = source_shape("input")
            start = item["start"]
            if len(shape) != 1 or len(previous) != 1 or type(start) is not int or start < 0 or start + shape[0] > previous[0]:
                raise ValueError("slice is outside the input vector")
            node = self.graph.call_function(operator.getitem, (value("input"), (Ellipsis, slice(start, start + shape[0]))))
        elif op in ("conv", "max_pool2d", "avg_pool2d"):
            previous = source_shape("input")
            flattened = self.graph.call_method("reshape", (value("input"), (-1, *previous)))
            if op == "conv":
                if self.batched[item["kernel"]] or self.batched[item["bias"]]:
                    raise NotImplementedError("convolution kernels and biases must be shared across examples")
                rank = len(previous) - 1
                if rank not in (1, 2, 3):
                    raise ValueError("convolution requires 1..3 spatial axes")
                kernel, bias = source_shape("kernel"), source_shape("bias")
                groups = _positive(item["groups"], "convolution groups", 16384)
                stride = self.dimensions(item["stride"], rank, "convolution stride")
                padding = self.dimensions(item["padding"], rank, "convolution padding", 0)
                dilation = self.dimensions(item["dilation"], rank, "convolution dilation")
                if (len(kernel) != rank + 2 or previous[0] % groups or kernel[0] % groups
                        or kernel[1] != previous[0] // groups or bias != (kernel[0],)):
                    raise ValueError("convolution kernel, bias or groups shape mismatch")
                expected = (kernel[0], *(1 + (size + 2 * pad - dil * (extent - 1) - 1) // step
                            for size, extent, pad, dil, step in zip(previous[1:], kernel[2:], padding, dilation, stride)))
                self.equal(shape, expected, op)
                node = self.graph.call_function((F.conv1d, F.conv2d, F.conv3d)[rank - 1], (
                    flattened, value("kernel"), value("bias"), stride, padding, dilation, groups,
                ))
            else:
                if len(previous) != 3:
                    raise ValueError("pool2d requires [channels, height, width]")
                kernel = self.dimensions(item["kernel"], 2, "pool2d kernel")
                stride = self.dimensions(item["stride"], 2, "pool2d stride")
                expected = (previous[0], *(1 + (size - extent) // step
                            for size, extent, step in zip(previous[1:], kernel, stride)))
                self.equal(shape, expected, op)
                function = F.max_pool2d if op == "max_pool2d" else F.avg_pool2d
                node = self.graph.call_function(function, (flattened, kernel, stride))
            node = self.reshape(node, shape, batched)
        else:
            raise NotImplementedError(f"PyTorch backend does not support instruction {op!r}")
        node.meta.update(wgt_shape=shape, wgt_name=out, wgt_op=op)
        self.values[out], self.batched[out] = node, batched

    @staticmethod
    def equal(left: tuple[int, ...], right: tuple[int, ...], op: str) -> None:
        if left != right:
            raise ValueError(f"{op} shape mismatch: {left} and {right}")

    @staticmethod
    def axis(axis: Any, shape: tuple[int, ...]) -> int:
        if type(axis) is not int or not 0 <= axis < len(shape):
            raise ValueError("tensor axis is outside the model rank")
        return axis - len(shape)

    @staticmethod
    def dimensions(value: Any, rank: int, label: str, minimum: int = 1) -> tuple[int, ...]:
        if (not isinstance(value, list) or len(value) != rank
                or any(type(item) is not int or not minimum <= item <= 16384 for item in value)):
            raise ValueError(f"{label} requires {rank} integers in {minimum}..16384")
        return tuple(value)


def build_module(definition: dict[str, Any]) -> fx.GraphModule:
    """Instantiate a compiler export on CPU, using float32 PyTorch parameters."""
    if not isinstance(definition, dict) or definition.get("format") != "weightlang.torch.v1":
        raise ValueError("expected a weightlang.torch.v1 compiler export")
    try:
        interface = definition["interface"]
        shape = (_positive(interface["input"], "input width", 16384),)
        _positive(interface["output"], "output width", 16384)
        if interface.get("output_type", "f32") not in ("f32", "logits", "probabilities"):
            raise ValueError("unknown model output type")
        builder = _Builder(shape)
        if definition["kind"] == "sequential":
            return builder.sequential(definition)
        if definition["kind"] == "forward":
            return builder.forward(definition)
        raise ValueError(f"unsupported model plan {definition['kind']!r}")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"malformed WeightLang compiler export: {error}") from error
