"""Lower compiler-produced model plans to ordinary PyTorch FX modules.

The Rust compiler owns WGT syntax and shape inference. This module translates
its explicit tensor instructions; model execution and differentiation use torch.
"""

from __future__ import annotations

from copy import deepcopy
import keyword
import math
import operator
import re
from typing import Any

import torch
from torch import fx, nn
from torch.nn import functional as F

from . import ops
from .initialization import checked_bindings, materialize


_MAX_ELEMENTS = 16_777_216
_UNARY = {
    "detach": torch.detach,
    "exp": torch.exp,
    "log": ops.log,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "neg": torch.neg,
    "sqrt": ops.sqrt,
    "validate_probabilities": ops.probabilities,
}
_BINARY = {"add": torch.add, "mul": torch.mul, "minimum": torch.minimum,
           "ge": ops.ge, "div": ops.div}
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
    reserved = {"definition", "input_shape", "output_shape", "parameter_names", "meta", "loss",
                "learn_domains", "max_work", "memory_bytes", "bind_optimizer", "project_parameters_",
                "revision", "last_execution", "run", "evaluate"}
    if (not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_")
            or name in reserved or hasattr(root, name) or hasattr(fx.GraphModule, name)):
        buffers = sum(name != "_wgt_gradient_anchor" for name in root._buffers)
        candidate = f"{prefix}_{len(root._parameters) + buffers + len(root._modules)}"
    while hasattr(root, candidate):
        candidate += "_"
    return candidate


def _initialization_random(seed: int):
    """Native SplitMix64 draws; independent of PyTorch and Python global RNGs."""
    mask = 2**64 - 1
    while True:
        seed = (seed + 0x9E3779B97F4A7C15) & mask
        value = seed
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        yield ((value ^ (value >> 31)) >> 11) / 2**53


def register_storage(root: nn.Module, specs: list[dict], modules: list[dict] | None = None,
                     bindings: dict[str, torch.Tensor] | None = None, *, schema_only: bool = False) -> dict[str, str]:
    """Register native storage once, restoring compiler-described submodules."""
    owners: dict[str, tuple[nn.Module, str, str]] = {}
    containers: dict[str, tuple[str, nn.ModuleList]] = {}
    for module in modules or []:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\[([0-9]+)\]", module["path"])
        if match is None or int(match[2]) >= 256:
            raise ValueError("invalid compiled submodel path")
        name, index = match[1], int(match[2])
        if name not in containers:
            attribute = _attribute(root, name, "submodels")
            container = nn.ModuleList()
            root.add_module(attribute, container)
            containers[name] = (attribute, container)
        attribute, container = containers[name]
        while len(container) <= index:
            container.append(nn.Module())
        child = container[index]
        child.model_name = module["model"]
        for local, original in module["parameters"].items():
            if original in owners:
                raise ValueError("duplicate compiled parameter owner")
            owners[original] = (child, local, f"{attribute}.{index}.")
    names: dict[str, str] = {}
    for spec in specs:
        original = spec["name"]
        if original in names:
            raise ValueError(f"duplicate parameter {original}")
        owner, local, prefix = owners.get(original, (root, original, ""))
        attribute = _attribute(owner, local, "parameter")
        shape = _shape(spec["shape"], f"parameter {original}")
        value = materialize(spec, bindings, schema_only=schema_only)
        owner.register_parameter(attribute, nn.Parameter(value, requires_grad=not spec.get("frozen", False)))
        names[original] = prefix + attribute
    if owners.keys() - names.keys():
        raise ValueError("compiled submodel references unknown storage")
    return names


class WeightLangGraphModule(fx.GraphModule):
    """A regular FX graph with an optional compiler-defined loss function."""

    def __deepcopy__(self, memo):
        result = super().__deepcopy__(memo)
        # GraphModule's default copy reconstructs its generated code and drops
        # ordinary Python metadata. Preserve the SDK contract around that graph.
        for name, value in self.__dict__.items():
            if name not in result.__dict__:
                setattr(result, name, deepcopy(value, memo))
        result._non_persistent_buffers_set = deepcopy(self._non_persistent_buffers_set, memo)
        result.__class__.__name__ = self.__class__.__name__
        return result

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        ops.reset_gradient_anchor(self)
        return self

    def loss(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        from .program import model_loss
        return model_loss(self, prediction, target, **kwargs)

    def project_parameters_(self):
        return self

    def bind_optimizer(self, optimizer: torch.optim.Optimizer):
        from .training import bind_optimizer
        return bind_optimizer(self, optimizer)


class _Builder:
    def __init__(self, input_shape: tuple[int, ...], bindings: dict[str, torch.Tensor] | None = None,
                 *, schema_only: bool = False):
        self.root = nn.Module()
        self.root.register_buffer("_wgt_gradient_anchor", torch.zeros((), requires_grad=True), persistent=False)
        self.graph = fx.Graph()
        self.parameter_names: dict[str, str] = {}
        self.input = self.graph.placeholder("x")
        self.input.meta["wgt_shape"] = input_shape
        self.anchor = self.graph.get_attr("_wgt_gradient_anchor")
        self.input = self.graph.call_function(ops.check_input, (self.input, input_shape, self.anchor))
        self.input.meta["wgt_shape"] = input_shape
        self._batch_shape: fx.Node | None = None
        self.input_rank = len(input_shape)
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.values: dict[str, fx.Node] = {}
        self.batched: dict[str, bool] = {}
        self.parameters: dict[str, tuple[dict[str, Any], fx.Node]] = {}
        self.bindings = bindings
        self.schema_only = schema_only

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
            if node.op == "call_function" and getattr(node.target, "__module__", None) == ops.__name__:
                # Emit fx.wrap in the generated module so a surrounding Python
                # model can itself be symbolically traced without executing checks.
                node.meta["is_wrapped"] = True
        self.graph.lint()
        name = definition.get("name", "WeightLangModel")
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            name = "WeightLangModel"
        model = WeightLangGraphModule(self.root, self.graph, name)
        # FX copies used buffers as ordinary persistent attributes by default.
        model.register_buffer("_wgt_gradient_anchor", self.root._wgt_gradient_anchor, persistent=False)
        # FX normally retains only attributes used in the graph. Keep declared
        # unused weights, too, so state_dict has the complete model contract.
        for name, parameter in self.root.named_parameters(recurse=False):
            if name not in model._parameters:
                model.register_parameter(name, parameter)
        for name, module in self.root.named_children():
            model.add_module(name, module)
        model.definition = deepcopy(definition)
        model.input_shape = (definition["interface"]["input"],)
        model.output_shape = output_shape
        model.parameter_names = dict(self.parameter_names)
        from .program import DEFAULT_MAX_WORK, DEFAULT_MEMORY_BYTES, ProgramExecutor
        loss = definition.get("program", {}).get("loss")
        model._loss_executor = ProgramExecutor(loss) if loss is not None else None
        model.max_work, model.memory_bytes = DEFAULT_MAX_WORK, DEFAULT_MEMORY_BYTES
        model.learn_domains = {}
        return model

    def sequential(self, definition: dict) -> fx.GraphModule:
        plan = definition["graph"]
        width = _positive(plan["input"], "graph input", 16384)
        if width != definition["interface"]["input"]:
            raise ValueError("graph input does not match model interface")
        seed = plan["seed"]
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("graph seed must be an unsigned 64-bit integer")
        generator = _initialization_random(seed)
        declared = definition.get("graph_parameters", plan.get("parameters") or [])
        initializers = {spec["name"]: spec for spec in declared}
        if len(initializers) != len(declared):
            raise ValueError("duplicate native graph initializer")
        used_initializers = set()
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
                # Register the Linear layout without calling nn.Linear's random
                # reset_parameters, including during checkpoint schema recovery.
                module = ops.Linear.__new__(ops.Linear)
                nn.Module.__init__(module)
                module.in_features, module.out_features = width, output
                module.register_parameter("weight", None)
                module.register_parameter("bias", None)
                if "graph_parameters" in definition or plan.get("parameters") is not None or self.bindings is not None or self.schema_only:
                    for part, shape in [("weight", (output, width)), *(([("bias", (output,))]) if layer["bias"] else [])]:
                        key = f"{name}.{part}"
                        spec = initializers.get(key, {"name": key, "shape": list(shape)})
                        if tuple(spec["shape"]) != shape:
                            raise ValueError("native graph initializer shape mismatch")
                        setattr(module, part, nn.Parameter(materialize(spec, self.bindings, schema_only=self.schema_only), requires_grad=not spec.get("frozen", False)))
                        if key in initializers:
                            used_initializers.add(key)
                else:
                    # Older v1 exports stored a seed only. Reproduce the native
                    # initializer instead of substituting PyTorch's RNG stream.
                    scale = math.sqrt(6 / (width + output))
                    data = [(next(generator) * 2 - 1) * scale for _ in range(output * width)]
                    module.weight = nn.Parameter(torch.tensor(data, dtype=torch.float32).reshape(output, width))
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
            arguments = (value, self.anchor) if op == "linear" else (value,)
            value = self.graph.call_module(attribute, arguments)
            value.meta.update(wgt_shape=(output,), wgt_name=name)
            value = self.graph.call_function(ops.check_finite, (value, self.anchor))
            width = output
        if width != definition["interface"]["output"]:
            raise ValueError("graph output does not match model interface")
        if used_initializers != initializers.keys():
            raise ValueError("unused native graph initializers")
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
        self.parameter_names = register_storage(self.root, program["parameters"], definition.get("modules"), self.bindings,
                                                schema_only=self.schema_only)
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
            raw = self.graph.get_attr(self.parameter_names[name])
            self.parameters[name] = (parameter, self.graph.call_function(ops.check_finite, (raw, self.anchor)))
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
        input_keys = ["input", "left", "right", "gamma", "beta", "scalar", "indices", "table"]
        if op == "dynamic_slice":
            input_keys.append("start")
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
            node = self.graph.call_function(ops.linear, (value("input"), weight_node, bias_node))
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
            node = self.graph.call_function(ops.matmul, (value("left"), value("right")))
        elif op in ("softmax", "log_softmax", "softmax_axis"):
            self.equal(shape, source_shape("input"), op)
            axis = self.axis(item["axis"], shape) if op == "softmax_axis" else -1
            if op != "softmax_axis" and len(shape) != 1:
                raise ValueError(f"{op} requires a vector")
            function = ops.log_softmax if op == "log_softmax" else ops.softmax
            arguments = (value("input"), axis, True) if op == "softmax_axis" else (value("input"), axis)
            node = self.graph.call_function(function, arguments)
        elif op == "layer_norm":
            self.equal(shape, source_shape("input"), op)
            axis = self.axis(item["axis"], shape)
            extent = shape[axis]
            self.equal(source_shape("gamma"), (extent,), op)
            self.equal(source_shape("beta"), (extent,), op)
            epsilon = _finite(item["epsilon"], "layer_norm epsilon")
            if epsilon <= 0:
                raise ValueError("layer_norm epsilon must be positive")
            affine_shape = tuple(extent if i == item["axis"] else 1 for i in range(len(shape)))
            gamma = self.reshape(value("gamma"), affine_shape, self.batched[item["gamma"]])
            beta = self.reshape(value("beta"), affine_shape, self.batched[item["beta"]])
            node = self.graph.call_function(ops.layer_norm, (value("input"), gamma, beta, axis, epsilon))
        elif op == "scale_by":
            self.equal(shape, source_shape("input"), op)
            self.equal(source_shape("scalar"), (1,), op)
            node = self.graph.call_function(ops.scale_by, (value("input"), value("scalar"), len(shape)))
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
            node = self.graph.call_function(ops.sum if op == "sum" else ops.mean, (value("input"), axes))
            node = self.graph.call_method("unsqueeze", (node, -1))
        elif op == "embedding":
            table, ids, padding = source_shape("table"), source_shape("indices"), item.get("padding_idx")
            if len(table) != 2 or len(ids) != 1:
                raise ValueError("embedding requires rank-2 table and rank-1 indices")
            if padding is not None and (type(padding) is not int or not 0 <= padding < table[0]):
                raise ValueError("invalid embedding padding_idx")
            self.equal(shape, (ids[0], table[1]), op)
            node = self.graph.call_function(ops.embedding, (value("table"), value("indices"), padding))
        elif op == "gather":
            previous, ids, axis = source_shape("input"), source_shape("indices"), item["axis"]
            if type(axis) is not int or not 0 <= axis < len(previous) or len(ids) != 1:
                raise ValueError("invalid gather axis/indices")
            expected = list(previous)
            expected[axis] = ids[0]
            self.equal(shape, tuple(expected), op)
            node = self.graph.call_function(ops.gather, (value("input"), value("indices"), axis, len(previous)))
        elif op == "dynamic_slice":
            previous, length = source_shape("input"), item["length"]
            if len(previous) != 1 or source_shape("start") != (1,) or type(length) is not int or not 0 < length <= previous[0]:
                raise ValueError("invalid dynamic_slice contract")
            self.equal(shape, (length,), op)
            node = self.graph.call_function(ops.dynamic_slice, (value("input"), value("start"), length))
        elif op == "concat":
            left, right, axis = source_shape("left"), source_shape("right"), item["axis"]
            if type(axis) is not int or not 0 <= axis < len(left) or len(left) != len(right) or any(a != b for i, (a,b) in enumerate(zip(left,right)) if i != axis):
                raise ValueError("invalid concat contract")
            expected = list(left)
            expected[axis] += right[axis]
            self.equal(shape, tuple(expected), op)
            node = self.graph.call_function(ops.concat, (value("left"), value("right"), axis, len(left)))
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
                node = self.graph.call_function(ops.conv, (
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
                function = F.max_pool2d if op == "max_pool2d" else ops.avg_pool2d
                node = self.graph.call_function(function, (flattened, kernel, stride))
            node = self.reshape(node, shape, batched)
        else:
            raise NotImplementedError(f"PyTorch backend does not support instruction {op!r}")
        node.meta.update(wgt_shape=shape, wgt_name=out, wgt_op=op)
        node = self.graph.call_function(ops.check_finite, (node, self.anchor))
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


def build_module(definition: dict[str, Any], *, bindings: dict[str, torch.Tensor] | None = None,
                 _schema_only: bool = False) -> nn.Module:
    """Instantiate a compiler export on CPU, using float32 PyTorch parameters."""
    if not isinstance(definition, dict) or definition.get("format") != "weightlang.torch.v1":
        raise ValueError("expected a weightlang.torch.v1 compiler export")
    if bindings is not None:
        bindings = checked_bindings(definition, bindings)
    try:
        if definition["kind"] == "operations":
            from .operations import OperationsModule
            return OperationsModule(definition, bindings=bindings, _schema_only=_schema_only)
        interface = definition["interface"]
        shape = (_positive(interface["input"], "input width", 16384),)
        _positive(interface["output"], "output width", 16384)
        if interface.get("output_type", "f32") not in ("f32", "logits", "probabilities"):
            raise ValueError("unknown model output type")
        builder = _Builder(shape, bindings, schema_only=_schema_only)
        if definition["kind"] == "sequential":
            return builder.sequential(definition)
        if definition["kind"] == "forward":
            from .program import ProgramExecutor, ProgramModule
            # Validate the complete program even when FX handles its tensor path.
            ProgramExecutor(definition["program"])
            if definition["program"].get("observations"):
                return ProgramModule(definition, bindings=bindings, _schema_only=_schema_only)
            instructions = definition["program"]["instructions"]
            if any(item["op"] in _UNSUPPORTED or (item["op"] == "return" and index != len(instructions) - 1)
                   for index, item in enumerate(instructions)):
                return ProgramModule(definition, bindings=bindings, _schema_only=_schema_only)
            if any(spec.get("learn", {"kind": "float"}) not in (None, {"kind": "float"})
                   for spec in definition["program"]["parameters"]):
                return ProgramModule(definition, bindings=bindings, _schema_only=_schema_only)
            try:
                return builder.forward(definition)
            except NotImplementedError:
                return ProgramModule(definition, bindings=bindings, _schema_only=_schema_only)
        raise ValueError(f"unsupported model plan {definition['kind']!r}")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"malformed WeightLang compiler export: {error}") from error
