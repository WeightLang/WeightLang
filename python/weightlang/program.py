"""Execute native tensor IR with PyTorch values and transactional storage.

The executor consumes compiler instructions, never WGT source. Tensor math and
its derivatives belong to PyTorch; this module supplies the program counter,
local working memory, and parameter versions defined by the native IR.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import math
from threading import RLock
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from . import ops
from .runtime import _finite, _positive, _shape, register_storage


DEFAULT_MAX_WORK = 1_000_000
DEFAULT_MEMORY_BYTES = 256 * 1024**2

_UNARY = {
    "detach": torch.detach, "exp": torch.exp, "log": ops.log,
    "tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": torch.relu,
    "neg": torch.neg, "sqrt": ops.sqrt,
}
_BINARY = {"add": torch.add, "mul": torch.mul, "minimum": torch.minimum,
           "ge": ops.ge, "div": ops.div}
_GEOMETRY = {"conv", "max_pool2d", "avg_pool2d", "reshape", "transpose", "permute",
             "matmul", "batch_matmul", "layer_norm", "softmax_axis", "scale",
             "ge", "div", "sqrt", "scale_by"}
_FIELDS = {
    **{name: "out input" for name in _UNARY},
    **{name: "out left right" for name in _BINARY},
    "conv": "out input kernel bias stride padding dilation groups",
    "max_pool2d": "out input kernel stride", "avg_pool2d": "out input kernel stride",
    "reshape": "out input", "transpose": "out input", "permute": "out input axes",
    "matmul": "out left right", "batch_matmul": "out left right",
    "layer_norm": "out input gamma beta axis epsilon", "softmax_axis": "out input axis",
    "scale": "out input factor", "scale_by": "out input scalar",
    "constant": "out data", "clip": "out input low high",
    "sum": "out input", "mean": "out input", "log_softmax": "out input",
    "softmax": "out input", "validate_probabilities": "out input",
    "memory_alloc": "out slots width", "memory_free": "memory",
    "memory_read": "out memory address", "memory_write": "memory address value",
    "resource_state": "out", "read": "out parameter", "write": "parameter value",
    "linear": "out input weight bias", "gate": "out probability input",
    "route": "out probability yes no", "inference_skip": "probability when_one target",
    "slice": "out input start", "label": "name", "branch": "condition positive otherwise",
    "jump": "target", "return": "value",
}
SUPPORTED_OPS = frozenset(_FIELDS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _all(condition: torch.Tensor) -> bool:
    return bool(condition.detach().all().item())


def _finite_tensor(value: torch.Tensor, label: str) -> None:
    _require(isinstance(value, torch.Tensor) and value.is_floating_point(), f"{label} must be a floating point tensor")
    _require(_all(torch.isfinite(value)), f"nonfinite {label}")


def _domain(domain: dict | None) -> None:
    if domain is None:
        return
    _require(isinstance(domain, dict), "learn domain must be an object")
    kind = domain.get("kind")
    fields = {"float": {"kind"}, "bool": {"kind"}, "positive": {"kind", "low"},
              "bounded": {"kind", "low", "high"}, "int": {"kind", "low", "high"}}
    _require(kind in fields and set(domain) == fields[kind], "invalid learn domain")
    if kind in {"positive", "bounded", "int"}:
        low = _finite(domain["low"], "learn lower bound")
        if kind == "positive":
            _require(0 <= low < torch.finfo(torch.float32).max, "invalid learn positive bound")
        else:
            high = _finite(domain["high"], "learn upper bound")
            _require(low < high, "invalid learn interval")
            if kind == "int":
                _require(low.is_integer() and high.is_integer() and max(abs(low), abs(high)) <= 16_777_216,
                         "learn int bounds must be exact f32 integers within +/-16777216")


def validate_parameter(value: torch.Tensor, spec: dict, *, discrete: bool = False) -> None:
    """Validate a storage value; discrete=True validates external hard updates."""
    name = spec["name"]
    _require(tuple(value.shape) == tuple(spec["shape"]), f"parameter {name} shape mismatch")
    _finite_tensor(value, f"parameter {name}")
    domain = spec.get("learn")
    _domain(domain)
    if spec.get("probability"):
        _require(tuple(value.shape) == (1,), "prob parameter must be scalar")
        if spec.get("frozen"):
            _require(_all((value >= 0) & (value <= 1)), "frozen prob must be in [0,1]")
    if domain is None or domain["kind"] == "float":
        return
    kind = domain["kind"]
    if kind in {"bounded", "int"}:
        valid = (value >= domain["low"]) & (value <= domain["high"])
        if discrete and kind == "int":
            valid &= value == torch.trunc(value)
    elif kind == "positive":
        valid = value > domain["low"]
    else:
        valid = (value >= 0) & (value <= 1)
        if discrete:
            valid &= (value == 0) | (value == 1)
    _require(_all(valid), f"learn parameter {name} is outside its domain; bind_optimizer() or project_parameters_() after optimizer.step()")


def read_parameter(value: torch.Tensor, spec: dict, differentiable: bool, *, validate: bool = True) -> torch.Tensor:
    """Read projected shadows; bool/int use native hard values and identity STE."""
    if validate:
        validate_parameter(value, spec)
    domain = spec.get("learn")
    if domain is None or domain["kind"] not in {"bool", "int"}:
        return value
    if domain["kind"] == "bool":
        hard = (value >= 0.5).to(value.dtype)
    else:
        # Rust f32::round rounds halfway values away from zero.
        integer = torch.trunc(value)
        hard = integer + torch.sign(value) * ((value - integer).abs() >= 0.5).to(value.dtype)
    return value + (hard - value).detach() if differentiable else hard


def project_value(value: torch.Tensor, domain: dict | None) -> torch.Tensor:
    """Project the optimizer's continuous shadow, without rounding it."""
    _domain(domain)
    _finite_tensor(value, "optimizer update")
    if domain is None or domain["kind"] == "float":
        return value
    if domain["kind"] in {"bounded", "int"}:
        return value.clamp(domain["low"], domain["high"])
    if domain["kind"] == "bool":
        return value.clamp(0, 1)
    low = torch.tensor(domain["low"], dtype=torch.float32)
    native_minimum = torch.nextafter(low, torch.tensor(float("inf"), dtype=torch.float32)).item()
    minimum = value.new_tensor(native_minimum)
    # Preserve the declared f32 boundary in higher precision; for a narrower
    # dtype choose its first representable value that remains strictly above it.
    if minimum.item() <= domain["low"]:
        minimum = torch.nextafter(minimum, value.new_tensor(float("inf")))
    return torch.maximum(value, minimum)


def _probabilities(value: torch.Tensor) -> None:
    _finite_tensor(value, "probabilities")
    _require(_all((value >= 0) & (value <= 1)), "probabilities must each be in [0,1]")
    # Native categorical validation sums f32 storage in f64. A scalar host check
    # also works on devices which do not implement float64 arithmetic.
    total = value.detach().to(device="cpu", dtype=torch.float64).sum().item()
    _require(abs(total - 1) <= 1e-5, "probabilities must sum to 1 within 1e-5")


@dataclass
class ExecutionResult:
    output: torch.Tensor
    parameters: dict[str, torch.Tensor]
    writes: set[str]
    work: int
    peak_memory_bytes: int


class ProgramExecutor:
    """A validated single-example program, independent of registered storage.

    Memory accounting uses the native f32 storage contract. It bounds logical
    registers/rollback/working memory, not PyTorch's allocator or autograd tape.
    """

    def __init__(self, program: dict, parameter_specs: list[dict] | None = None, *, transactional: bool = False):
        self.transactional = transactional
        required = {"version", "input", "registers", "parameters", "instructions"}
        _require(isinstance(program, dict) and required <= program.keys()
                 and program.keys() <= required | {"output_type", "loss"}, "invalid adaptive program fields")
        # Snapshot executable metadata without duplicating full initializers for
        # every entry in a model with several typed operations.
        self.instructions = deepcopy(program["instructions"])
        _require(type(program["version"]) is int and program["version"] in (1, 2, 3), "unsupported adaptive IR version")
        _require(isinstance(self.instructions, list) and 0 < len(self.instructions) <= 100_000, "invalid instruction count")
        self.shapes: dict[str, tuple[int, ...]] = {}
        for register in program["registers"]:
            _require(isinstance(register, dict) and set(register) == {"name", "len", "shape"}, "invalid register fields")
            name, shape = register["name"], _shape(register["shape"], "register")
            _require(isinstance(name, str) and name and name not in self.shapes, "duplicate/empty register")
            _require(type(register["len"]) is int and register["len"] == math.prod(shape), "register shape/length mismatch")
            _require(all(axis <= 16384 for axis in shape), "register extent exceeds 16384")
            _require(program["version"] >= 3 or shape == (register["len"],), "multiaxis registers require IR version 3")
            self.shapes[name] = shape
        self.input = program["input"]
        _require(self.input in self.shapes and len(self.shapes[self.input]) == 1, "invalid input register")
        self.input_shape = self.shapes[self.input]
        all_specs = {spec["name"]: spec for spec in parameter_specs or program["parameters"]}
        self.specs: dict[str, dict] = {}
        for declared in program["parameters"]:
            fields = {"name", "shape", "data"}
            _require(isinstance(declared, dict) and fields <= declared.keys()
                     and declared.keys() <= fields | {"frozen", "probability", "learn"}, "invalid parameter fields")
            name = declared["name"]
            _require(isinstance(name, str) and name and name not in self.specs and name != self.input, "duplicate/invalid parameter")
            spec = deepcopy({key: value for key, value in all_specs[name].items() if key != "data"})
            spec["shape"] = _shape(spec["shape"], f"parameter {name}")
            _require(tuple(declared["shape"]) == spec["shape"], "parameter storage schema mismatch")
            _require(type(spec.get("frozen", False)) is bool and type(spec.get("probability", False)) is bool, "invalid parameter flags")
            _domain(spec.get("learn"))
            if spec.get("learn") is not None:
                _require(len(spec["shape"]) == 1 and not spec.get("frozen") and not spec.get("probability"), "learn parameter must be a trainable vector")
            self.specs[name] = spec
        self.labels: dict[str, int] = {}
        self.memories: dict[str, tuple[int, int]] = {}
        for pc, item in enumerate(self.instructions):
            op = item.get("op")
            _require(op in _FIELDS, f"unknown adaptive instruction {op!r}")
            required = set(_FIELDS[op].split()) | {"op"}
            actual = set(item)
            if op == "linear" and "bias" not in actual:
                actual.add("bias")
            _require(actual == required, f"invalid fields for {op}")
            _require(program["version"] >= 3 or op not in _GEOMETRY, f"{op} requires IR version 3")
            _require(program["version"] >= 2 or op != "validate_probabilities", "probability validation requires IR version 2")
            if op == "label":
                name = item["name"]
                _require(isinstance(name, str) and name and name not in self.labels, "duplicate/empty label")
                self.labels[name] = pc
            elif op == "memory_alloc":
                size = (_positive(item["slots"], "memory slots", 16384), _positive(item["width"], "memory width", 16384))
                previous = self.memories.setdefault(item["out"], size)
                _require(previous == size, "memory shape changes")
        self.output_shape: tuple[int, ...] | None = None
        self.costs: list[int] = []
        for item in self.instructions:
            self.costs.append(self._validate(item))
        _require(self.output_shape is not None, "program requires a return")
        self.output_type = program.get("output_type", "f32")
        _require(self.output_type in {"f32", "logits", "probabilities"}, "invalid output type")
        _require(program["version"] >= 2 or self.output_type == "f32", "categorical output requires IR version 2")
        parameter_elements = sum(math.prod(spec["shape"]) for spec in self.specs.values())
        constant_elements = sum(len(item["data"]) for item in self.instructions if item["op"] == "constant")
        register_elements = [math.prod(shape) for shape in self.shapes.values()]
        self.managed_bytes = 4 * (2 * parameter_elements + constant_elements + sum(register_elements) + max(register_elements))
        self.has_writes = any(item["op"] == "write" for item in self.instructions)
        loss = program.get("loss")
        if loss is not None:
            _require(isinstance(loss, dict) and loss.get("loss") is None and loss.get("parameters") == [],
                     "loss must be a parameter-free tensor program without a nested loss")
            loss_executor = ProgramExecutor(loss)
            _require(loss_executor.output_shape == (1,), "loss must return one scalar")

    def _register(self, name: str, *, output: bool = False) -> tuple[int, ...]:
        _require(name in self.shapes and name not in self.memories, f"unknown tensor register {name!r}")
        _require(not output or name != self.input, "input register is read-only")
        return self.shapes[name]

    @staticmethod
    def _dimensions(values: Any, rank: int, minimum: int = 1) -> tuple[int, ...]:
        _require(isinstance(values, list) and len(values) == rank
                 and all(type(v) is int and minimum <= v <= 16384 for v in values), "invalid spatial options")
        return tuple(values)

    def _validate(self, item: dict) -> int:
        op = item["op"]
        shape = None
        if "out" in item and op != "memory_alloc":
            shape = self._register(item["out"], output=True)
        reg = lambda key: self._register(item[key])
        if op == "memory_alloc":
            _require(item["out"] != self.input and self.shapes.get(item["out"]) == (1,), "invalid memory handle register")
            return 1 + math.prod(self.memories[item["out"]])
        if op in {"memory_free", "memory_read", "memory_write"}:
            _require(item["memory"] in self.memories, "unknown memory handle")
            slots, width = self.memories[item["memory"]]
            if op != "memory_free":
                _require(reg("address") == (slots,), "memory address shape mismatch")
                _require((shape if op == "memory_read" else reg("value")) == (width,), "memory value shape mismatch")
            return 1 if op == "memory_free" else 1 + slots * width
        if op in {"label", "jump", "branch", "inference_skip"}:
            targets = {"jump": ["target"], "branch": ["positive", "otherwise"], "inference_skip": ["target"]}.get(op, [])
            _require(all(item[key] in self.labels for key in targets), "unknown instruction label")
            if op == "branch":
                _require(reg("condition") == (1,), "branch condition must be a scalar vector")
            if op == "inference_skip":
                reg("probability")
                _require(type(item["when_one"]) is bool, "invalid skip endpoint")
            return 1
        if op == "return":
            shape = reg("value")
            _require(len(shape) == 1, "model return requires a vector")
            _require(self.output_shape is None or self.output_shape == shape, "all returns must have equal shape")
            self.output_shape = shape
        elif op == "resource_state":
            _require(shape == (3,), "resource_state requires f32[3]")
        elif op == "constant":
            data = item["data"]
            _require(isinstance(data, list) and len(data) == math.prod(shape), "constant shape mismatch")
            for number in data:
                _require(abs(_finite(number, "constant")) <= torch.finfo(torch.float32).max, "constant exceeds f32 range")
        elif op in {"read", "write"}:
            spec = self.specs[item["parameter"]]
            tensor_shape = shape if op == "read" else reg("value")
            expected_shape = spec["shape"] if op == "write" and self.transactional else (math.prod(spec["shape"]),)
            _require(tensor_shape == expected_shape,
                     "typed parameter write must match storage shape" if op == "write" and self.transactional
                     else "parameter read/write requires its flattened vector")
            if op == "write":
                _require(not spec.get("frozen") and not spec.get("probability") and (self.transactional or spec.get("learn") is None),
                         "cannot write frozen/prob/learn storage")
        elif op == "linear":
            input_shape = reg("input")
            weight = self.specs[item["weight"]]
            _require(len(shape) == len(input_shape) == 1 and weight["shape"] == (shape[0], input_shape[0]), "linear shape mismatch")
            _require(weight.get("learn") is None and not weight.get("probability"), "linear cannot consume raw learn/prob storage")
            if item.get("bias") is not None:
                bias = self.specs[item["bias"]]
                _require(bias["shape"] == shape and bias.get("learn") is None and not bias.get("probability"), "linear bias mismatch")
        elif op in _BINARY:
            _require(shape == reg("left") == reg("right"), f"{op} shape mismatch")
        elif op in _UNARY or op in {"softmax", "log_softmax", "validate_probabilities", "clip", "scale", "scale_by", "softmax_axis", "layer_norm"}:
            previous = reg("input")
            _require(shape == previous, f"{op} shape mismatch")
            if op in {"softmax", "log_softmax", "validate_probabilities"}:
                _require(len(shape) == 1, f"{op} requires a vector")
            if op == "clip":
                _require(_finite(item["low"], "clip low") <= _finite(item["high"], "clip high"), "invalid clip bounds")
            if op == "scale":
                _finite(item["factor"], "scale factor")
            if op == "scale_by":
                _require(reg("scalar") == (1,), "scale_by scalar must have shape [1]")
            if op in {"softmax_axis", "layer_norm"}:
                axis = item["axis"]
                _require(type(axis) is int and 0 <= axis < len(shape), "invalid tensor axis")
                if op == "layer_norm":
                    _require(reg("gamma") == reg("beta") == (shape[axis],), "layer_norm affine shape mismatch")
                    _require(_finite(item["epsilon"], "layer_norm epsilon") > 0, "layer_norm epsilon must be positive")
                    return 1 + 3 * math.prod(shape)
        elif op in {"mean", "sum"}:
            reg("input")
            _require(shape == (1,), "reduction output must be a scalar vector")
        elif op in {"gate", "route"}:
            _require(shape == reg("input" if op == "gate" else "yes"), "gate/route result shape mismatch")
            if op == "route":
                _require(shape == reg("no"), "route candidate shape mismatch")
            _require(reg("probability") in {(1,), shape}, "gate/route probability shape mismatch")
        elif op == "slice":
            previous, start = reg("input"), item["start"]
            _require(len(shape) == len(previous) == 1 and type(start) is int and start >= 0 and start + shape[0] <= previous[0], "slice outside vector")
        elif op in {"reshape", "transpose", "permute"}:
            previous = reg("input")
            if op == "reshape":
                _require(math.prod(shape) == math.prod(previous), "reshape changes element count")
            else:
                axes = [1, 0] if op == "transpose" else item["axes"]
                _require(isinstance(axes, list) and all(type(axis) is int for axis in axes)
                         and sorted(axes) == list(range(len(previous))), "invalid tensor permutation")
                _require(shape == tuple(previous[axis] for axis in axes), "permutation shape mismatch")
            return 1 + math.prod(shape)
        elif op in {"matmul", "batch_matmul"}:
            left, right = reg("left"), reg("right")
            rank = 3 if op == "batch_matmul" else 2
            _require(len(left) == len(right) == rank and left[:-2] == right[:-2] and left[-1] == right[-2], "matmul axes mismatch")
            _require(shape == (*left[:-2], left[-2], right[-1]), "matmul result shape mismatch")
            return 1 + math.prod(left) * right[-1]
        elif op == "conv":
            previous, kernel, bias = reg("input"), reg("kernel"), reg("bias")
            rank, groups = len(previous) - 1, item["groups"]
            _require(rank in (1, 2, 3) and len(kernel) == rank + 2, "conv requires 1..3 spatial axes")
            groups = _positive(groups, "groups", 16384)
            _require(previous[0] % groups == kernel[0] % groups == 0 and kernel[1] == previous[0] // groups and bias == (kernel[0],), "conv channel/group mismatch")
            stride, padding, dilation = (self._dimensions(item[key], rank, minimum) for key, minimum in (("stride", 1), ("padding", 0), ("dilation", 1)))
            expected = (kernel[0], *(1 + (size + 2 * pad - dil * (extent - 1) - 1) // step
                        for size, extent, pad, dil, step in zip(previous[1:], kernel[2:], padding, dilation, stride)))
            _require(shape == expected, "conv result shape mismatch")
            return 1 + math.prod(shape) * math.prod(kernel[1:])
        elif op in {"max_pool2d", "avg_pool2d"}:
            previous = reg("input")
            _require(len(previous) == 3, "pool2d requires [C,H,W]")
            kernel, stride = self._dimensions(item["kernel"], 2), self._dimensions(item["stride"], 2)
            expected = (previous[0], *(1 + (size - extent) // step for size, extent, step in zip(previous[1:], kernel, stride)))
            _require(shape == expected, "pool2d result shape mismatch")
            return 1 + math.prod(shape) * math.prod(kernel)
        return 1

    def run(self, input: torch.Tensor, parameters: Mapping[str, torch.Tensor], *, differentiable: bool,
            max_work: int = DEFAULT_MAX_WORK, memory_bytes: int = DEFAULT_MEMORY_BYTES,
            diagnostic_anchor: torch.Tensor | None = None) -> ExecutionResult:
        _require(type(max_work) is int and 0 < max_work < 2**64, "max_work must be a positive u64")
        _require(type(memory_bytes) is int and memory_bytes >= self.managed_bytes,
                 f"adaptive managed memory budget exceeded: need {self.managed_bytes}, have {memory_bytes}")
        _require(tuple(input.shape) == self.input_shape, "invalid adaptive input shape")
        _finite_tensor(input, "adaptive input")
        if differentiable and diagnostic_anchor is None:
            diagnostic_anchor = input.new_zeros((), requires_grad=True)
        context = nullcontext() if differentiable else torch.no_grad()
        with context:
            values = {self.input: ops.check_finite(input.clone(), diagnostic_anchor)}
            candidates = {}
            for name, spec in self.specs.items():
                value = parameters[name]
                validate_parameter(value, spec)
                _require(value.device == input.device and value.dtype == input.dtype, f"parameter {name} and input device/dtype differ")
                candidates[name] = ops.check_finite(value.clone(), diagnostic_anchor)
            memories: dict[str, torch.Tensor] = {}
            written: set[str] = set()
            work = memory_used = peak = pc = 0
            while True:
                _require(0 <= pc < len(self.instructions), "adaptive program fell through without return")
                item = self.instructions[pc]
                op, cost = item["op"], self.costs[pc]
                _require(cost <= max_work - work, "adaptive instruction budget exceeded")
                work += cost
                get = lambda key: values[item[key]]
                out = None
                try:
                    if op == "label":
                        pass
                    elif op == "jump":
                        pc = self.labels[item["target"]]
                        continue
                    elif op == "branch":
                        branch = "positive" if get("condition").item() > 0 else "otherwise"
                        pc = self.labels[item[branch]]
                        continue
                    elif op == "inference_skip":
                        probability = get("probability")
                        _require(_all((probability >= 0) & (probability <= 1)), "gate/route probability must be in [0,1]")
                        if not differentiable and _all(probability == int(item["when_one"])):
                            pc = self.labels[item["target"]]
                            continue
                    elif op == "return":
                        output = get("value")
                        if self.output_type == "probabilities":
                            _probabilities(output)
                        return ExecutionResult(output, candidates, written, work, peak)
                    elif op == "constant":
                        out = input.new_tensor(item["data"]).reshape(self.shapes[item["out"]])
                    elif op == "read":
                        name = item["parameter"]
                        out = read_parameter(candidates[name], self.specs[name], differentiable,
                                             validate=not self.transactional).reshape(self.shapes[item["out"]])
                    elif op == "write":
                        name = item["parameter"]
                        candidate = get("value").reshape(self.specs[name]["shape"]).clone()
                        if self.transactional:
                            _finite_tensor(candidate, f"parameter {name}")
                        else:
                            validate_parameter(candidate, self.specs[name])
                        candidates[name] = candidate
                        written.add(name)
                    elif op == "resource_state":
                        out = input.new_tensor([max_work - work, memory_bytes - self.managed_bytes - memory_used, memory_used])
                    elif op == "memory_alloc":
                        name = item["out"]
                        _require(name not in memories, "memory already allocated; free it before reallocating")
                        size = self.memories[name]
                        needed = 4 * math.prod(size)
                        _require(memory_used + needed <= memory_bytes - self.managed_bytes, "adaptive workspace memory budget exceeded")
                        memories[name] = ops.check_finite(input.new_zeros(size), diagnostic_anchor)
                        memory_used += needed
                        peak = max(peak, memory_used)
                    elif op in {"memory_free", "memory_read", "memory_write"}:
                        name = item["memory"]
                        _require(name in memories, "memory is not allocated or has been freed")
                        memory = memories[name]
                        if op == "memory_free":
                            memory_used -= 4 * memory.numel()
                            del memories[name]
                        else:
                            address = get("address")
                            _require(_all(torch.isfinite(address) & (address >= 0) & (address <= 1)), "memory address must have finite [0,1] weights")
                            if op == "memory_read":
                                out = ops.matmul(address, memory)
                            else:
                                candidate = ops.memory_write(memory, address, get("value"))
                                _finite_tensor(candidate, "memory write")
                                memories[name] = ops.check_finite(candidate, diagnostic_anchor)
                    elif op == "linear":
                        bias = candidates[item["bias"]] if item.get("bias") is not None else None
                        out = ops.linear(get("input"), candidates[item["weight"]], bias)
                    elif op in _UNARY:
                        out = _UNARY[op](get("input"))
                    elif op in _BINARY:
                        _require(op != "ge" or not differentiable,
                                 "ge does not support automatic differentiation; use non-AD evaluation")
                        out = _BINARY[op](get("left"), get("right"))
                    elif op in {"gate", "route"}:
                        probability = get("probability")
                        _require(_all((probability >= 0) & (probability <= 1)), "gate/route probability must be in [0,1]")
                        if not differentiable and _all(probability == 0):
                            out = get("no") if op == "route" else input.new_zeros(self.shapes[item["out"]])
                        elif not differentiable and _all(probability == 1):
                            out = get("yes" if op == "route" else "input")
                        else:
                            yes = get("yes" if op == "route" else "input")
                            no = get("no") if op == "route" else None
                            out = ops.blend(probability, yes, no)
                    elif op == "reshape":
                        out = get("input").reshape(self.shapes[item["out"]])
                    elif op == "transpose":
                        out = get("input").transpose(0, 1)
                    elif op == "permute":
                        out = get("input").permute(item["axes"])
                    elif op in {"matmul", "batch_matmul"}:
                        out = ops.matmul(get("left"), get("right"))
                    elif op in {"softmax", "softmax_axis", "log_softmax"}:
                        function = ops.log_softmax if op == "log_softmax" else ops.softmax
                        out = function(get("input"), item["axis"], True) if op == "softmax_axis" else function(get("input"), -1)
                    elif op == "layer_norm":
                        axis = item["axis"]
                        value = get("input")
                        affine = tuple(value.shape[index] if index == axis else 1 for index in range(value.ndim))
                        out = ops.layer_norm(value, get("gamma").reshape(affine), get("beta").reshape(affine), axis, item["epsilon"])
                    elif op == "scale":
                        out = get("input") * item["factor"]
                    elif op == "scale_by":
                        out = ops.scale_by(get("input"), get("scalar"), len(self.shapes[item["out"]]))
                    elif op == "clip":
                        out = ops.clip(get("input"), item["low"], item["high"])
                    elif op in {"sum", "mean"}:
                        out = (ops.sum(get("input")) if op == "sum" else ops.mean(get("input"))).reshape(1)
                    elif op == "slice":
                        start = item["start"]
                        out = get("input")[start:start + self.shapes[item["out"]][0]]
                    elif op == "validate_probabilities":
                        out = get("input")
                        _probabilities(out)
                    elif op == "conv":
                        out = ops.conv(get("input"), get("kernel"), get("bias"),
                                tuple(item["stride"]), tuple(item["padding"]), tuple(item["dilation"]), item["groups"])
                    elif op in {"max_pool2d", "avg_pool2d"}:
                        function = F.max_pool2d if op == "max_pool2d" else ops.avg_pool2d
                        out = function(get("input"), tuple(item["kernel"]), tuple(item["stride"]))
                    else:
                        raise ValueError(f"unknown adaptive instruction {op!r}")
                    if out is not None:
                        _require(tuple(out.shape) == self.shapes[item["out"]], f"{op} produced an unexpected shape")
                        _finite_tensor(out, "activation")
                        values[item["out"]] = ops.check_finite(out, diagnostic_anchor)
                except (KeyError, RuntimeError, ValueError) as error:
                    raise RuntimeError(f"instruction {pc} ({op}): {error}") from error
                pc += 1


def _limits(model: nn.Module, max_work: int | None, memory_bytes: int | None) -> tuple[int, int]:
    return (model.max_work if max_work is None else max_work,
            model.memory_bytes if memory_bytes is None else memory_bytes)


def model_loss(model: nn.Module, prediction: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean",
               max_work: int | None = None, memory_bytes: int | None = None) -> torch.Tensor:
    executor = model._loss_executor
    if executor is None:
        raise ValueError("this model has no WGT fn loss; use a PyTorch loss function")
    interface = model.definition["interface"]
    output = interface["output"]
    target_width = executor.input_shape[0] - output
    ops.check_input(prediction, (output,))
    ops.check_input(target, (target_width,))
    _require(prediction.shape[:-1] == target.shape[:-1], "prediction and target batch shapes differ")
    _require(prediction.device == target.device and prediction.dtype == target.dtype, "prediction and target device/dtype differ")
    _require(reduction in {"none", "mean", "sum"}, "loss reduction must be none, mean or sum")
    maximum, memory = _limits(model, max_work, memory_bytes)
    rows = torch.cat((prediction, target), dim=-1).reshape(-1, executor.input_shape[0])
    _require(rows.shape[0] > 0, "loss requires a nonempty batch")
    results, used = [], 0
    for row in rows:
        result = executor.run(row, {}, differentiable=torch.is_grad_enabled(), max_work=maximum - used,
                              memory_bytes=memory, diagnostic_anchor=model._wgt_gradient_anchor)
        results.append(result.output)
        used += result.work
    losses = torch.stack(results).reshape(prediction.shape[:-1])
    return losses if reduction == "none" else losses.mean() if reduction == "mean" else losses.sum()


class ProgramModule(nn.Module):
    """A native forward program with standard registered PyTorch parameters."""

    def __init__(self, definition: dict, *, max_work: int = DEFAULT_MAX_WORK, memory_bytes: int = DEFAULT_MEMORY_BYTES):
        super().__init__()
        self.register_buffer("_wgt_gradient_anchor", torch.zeros((), requires_grad=True), persistent=False)
        self.definition = deepcopy(definition)
        self.input_shape = (definition["interface"]["input"],)
        self.output_shape = (definition["interface"]["output"],)
        self.max_work, self.memory_bytes = max_work, memory_bytes
        self.parameter_names: dict[str, str] = {}
        self.learn_domains: dict[str, dict] = {}
        definition = self.definition
        self._specs = {spec["name"]: spec for spec in definition["program"]["parameters"]}
        self.parameter_names = register_storage(self, list(self._specs.values()), definition.get("modules"))
        for name, spec in self._specs.items():
            validate_parameter(self.get_parameter(self.parameter_names[name]), spec)
            if spec.get("learn") is not None:
                self.learn_domains[name] = spec["learn"]
        self._executor = ProgramExecutor(definition["program"])
        _require(self._executor.input_shape == self.input_shape and self._executor.output_shape == self.output_shape, "forward interface mismatch")
        loss = definition["program"].get("loss")
        self._loss_executor = ProgramExecutor(loss) if loss is not None else None
        self.last_execution: dict[str, int] | None = None
        self.revision = 0
        self._execution_lock = RLock()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_execution_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._execution_lock = RLock()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        ops.reset_gradient_anchor(self)
        return self

    def _parameters_by_name(self) -> dict[str, torch.Tensor]:
        return {name: self.get_parameter(attribute) for name, attribute in self.parameter_names.items()}

    def _execute(self, value: torch.Tensor, *, commit: bool, differentiable: bool,
                 max_work: int | None, memory_bytes: int | None) -> torch.Tensor:
        with self._execution_lock:
            return self._execute_locked(value, commit=commit, differentiable=differentiable,
                                        max_work=max_work, memory_bytes=memory_bytes)

    def _execute_locked(self, value: torch.Tensor, *, commit: bool, differentiable: bool,
                        max_work: int | None, memory_bytes: int | None) -> torch.Tensor:
        value = ops.check_input(value, self.input_shape, self._wgt_gradient_anchor)
        maximum, memory = _limits(self, max_work, memory_bytes)
        rows = value.reshape(-1, self.input_shape[0])
        _require(rows.shape[0] > 0, "program execution requires a nonempty batch")
        snapshot = self._parameters_by_name()
        staged = snapshot
        outputs, written, work, peak = [], set(), 0, 0
        for row in rows:
            result = self._executor.run(row, staged if commit else snapshot,
                                       differentiable=differentiable, max_work=maximum - work, memory_bytes=memory,
                                       diagnostic_anchor=self._wgt_gradient_anchor)
            outputs.append(result.output)
            work += result.work
            peak = max(peak, result.peak_memory_bytes)
            written.update(result.writes)
            if commit:
                staged = result.parameters
        if commit and written:
            with torch.no_grad():
                for name in written:
                    snapshot[name].copy_(staged[name])
            self.revision += 1
        self.last_execution = {"work": work, "peak_memory_bytes": peak, "memory_bytes": 0}
        return torch.stack(outputs).reshape(*value.shape[:-1], self.output_shape[0])

    def forward(self, value: torch.Tensor, *, max_work: int | None = None, memory_bytes: int | None = None) -> torch.Tensor:
        """Read-only snapshot forward; grad mode controls native lazy evaluation."""
        return self._execute(value, commit=False, differentiable=torch.is_grad_enabled(), max_work=max_work, memory_bytes=memory_bytes)

    @torch.no_grad()
    def run(self, value: torch.Tensor, *, max_work: int | None = None, memory_bytes: int | None = None) -> torch.Tensor:
        """Commit successful forward writes; a failed call leaves storage intact."""
        return self._execute(value, commit=True, differentiable=False, max_work=max_work, memory_bytes=memory_bytes)

    @torch.no_grad()
    def evaluate(self, value: torch.Tensor, *, max_work: int | None = None, memory_bytes: int | None = None) -> torch.Tensor:
        return self._execute(value, commit=False, differentiable=False, max_work=max_work, memory_bytes=memory_bytes)

    def loss(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return model_loss(self, prediction, target, **kwargs)

    @torch.no_grad()
    def project_parameters_(self) -> ProgramModule:
        parameters = self._parameters_by_name()
        projected = {name: project_value(parameters[name], domain) for name, domain in self.learn_domains.items()}
        for name, value in projected.items():
            parameters[name].copy_(value)
        return self

    def bind_optimizer(self, optimizer: torch.optim.Optimizer):
        from .training import bind_optimizer
        return bind_optimizer(self, optimizer)
