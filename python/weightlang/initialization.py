"""Execute compiler-produced initialization recipes, never WGT source."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import torch


def parameter_specs(definition: dict) -> list[dict]:
    """Return logical parameter schemas without constructing a model or values."""
    if not isinstance(definition, dict) or definition.get("format") != "weightlang.torch.v1":
        raise ValueError("expected a weightlang.torch.v1 compiler export")
    kind = definition.get("kind")
    if kind == "forward":
        specs = definition["program"]["parameters"]
    elif kind == "operations":
        specs = definition["operations"]["parameters"]
    elif kind == "sequential":
        specs = definition.get("graph_parameters", definition["graph"].get("parameters") or definition.get("parameters", []))
    else:
        raise ValueError(f"unsupported model plan {kind!r}")
    from .runtime import _MAX_ELEMENTS, _shape
    result, names, count = [], set(), 0
    for spec in specs:
        name = spec.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("duplicate or invalid logical parameter name")
        names.add(name)
        count += math.prod(_shape(spec["shape"], f"parameter {name}"))
        if count > _MAX_ELEMENTS:
            raise ValueError("model parameter element limit exceeded")
        if spec.get("dtype", "F32") not in ("F32", "f32"):
            raise ValueError(f"unsupported parameter dtype: {name}")
        result.append(deepcopy(spec))
    return result


def checked_binding(value: torch.Tensor, spec: dict, *, f32: bool = False) -> torch.Tensor:
    from .program import validate_parameter
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError(f"parameter {spec['name']} requires a dense torch.Tensor")
    if f32 and value.dtype != torch.float32:
        raise ValueError(f"parameter {spec['name']} requires F32; convert dtype explicitly")
    validate_parameter(value, spec)
    return value.detach().to(device="cpu").contiguous().clone()


def checked_bindings(definition: dict, values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    specs = parameter_specs(definition)
    expected = {spec["name"] for spec in specs}
    if set(values) != expected:
        raise ValueError(f"weight bindings must exactly cover parameters; missing={sorted(expected - set(values))}, unexpected={sorted(set(values) - expected)}")
    return {spec["name"]: checked_binding(values[spec["name"]], spec) for spec in specs}


def recipe_shape(recipe: dict, *, depth: int = 0, remaining: list[int] | None = None) -> tuple[int, ...]:
    """Check the complete recipe before allocating generated parameter values."""
    from .runtime import _finite, _shape
    remaining = [1_000_000] if remaining is None else remaining
    if not isinstance(recipe, dict) or depth > 64 or remaining[0] <= 0:
        raise ValueError("invalid or excessively complex initializer recipe")
    remaining[0] -= 1
    fields = {"unbound": set(), "scalar": {"value"}, "literal": {"shape", "values"},
              "full": {"shape", "value"}, "eye": {"shape"}, "uniform": {"shape", "seed", "low", "high"},
              "unary": {"function", "input"}, "binary": {"operator", "left", "right"},
              "reshape": {"shape", "input"}, "transpose": {"input"}, "concat": {"left", "right"},
              "slice": {"input", "start", "end"}}
    op = recipe.get("op")
    if op not in fields or set(recipe) != fields[op] | {"op"}:
        raise ValueError("invalid initializer recipe fields")
    child = lambda value: recipe_shape(value, depth=depth + 1, remaining=remaining)
    declared = lambda: () if recipe["shape"] == [] else _shape(recipe["shape"], "initializer")
    if op == "unbound":
        raise ValueError("unbound parameter requires explicit weight bindings")
    if op == "scalar":
        _finite(recipe["value"], "initializer scalar")
        shape = ()
    elif op in {"literal", "full", "eye", "uniform"}:
        shape = declared()
        if op == "literal":
            values = recipe["values"]
            if not isinstance(values, list) or len(values) != math.prod(shape):
                raise ValueError("initializer literal shape mismatch")
            remaining[0] -= len(values)
            if remaining[0] < 0:
                raise ValueError("initializer literal exceeds expression budget")
            if any(abs(_finite(value, "initializer literal")) > torch.finfo(torch.float32).max for value in values):
                raise ValueError("initializer literal exceeds F32 range")
        elif op == "full":
            if abs(_finite(recipe["value"], "initializer full")) > torch.finfo(torch.float32).max:
                raise ValueError("initializer full exceeds F32 range")
        elif op == "eye" and len(shape) != 2:
            raise ValueError("initializer eye requires rank two")
        elif op == "uniform":
            low, high, seed = _finite(recipe["low"], "uniform low"), _finite(recipe["high"], "uniform high"), recipe["seed"]
            if (not low < high or max(abs(low), abs(high)) > torch.finfo(torch.float32).max
                    or type(seed) is not int or not 0 <= seed < 2**64):
                raise ValueError("uniform requires bounded F32 endpoints and an explicit u64 seed")
    elif op == "unary":
        if recipe["function"] not in {"neg", "sqrt", "exp", "log", "tanh", "sigmoid", "relu", "square"}:
            raise ValueError("unknown initializer function")
        shape = child(recipe["input"])
    elif op == "binary":
        left, right, operator = child(recipe["left"]), child(recipe["right"]), recipe["operator"]
        if operator == "@":
            if len(left) != 2 or len(right) != 2 or left[1] != right[0]:
                raise ValueError("initializer matmul shape mismatch")
            shape = (left[0], right[1])
        else:
            if operator not in {"+", "-", "*", "/"} or (left and right and left != right):
                raise ValueError("initializer arithmetic shape or operator mismatch")
            shape = left or right
    elif op == "reshape":
        shape = declared()
        if math.prod(shape) != math.prod(child(recipe["input"])):
            raise ValueError("initializer reshape element mismatch")
    elif op == "transpose":
        shape = child(recipe["input"])
        if len(shape) != 2:
            raise ValueError("initializer transpose requires rank two")
        shape = shape[::-1]
    elif op == "concat":
        left, right = child(recipe["left"]), child(recipe["right"])
        if len(left) != 1 or len(right) != 1:
            raise ValueError("initializer concat requires vectors")
        shape = (left[0] + right[0],)
    else:
        shape, start, end = child(recipe["input"]), recipe["start"], recipe["end"]
        if len(shape) != 1 or type(start) is not int or type(end) is not int or not 0 <= start < end <= shape[0]:
            raise ValueError("invalid initializer slice")
        shape = (end - start,)
    if math.prod(shape) > 16_777_216 or any(axis > 16384 for axis in shape):
        raise ValueError("initializer shape exceeds bounds")
    return shape


def recipe_value(recipe: dict, *, depth: int = 0) -> torch.Tensor:
    """Materialize the shared initializer IR with explicit, private random streams."""
    from .runtime import _finite, _initialization_random, _shape, _tensor
    if not isinstance(recipe, dict) or depth > 64:
        raise ValueError("invalid or excessively nested initializer recipe")
    op = recipe.get("op")
    child = lambda value: recipe_value(value, depth=depth + 1)
    declared = lambda: () if recipe["shape"] == [] else _shape(recipe["shape"], "initializer")
    if op == "unbound":
        raise ValueError("unbound parameter requires explicit weight bindings")
    if op == "scalar":
        result = torch.tensor(_finite(recipe["value"], "initializer scalar"), dtype=torch.float32)
    elif op == "literal":
        result = _tensor(recipe["values"], declared(), "initializer")
    elif op == "full":
        result = torch.full(declared(), _finite(recipe["value"], "initializer full"), dtype=torch.float32)
    elif op == "eye":
        shape = _shape(recipe["shape"], "initializer")
        if len(shape) != 2:
            raise ValueError("eye initializer requires rank two")
        result = torch.eye(*shape, dtype=torch.float32)
    elif op == "uniform":
        shape, seed = declared(), recipe["seed"]
        low, high = _finite(recipe["low"], "uniform low"), _finite(recipe["high"], "uniform high")
        if type(seed) is not int or not 0 <= seed < 2**64 or not low < high:
            raise ValueError("uniform requires an explicit u64 seed and low < high")
        generator = _initialization_random(seed)
        result = torch.tensor([low + (high - low) * next(generator) for _ in range(math.prod(shape))], dtype=torch.float32).reshape(shape)
    elif op == "unary":
        value, function = child(recipe["input"]), recipe["function"]
        functions = {"neg": torch.neg, "sqrt": torch.sqrt, "exp": torch.exp, "log": torch.log,
                     "tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": torch.relu, "square": torch.square}
        if function not in functions:
            raise ValueError(f"unknown initializer function {function!r}")
        if (function == "sqrt" and bool((value < 0).any())) or (function == "log" and bool((value <= 0).any())):
            raise ValueError(f"invalid initializer {function} domain")
        result = functions[function](value.double()).float()
    elif op == "binary":
        left, right, operator = child(recipe["left"]), child(recipe["right"]), recipe["operator"]
        if operator == "@":
            if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
                raise ValueError("initializer matmul shape mismatch")
            result = (left.double() @ right.double()).float()
        else:
            if left.shape != right.shape and left.ndim and right.ndim:
                raise ValueError("initializer arithmetic requires equal shapes or an explicit scalar")
            functions = {"+": torch.add, "-": torch.sub, "*": torch.mul, "/": torch.div}
            if operator not in functions or (operator == "/" and bool((right == 0).any())):
                raise ValueError("invalid initializer arithmetic")
            result = functions[operator](left.double(), right.double()).float()
    elif op == "reshape":
        result = child(recipe["input"]).reshape(declared())
    elif op == "transpose":
        value = child(recipe["input"])
        if value.ndim != 2:
            raise ValueError("initializer transpose requires rank two")
        result = value.T.contiguous()
    elif op == "concat":
        left, right = child(recipe["left"]), child(recipe["right"])
        if left.ndim != 1 or right.ndim != 1:
            raise ValueError("initializer concat requires vectors")
        result = torch.cat((left, right))
    elif op == "slice":
        value, start, end = child(recipe["input"]), recipe["start"], recipe["end"]
        if value.ndim != 1 or type(start) is not int or type(end) is not int or not 0 <= start < end <= value.numel():
            raise ValueError("invalid initializer slice")
        result = value[start:end].clone()
    else:
        raise ValueError(f"unknown initializer recipe {op!r}")
    if result.numel() > 16_777_216 or not bool(torch.isfinite(result).all()):
        raise ValueError("initializer exceeds element budget or produces nonfinite values")
    return result


def materialize(spec: dict, bindings: Mapping[str, torch.Tensor] | None = None, *, schema_only: bool = False) -> torch.Tensor:
    """Bindings take precedence; a restored parameter never runs its initializer."""
    if schema_only:
        from .runtime import _shape
        return torch.empty(_shape(spec["shape"], "parameter"), dtype=torch.float32, device="meta")
    if bindings is not None:
        return checked_binding(bindings[spec["name"]], spec)
    if spec.get("initializer") is not None:
        if spec.get("data"):
            raise ValueError("initializer cannot supply both a recipe and materialized data")
        if recipe_shape(spec["initializer"]) != tuple(spec["shape"]):
            raise ValueError(f"parameter {spec['name']} initializer shape mismatch")
        value = recipe_value(spec["initializer"])
    else:
        from .runtime import _shape, _tensor
        value = _tensor(spec.get("data", []), _shape(spec["shape"], "parameter"), spec["name"])
    return checked_binding(value, spec)
