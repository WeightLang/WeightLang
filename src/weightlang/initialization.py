"""Explicit construction-time weight generation, separate from forward/training."""
from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import math
from pathlib import Path

import torch

from .syntax import WgtError

HEADER = "wgt init 1"
METHODS = {
    "zeros": {}, "ones": {}, "constant": {"value": 0.0},
    "eye": {"gain": 1.0}, "dct": {"gain": 1.0}, "orthogonal": {"gain": 1.0},
    "xavier_uniform": {"gain": 1.0}, "xavier_normal": {"gain": 1.0},
    "normal": {"mean": 0.0, "std": 1.0}, "uniform": {"low": 0.0, "high": 1.0},
}


def read_rules(filename, source_path=None):
    path = (Path(source_path).parent if source_path else Path.cwd()) / filename
    try:
        raw = path.read_bytes()
        from .init_algorithm import parse_source
        declarations, library = parse_source(raw.decode(), str(path))
        rules = []
        for line, words, function in declarations:
            if function is not None:
                rules.append((words[1], "algorithm", {"function": function, "library": library}))
                continue
            if len(words) < 3 or words[0] != "tensor" or words[2] not in METHODS:
                raise WgtError('expected tensor "PATTERN" METHOD [key=value ...]', line, str(path))
            _, pattern, method, *options = words
            values = dict(METHODS[method])
            seen = set()
            for option in options:
                key, sep, value = option.partition("=")
                if not sep or key not in values or key in seen:
                    raise WgtError(f"invalid or duplicate {method} option: {option}", line, str(path))
                values[key] = float(value)
                seen.add(key)
            if not pattern or not all(math.isfinite(v) for v in values.values()):
                raise WgtError("require non-empty pattern and finite options", line, str(path))
            if method == "normal" and values["std"] < 0 or method == "uniform" and values["low"] > values["high"]:
                raise WgtError("require std >= 0 and low <= high", line, str(path))
            rules.append((pattern, method, values))
        if not rules:
            raise ValueError("initialization file needs tensor rules")
        return rules, {"source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    except (OSError, UnicodeError, ValueError) as exc:
        raise WgtError(f"cannot read initialization rules: {exc}", path=str(path)) from exc


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    # Equal names identify parameters that must receive exactly equal values.
    alias_of: str


@dataclass(frozen=True)
class InitializationContext:
    tensors: tuple[TensorSpec, ...]
    seed: int


def materialize(module):
    """Allocate without random defaults, restoring aliases after meta conversion."""
    canonical, aliases = {}, {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases[name] = canonical.setdefault(id(parameter), name)
    module.to_empty(device="cpu")
    for name, original in aliases.items():
        if name != original:
            parent, _, leaf = name.rpartition(".")
            module.get_submodule(parent).register_parameter(leaf, module.get_parameter(original))
    return module


def initialize(program, module):
    expected = module.state_dict()
    aliases, canonical = {}, {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases[name] = canonical.setdefault(id(parameter), name)
    specs = tuple(TensorSpec(name, tuple(value.shape), value.dtype, aliases.get(name, name))
                  for name, value in expected.items())
    try:
        if "python" in program.initializer:
            from .plugins import load_callback
            callback, record = load_callback(program.initializer["python"], None,
                                             "weightlang.initializers", program.path, "initialize")
            with torch.no_grad():
                result = callback(InitializationContext(specs, program.seed))
            method = "python"
        else:
            rules, record = read_rules(program.initializer["file"], program.path)
            result, selected = {}, {}
            for pattern, algorithm, options in rules:
                matches = [name for name in expected if fnmatchcase(name, pattern)]
                if not matches:
                    raise ValueError(f"initialization pattern matches no tensors: {pattern}")
                selected.update({name: (algorithm, options) for name in matches})
            for spec in specs:
                if spec.name not in selected:
                    raise ValueError(f"no initialization rule for {spec.name}")
                algorithm, options = selected[spec.name]
                if spec.alias_of != spec.name:
                    if selected.get(spec.alias_of) != selected[spec.name]:
                        raise ValueError(f"conflicting tied initialization rules: {spec.alias_of}, {spec.name}")
                    result[spec.name] = result[spec.alias_of]
                    continue
                if algorithm == "algorithm":
                    result[spec.name] = options["library"].generate(options["function"], spec, program.seed)
                    continue
                value = torch.empty(spec.shape, dtype=spec.dtype, device="cpu")
                if algorithm == "dct":
                    if value.ndim != 2:
                        raise ValueError(f"dct requires a matrix: {spec.name}")
                    rows, cols = value.shape
                    size, count = max(rows, cols), min(rows, cols)
                    frequencies = torch.arange(count, dtype=torch.float64)[:, None]
                    positions = torch.arange(size, dtype=torch.float64)[None, :] + 0.5
                    basis = torch.cos(math.pi * frequencies * positions / size) * math.sqrt(2 / size)
                    basis[0].mul_(1 / math.sqrt(2))
                    value.copy_(basis if rows <= cols else basis.T)
                    value.mul_(options["gain"])
                elif algorithm == "eye":
                    torch.nn.init.eye_(value)
                    value.mul_(options["gain"])
                elif algorithm == "uniform":
                    torch.nn.init.uniform_(value, a=options["low"], b=options["high"])
                elif algorithm == "constant":
                    torch.nn.init.constant_(value, options["value"])
                else:
                    getattr(torch.nn.init, algorithm + "_")(value, **options)
                result[spec.name] = value
            method = "wgt-algorithm" if any(method == "algorithm" for method, _ in selected.values()) else "wgt-rules"
        if not isinstance(result, dict) or set(result) != set(expected):
            raise ValueError("initializer must return exactly all state tensor names")
        for spec in specs:
            value = result[spec.name]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.device.type != "cpu" or tuple(value.shape) != spec.shape
                    or value.dtype != spec.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"invalid initialized tensor {spec.name}: require finite CPU tensor with exact shape/dtype")
            if not torch.equal(value, result[spec.alias_of]):
                raise ValueError(f"conflicting tied initialization values: {spec.alias_of}, {spec.name}")
        module.load_state_dict(result, strict=True)
        return {"method": method, **record, "training_verified": False}
    except Exception as exc:
        if isinstance(exc, WgtError):
            raise
        raise WgtError(f"initialization failed: {exc}", path=program.path) from exc
