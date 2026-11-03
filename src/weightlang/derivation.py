"""Declarative model derivation: immutable baseline, disjoint tensor assignments."""

from __future__ import annotations

import copy
import math
import operator
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import syntax as syn
from .artifact import build_module, load_artifact, sha256_file

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
DOTTED = rf"{IDENT}(?:\.{IDENT})*"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
HEADER = re.compile(rf'model\s+({IDENT})\s+extends\s+"([^"\\]+)"\s+sha256="([a-f0-9]{{64}})"\s*:')
REFERENCE = rf"({DOTTED})(\[[^\]]+\])?"


@dataclass
class Assignment:
    target: str
    selection: str | None
    expression: str
    line: int


@dataclass
class WeightGroup:
    name: str
    line: int
    default: str | None = None
    assignments: list[Assignment] = field(default_factory=list)


@dataclass
class Derivation:
    source: str
    sha256: str
    line: int
    replaced: list[str]
    removed: list[str]
    inserted: dict[str, tuple[str, str]]
    groups: dict[str, WeightGroup]


def is_derived(source):
    return any(re.match(r"model\s+\w+\s+extends\b", syn._strip_comment(line).strip()) for line in source.splitlines())


def _error(message, line, path):
    return syn.WgtError(message, line, path)


def _load(spec, path):
    root = (Path(path).parent if path else Path.cwd()) / spec.source
    try:
        if sha256_file(root / "manifest.json") != spec.sha256:
            raise _error("derived model baseline manifest sha256 mismatch", spec.line, path)
        with torch.random.fork_rng(devices=[]):
            artifact = load_artifact(root, warn_untrained=False)
        if sha256_file(root / "manifest.json") != spec.sha256:
            raise _error("derived model baseline changed while reading", spec.line, path)
        return artifact
    except OSError as exc:
        raise _error(f"cannot read derived model baseline: {exc}", spec.line, path) from exc


def _decode(item, line, path):
    fields = {
        k: copy.deepcopy(item[k])
        for k in ("name", "op", "args", "block", "inputs")
        if k in item and k in syn.Layer.__dataclass_fields__
    }
    return syn.Layer(**fields, line=line, path=path, children=[_decode(c, line, path) for c in item.get("layers", [])])


def _siblings(layers, target, line, path):
    parent, _, leaf = target.rpartition(".")
    if not parent:
        return layers, leaf
    layer = syn.find_layer(layers, parent)
    if layer is None or not layer.children:
        raise _error(f"unknown parent layer {parent!r}", line, path)
    return layer.children, leaf


def parse(source, path=None, *, check_training=True):
    physical = source.splitlines()
    lines = list(syn.logical_lines(source, syn._strip_comment, lambda message, line: _error(message, line, path)))
    header = next(((n, text) for n, text, _ in lines if text.strip().startswith("model ")), None)
    if header is None or not (match := HEADER.fullmatch(header[1])):
        raise _error('expected model NAME extends "ARTIFACT" sha256="MANIFEST_HASH":', header[0] if header else 1, path)
    number, _ = header
    model, baseline, digest = match.groups()
    spec = Derivation(baseline, digest, number, [], [], {}, {})
    regular = [""] * len(physical)
    regular[number - 1] = f"model {model}"
    targets = {}
    group = None
    body_indent = None
    declarations = set()
    for lineno, text, _ in lines:
        if not text.strip() or lineno == number:
            continue
        if lineno < number:
            regular[lineno - 1] = text
            continue
        if "\t" in text[: len(text) - len(text.lstrip())]:
            raise _error("derived model indentation must use spaces", lineno, path)
        indent = len(text) - len(text.lstrip())
        if body_indent is None:
            body_indent = indent
        if indent < body_indent or not body_indent:
            raise _error("derived model body must be indented", lineno, path)
        text = text[body_indent:]
        if group is not None and text.startswith(" "):
            statement = text.strip()
            if statement.startswith("default:"):
                if group.default is not None:
                    raise _error("duplicate weight default", lineno, path)
                group.default = statement[len("default:") :].strip()
            else:
                edit = re.fullmatch(REFERENCE + r"\s*=\s*(.+)", statement)
                if not edit:
                    raise _error(
                        "expected tensor[selection] = value or base.tensor[selection] expression", lineno, path
                    )
                group.assignments.append(Assignment(f"{group.name}.{edit[1]}", edit[2], edit[3], lineno))
            continue
        group = None
        if weight := re.fullmatch(rf"weights\s+({DOTTED})\s*:", text):
            if weight[1] in spec.groups:
                raise _error(f"duplicate weights group {weight[1]!r}", lineno, path)
            group = WeightGroup(weight[1], lineno)
            spec.groups[group.name] = group
        elif removal := re.fullmatch(rf"remove\s+layer\s+({DOTTED})", text):
            target = removal[1]
            if target in declarations:
                raise _error(f"conflicting layer declaration {target!r}", lineno, path)
            declarations.add(target)
            spec.removed.append(target)
        elif layer := re.fullmatch(rf"layer\s+({DOTTED})(?:\s+(before|after)\s+({IDENT}))?\s*:\s*(.+)", text):
            target, side, anchor, expression = layer.groups()
            if target in declarations:
                raise _error(f"conflicting layer declaration {target!r}", lineno, path)
            declarations.add(target)
            alias = f"derived_layer_{len(targets)}"
            targets[alias] = target
            if side:
                if "." in target:
                    raise _error("new layers must be top-level", lineno, path)
                spec.inserted[target] = (side, anchor)
            else:
                spec.replaced.append(target)
            regular[lineno - 1] = f"layer {alias}: {expression}"
        else:
            regular[lineno - 1] = text
    fp = syn._FileParser("\n".join(regular), path, syn._Loader(), library=False)
    fp.run()
    if fp.version != "0.3":
        raise _error("model derivation requires 'wgt 0.3'", number, path)
    if fp.imports or fp.inherit:
        raise _error("derived models use explicit weights sources, not import or inherit", number, path)
    ancestor = _load(spec, path)
    manifest = ancestor.manifest
    layers = [_decode(item, number, path) for item in manifest["layers"]]
    baseline_names = {layer.name for layer in layers}
    changed = spec.replaced + spec.removed
    for a in changed:
        if any(a != b and a.startswith(b + ".") for b in changed):
            raise _error("overlapping parent/child layer declarations", number, path)
        if syn.find_layer(layers, a) is None:
            raise _error(f"unknown baseline layer {a!r}; new layers need before/after", number, path)
    for target in sorted(spec.removed):
        siblings, leaf = _siblings(layers, target, number, path)
        siblings[:] = [layer for layer in siblings if layer.name != leaf]
    insertions = {}
    for alias, unresolved in fp.layer_specs:
        if alias not in targets:
            raise _error("derived layer declarations must be inside the model body", unresolved.line, path)
        target = targets[alias]
        resolved = syn._resolve(target.rsplit(".", 1)[-1], unresolved, fp.scope, {}, ())
        if target in spec.inserted:
            side, anchor = spec.inserted[target]
            if target in baseline_names or anchor not in baseline_names or anchor in spec.removed:
                raise _error("insertions require a new name and a retained baseline anchor", unresolved.line, path)
            if (side, anchor) in insertions:
                raise _error("multiple insertions at the same anchor are ambiguous", unresolved.line, path)
            insertions[(side, anchor)] = resolved
        else:
            siblings, leaf = _siblings(layers, target, unresolved.line, path)
            siblings[:] = [resolved if layer.name == leaf else layer for layer in siblings]
    expanded = []
    for layer in layers:
        if before := insertions.get(("before", layer.name)):
            expanded.append(before)
        expanded.append(layer)
        if after := insertions.get(("after", layer.name)):
            expanded.append(after)
    if not expanded:
        raise _error("derived model has no layers", number, path)
    ports = {
        role: fp.ports.get(role)
        or syn.Port(
            **{**manifest["interface"][role], "shape": tuple(manifest["interface"][role]["shape"]), "line": number}
        )
        for role in ("input", "output")
    }
    program = syn.Program(
        version="0.3",
        model=model,
        input=ports["input"],
        output=ports["output"],
        layers=expanded,
        datasets=fp.datasets,
        seed=fp.seed if fp.seed is not None else manifest["init"]["seed"],
        train=fp.train,
        path=path,
        uses=fp.uses,
        freezing=fp.freezing,
        derivation=spec,
    )
    for ds in program.datasets.values():
        if ds.file and check_training:
            syn._load_csv(ds, math.prod(program.input.shape), path)
    syn.check(program, check_training=check_training)
    _plan(program, ancestor)
    return program


def _selection(text, shape, error):
    parts = text[1:-1].split(",") if text else [":"] * len(shape)
    if len(parts) != len(shape):
        raise error("selection must specify exactly one index or slice per dimension")
    indices, bounds, selected = [], [], []
    for part, size in zip(parts, shape):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            index = int(part)
            if index >= size:
                raise error("weight index out of bounds")
            indices.append(index)
            bounds.append((index, index + 1))
        elif re.fullmatch(r"\d*:\d*", part):
            start, stop = part.split(":")
            lo, hi = int(start or 0), int(stop) if stop else size
            if not 0 <= lo < hi <= size:
                raise error("weight slice is empty or out of bounds")
            indices.append(slice(lo, hi))
            bounds.append((lo, hi))
            selected.append(hi - lo)
        else:
            raise error("weight selection supports non-negative indices and start:stop slices")
    return tuple(indices), bounds, tuple(selected)


def _rhs(expression, baseline, error):
    if re.fullmatch(NUMBER, expression):
        number = float(expression)
        if not math.isfinite(number) or abs(number) > torch.finfo(torch.float32).max:
            raise error("weight value must be finite float32")
        return number, None
    match = re.fullmatch(r"base\." + REFERENCE + rf"(?:\s*([+*/-])\s*({NUMBER}))?", expression)
    if not match or match[1] not in baseline:
        raise error("expected a number or an existing base.tensor with optional scalar arithmetic")
    name, selection, op, operand = match.groups()
    indices, _, shape = _selection(selection, baseline[name].shape, error)
    if op and (not math.isfinite(float(operand)) or (op == "/" and float(operand) == 0)):
        raise error("invalid scalar arithmetic in weight expression")
    return (name, indices, op, float(operand) if operand else None), shape


def _plan(program, ancestor):
    spec = program.derivation

    def error(message):
        return _error(message, spec.line, program.path)

    baseline = ancestor.module.state_dict()
    with torch.device("meta"):
        module = build_module([layer.to_dict() for layer in program.layers])
    state = module.state_dict()
    defaults, assignments, covered = {}, {}, {}
    for group_name, group in sorted(spec.groups.items()):

        def fail(message):
            return _error(message, group.line, program.path)

        layer = program.find(group_name)
        if layer is None or not layer.has_params:
            raise fail(f"weights group {group_name!r} has no parameter layer")
        members = [name for name in state if name.startswith(group_name + ".")]
        if not group.default and not group.assignments:
            raise fail("empty weights group")
        if group.default:
            init = re.fullmatch(r"init\(seed=(\d+)\)", group.default)
            origin = re.fullmatch(r"base\.(" + DOTTED + r")", group.default)
            if not init and not origin:
                raise fail("weight default must be init(seed=N) or base.layer")
            if init and int(init[1]) >= 2**64:
                raise fail("initialization seed must be below 2**64")
            for name in members:
                if name in defaults:
                    raise fail("overlapping weight defaults")
                if init:
                    defaults[name] = ("init", group_name, int(init[1]))
                else:
                    source = origin[1] + name[len(group_name) :]
                    if (
                        source not in baseline
                        or baseline[source].shape != state[name].shape
                        or baseline[source].dtype != state[name].dtype
                    ):
                        raise fail(f"default source shape/type mismatch for {name}")
                    defaults[name] = ("base", source)
        for assignment in group.assignments:

            def fail(message):
                return _error(message, assignment.line, program.path)

            name = assignment.target
            if name not in state:
                raise fail(f"unknown weight tensor {name!r}")
            indices, bounds, shape = _selection(assignment.selection, state[name].shape, fail)
            rhs, rhs_shape = _rhs(assignment.expression, baseline, fail)
            if rhs_shape is not None and rhs_shape != shape:
                raise fail(f"weight assignment shape mismatch: target {shape}, source {rhs_shape}; no broadcasting")
            prior = covered.setdefault(name, [])
            if any(all(max(a, c) < min(b, d) for (a, b), (c, d) in zip(bounds, other)) for other in prior):
                raise fail(f"overlapping weight assignments for {name}")
            prior.append(bounds)
            assignments.setdefault(name, []).append((bounds, indices, rhs))
    inherited = []
    modified = spec.replaced + list(spec.inserted)
    for name, tensor in state.items():
        untouched = not any(name.startswith(prefix + ".") for prefix in modified)
        if (
            name not in defaults
            and untouched
            and name in baseline
            and baseline[name].shape == tensor.shape
            and baseline[name].dtype == tensor.dtype
        ):
            inherited.append(name)
        elif name not in defaults:
            coverage = sum(math.prod(b - a for a, b in bounds) for bounds in covered.get(name, []))
            if coverage != tensor.numel():
                raise error(f"weight {name!r} has no complete source; declare a default or cover every element")
    return state, baseline, defaults, assignments, inherited


def protect_output(program, destination):
    baseline = (Path(program.path).parent / program.derivation.source).resolve()
    destination = Path(destination).resolve()
    if baseline.is_relative_to(destination) or destination.is_relative_to(baseline):
        raise _error(
            "derived output must be separate from its immutable baseline directory",
            program.derivation.line,
            program.path,
        )


def apply(program, module):
    spec = program.derivation
    ancestor = _load(spec, program.path)
    _, baseline, defaults, assignments, inherited = _plan(program, ancestor)
    state = module.state_dict()
    initialized = {}
    for group in sorted({value[1] for value in defaults.values() if value[0] == "init"}):
        seed = next(value[2] for value in defaults.values() if value[:2] == ("init", group))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            layer = program.find(group)
            fresh = build_module([layer.to_dict()]).state_dict()
        for name, tensor in fresh.items():
            initialized[group + name[len(layer.name) :]] = tensor
    actions = {}
    with torch.no_grad():
        for name, tensor in state.items():
            source = None
            if name in defaults:
                default = defaults[name]
                if default[0] == "init":
                    tensor.copy_(initialized[name])
                    action = "initialized"
                else:
                    source = default[1]
                    tensor.copy_(baseline[source])
                    action = "inherited"
            elif name in inherited:
                source = name
                tensor.copy_(baseline[name])
                action = "inherited"
            else:
                action = "assigned"
            for bounds, indices, rhs in sorted(assignments.get(name, []), key=lambda item: item[0]):
                if isinstance(rhs, float):
                    tensor[indices] = rhs
                else:
                    ref, selection, op, operand = rhs
                    value = baseline[ref][selection]
                    if op:
                        value = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}[op](
                            value, operand
                        )
                    tensor[indices] = value
                action = "assigned"
            if not torch.isfinite(tensor).all():
                raise _error(f"derived weights contain non-finite values in {name}", spec.line, program.path)
            actions[name] = {"action": action}
            if source is not None:
                meta = ancestor.manifest["weights"]["tensors"][source]
                actions[name]["source"] = source
                actions[name]["upstream_origin"] = (
                    meta.get("upstream_origin", meta["origin"]) if meta["origin"] == "imported" else meta["origin"]
                )
            if name in baseline and baseline[name].shape == tensor.shape:
                actions[name]["changed_elements"] = int((tensor != baseline[name]).sum())
    return {
        "format": "weightlang.derivation",
        "format_version": 1,
        "stage": "before_training",
        "baseline": {
            "path": spec.source,
            "manifest_sha256": spec.sha256,
            "weights_sha256": ancestor.manifest["weights"]["sha256"],
        },
        "structure": {
            "replaced": sorted(spec.replaced),
            "removed": sorted(spec.removed),
            "inserted": dict(sorted(spec.inserted.items())),
        },
        "weights": {key: asdict(value) for key, value in sorted(spec.groups.items())},
        "tensors": actions,
        "dropped_tensors": sorted(set(baseline) - set(state)),
    }


def record(manifest, derivation, trainable):
    manifest["derivation"] = derivation
    for name, info in derivation["tensors"].items():
        meta = manifest["weights"]["tensors"][name]
        trained = manifest["training"] is not None and name in trainable
        if info["action"] == "inherited":
            meta["origin"] = "imported+trained" if trained else "imported"
            meta["upstream_origin"] = info["upstream_origin"]
        else:
            meta["origin"] = "trained" if trained else "random-init" if info["action"] == "initialized" else "edited"


def preview(program):
    """Compute the derived state in memory, without training or publishing files."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(program.seed)
        module = build_module([layer.to_dict() for layer in program.layers])
        report = apply(program, module)
    report["execution"] = "planned"
    report["training_requested"] = program.train is not None
    report["architecture"] = [layer.to_dict() for layer in program.layers]
    return report
