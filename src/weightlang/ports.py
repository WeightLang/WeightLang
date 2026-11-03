"""Checked named tensor boundaries shared by source construction and artifacts."""
from __future__ import annotations

import re

from .syntax import Port, WgtError, output_contract


def interface(program):
    """Return the explicit boundary IR, or None for legacy tensor-returning graphs."""
    multi_results = any(layer.results for layer in program.layers)
    if not (getattr(program,"weight_types",{}) or getattr(program,"weight_assignments",[]) or multi_results or getattr(program,"adapters",{}) or getattr(program,"parameter_ids",{}) or getattr(program,"parameter_shares",{}) or program.dimensions or any(not p.shape or 0 in p.shape for p in program.inputs + program.outputs) or len(program.inputs) > 1 or len(program.outputs) > 1 or program.returns):
        return None
    def encode(port):
        result = {"name": port.name, "dtype": port.dtype, "shape": list(port.shape)}
        if program.dimensions:
            result["shape_expr"] = list(port.shape_expr)
        if port.axes:
            result["axes"] = list(port.axes)
        return result
    result = {"inputs": [encode(p) for p in program.inputs],
              "outputs": [{**encode(p), "from": program.returns.get(p.name,program.layers[-1].name)} for p in program.outputs]}
    if program.dimensions:
        result.update(dimensions=program.dimensions, constraints=program.constraints)
    if program.weight_types or program.weight_assignments or getattr(program,"adapters",{}) or getattr(program,"parameter_ids",{}) or getattr(program,"parameter_shares",{}):
        result.update(parameter_ids=program.parameter_ids, parameter_shares=program.parameter_shares)
    if getattr(program,"adapters",{}):
        result["adapters"] = program.adapters
    if program.weight_types or program.weight_assignments:
        result.update(weight_types=program.weight_types, weight_assignments=program.weight_assignments)
    return result


def check_ports(program):
    inputs, outputs = program.inputs, program.outputs
    names = [p.name for p in inputs + outputs]
    if len(names) != len(set(names)):
        raise WgtError("duplicate interface name", path=program.path)
    if any(p.name == "input" for p in inputs[1:]):
        raise WgtError("input is reserved for the first port", path=program.path)
    if program.train or program.datasets:
        raise WgtError("named interfaces require Python data preparation and training", path=program.path)
    if program.inherit or program.derivation:
        raise WgtError("named interface inherit/derivation is not supported", path=program.path)
    if any(layer.name in {p.name for p in inputs} for layer in program.layers):
        raise WgtError("layer name conflicts with an input port", path=program.path)
    bindings = {p.name: (p.dtype, p.shape) for p in inputs[1:]}
    output_contract(program.layers, inputs[0].dtype, inputs[0].shape, bindings,
                    axes=inputs[0].axes, binding_axes={p.name: list(p.axes or [None] * len(p.shape)) for p in inputs[1:]})
    values = {p.name: {"dtype": p.dtype, "shape": list(p.shape), "axes": list(p.axes)} for p in inputs}
    values["input"] = values[inputs[0].name]
    values.update({layer.name: layer.output_spec for layer in program.layers})
    values.update({f"{layer.name}.{field}": spec for layer in program.layers for field,spec in layer.results.items()})
    if len(outputs) > 1 and set(program.returns) != {p.name for p in outputs}:
        raise WgtError("every output in a multi-output model requires 'from tensor'", path=program.path)
    for port in outputs:
        ref = program.returns.setdefault(port.name, program.layers[-1].name)
        if ref == inputs[0].name:
            ref = program.returns[port.name] = "input"
        if ref not in values:
            raise WgtError(f"unknown output tensor {ref!r}", port.line, program.path)
        value = values[ref]
        axes = value.get("axes") or [None] * len(value["shape"])
        if value["dtype"] != port.dtype or tuple(value["shape"]) != port.shape or any(a and b and a != b for a, b in zip(axes, port.axes)):
            raise WgtError(f"output {port.name!r} shape/dtype/axis mismatch", port.line, program.path)


def validate_interface(record, layers):
    """Decode untrusted manifest metadata and re-infer every named output."""
    from types import SimpleNamespace
    def decode(items):
        if not isinstance(items, list) or not items:
            raise WgtError("named interface requires nonempty port lists")
        result = []
        for item in items:
            name, dtype, shape = item["name"], item["dtype"], item["shape"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise WgtError("invalid port name")
            if dtype not in {"f32", "i64", "bool"} or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise WgtError("invalid port dtype/shape")
            axes = item.get("axes", ())
            output_contract([], dtype, tuple(shape), axes=axes)
            result.append(Port(name, dtype, tuple(shape), 0, tuple(axes), tuple(item.get("shape_expr", ()))))
        return result
    inputs, outputs = decode(record["inputs"]), decode(record["outputs"])
    returns = {p["name"]: p["from"] for p in record["outputs"]}
    if not all(isinstance(v, str) for v in returns.values()):
        raise WgtError("output references must be names")
    program = SimpleNamespace(inputs=inputs, outputs=outputs, returns=returns, layers=layers,
                              path=None, train=None, datasets={}, inherit=None, derivation=None,
                              dimensions=record.get("dimensions", {}), constraints=record.get("constraints", []))
    if program.dimensions:
        if "constraints" not in record:
            raise WgtError("symbolic interface requires a constraints list")
        from .symbolic import check_symbolic
        original = [p.shape for p in inputs + outputs]
        if any(len(p.shape_expr) != len(p.shape) for p in inputs + outputs):
            raise WgtError("symbolic shape_expr must match port rank")
        check_symbolic(program)
        if original != [p.shape for p in inputs + outputs]:
            raise WgtError("symbolic representative shape mismatch")
        layers[:] = program.layers
    else:
        check_ports(program)


def build_named_module(layers, record):
    from .artifact import build_module
    from .dataflow import NamedFlow
    module = build_module(layers)
    # Keep all existing parameter paths and aliases: no wrapper module prefix.
    named = NamedFlow(layers, record)
    named._modules.update(module._modules)
    from .parameter_identity import bind_parameters
    try:
        bind_parameters(named,record.get('parameter_ids',{}),record.get('parameter_shares',{}))
        from .adapters import attach_adapters
        attach_adapters(named,record.get('adapters',{}))
        from .weight_types import attach_weight_types
        attach_weight_types(named, record.get('weight_types',{}), record.get('weight_assignments',[]))
    except ValueError as exc:
        raise WgtError(str(exc)) from None
    return named
