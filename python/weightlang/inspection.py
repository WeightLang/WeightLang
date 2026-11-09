"""Model inspection built on torchinfo, torch.fx, and Graphviz."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import fx, nn


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


@contextmanager
def _preserve_rng(model: nn.Module, inputs: Any = None, device: Any = None):
    """Preserve torch RNGs on the CPU and the inspected tensors' devices."""
    devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
    devices.update(tensor.device for tensor in _tensors(inputs))
    if device is not None:
        devices.add(torch.device(device))
    cpu_state = torch.random.get_rng_state()
    states = []
    for target in devices:
        if target.type in {"cuda", "xpu", "mps"}:
            backend = getattr(torch, target.type)
            states.append((backend, target, backend.get_rng_state(target)))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        for backend, target, state in states:
            backend.set_rng_state(state, target)


def _copy_model(model: nn.Module) -> nn.Module:
    try:
        return copy.deepcopy(model)
    except Exception as error:
        raise TypeError(
            "Inspection requires a model that supports copy.deepcopy so that "
            "its parameters, buffers, and training modes remain unchanged."
        ) from error


def _copy_inputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_inputs(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_inputs(child) for child in value)
    if isinstance(value, list):
        return [_copy_inputs(child) for child in value]
    return copy.deepcopy(value)


def summary(
    model: nn.Module,
    input_size: Any = None,
    *,
    input_data: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a ``torchinfo.ModelStatistics`` report without printing by default.

    With no ``input_size`` or ``input_data``, report registered layers and
    parameters without running forward. Supplying either opts into a forward
    pass on a copy in eval mode for shapes and torchinfo's operation estimates
    (pass ``mode="train"`` to inspect training). The original
    model, input tensors, and torch RNG states are preserved. Other keyword
    arguments are passed to ``torchinfo.summary``; its forward cache is disabled.
    From a checkout, install with ``pip install './python[visualize]'``;
    an existing environment can also install ``torchinfo numpy`` directly.
    """
    try:
        from torchinfo import summary as torchinfo_summary
    except ImportError as error:
        raise ImportError(
            "summary() requires torchinfo and numpy; run 'pip install torchinfo numpy' "
            "or install './python[visualize]' from a WeightLang checkout."
        ) from error
    if kwargs.get("cache_forward_pass"):
        raise ValueError("summary() disables torchinfo's forward cache to avoid stale model reports")
    kwargs["cache_forward_pass"] = False
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("mode", "eval")
    if input_size is None and input_data is None:
        return torchinfo_summary(model, **kwargs)
    with _preserve_rng(model, (input_data, kwargs), kwargs.get("device")):
        return torchinfo_summary(
            _copy_model(model),
            input_size=input_size,
            input_data=_copy_inputs(input_data),
            **_copy_inputs(kwargs),
        )


def trace(model: nn.Module, *, concrete_args: dict[str, Any] | None = None) -> fx.GraphModule:
    """Return the model's PyTorch FX graph.

    An existing ``GraphModule`` is returned unchanged. Other modules are
    symbolically traced on a copy, preserving their weights, buffers, training
    modes, and torch RNG states. ``concrete_args`` specializes Python arguments
    using FX's normal rules. Runtime-control/stateful WGT modules expose their
    compiler IR through ``visualize``; they have no single static FX graph.
    """
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, nn.Module):
        model = original
    if isinstance(model, fx.GraphModule):
        if concrete_args:
            raise ValueError("concrete_args applies to symbolic tracing; this model already has an FX graph")
        return model
    definition = getattr(model, "definition", None)
    if isinstance(definition, dict) and definition.get("kind") in {"forward", "operations"}:
        raise ValueError(
            "This WeightLang module uses runtime control flow or typed operations "
            "and has no single static FX graph. Use visualize(model) to inspect its compiler IR."
        )
    with _preserve_rng(model, concrete_args):
        try:
            return fx.symbolic_trace(_copy_model(model), concrete_args=_copy_inputs(concrete_args))
        except fx.proxy.TraceError as error:
            raise ValueError(
                "torch.fx could not trace this model. Data-dependent Python control "
                f"flow is unsupported; use traceable tensor operations in forward. {error}"
            ) from error


def _label(model: fx.GraphModule, node: fx.Node) -> str:
    if node.op == "call_module":
        return f"{node.target}: {type(model.get_submodule(str(node.target))).__name__}"
    if node.op == "call_function":
        return f"{node.name}: {getattr(node.target, '__name__', str(node.target))}"
    label = f"{node.op}: {node.target}"
    if node.op == "get_attr":
        value: Any = model
        for part in str(node.target).split("."):
            value = getattr(value, part)
        if isinstance(value, torch.Tensor):
            label += f" {tuple(value.shape)}"
    return label


def _mermaid_label(text: str) -> str:
    return "".join(
        f"#{ord(char)};" if char in '\"<>&#' else " " if char in "\n\r" else char
        for char in text
    )


def _render_graph(
    nodes: list[tuple[str, str]],
    edges: list[tuple[str, str, str, bool]],
    format: str,
    title: str | None = None,
) -> str:
    if format == "mermaid":
        lines = ["flowchart TD"]
        if title:
            lines.append(f"  %% {' '.join(title.splitlines())}")
        lines.extend(f'  {name}["{_mermaid_label(label)}"]' for name, label in nodes)
        for source, target, label, storage in edges:
            arrow = f"-. {label} .->" if storage else f"-->|{label}|" if label else "-->"
            lines.append(f"  {source} {arrow} {target}")
    else:
        lines = [
            "digraph WeightLang {",
            '  graph [rankdir=TB, bgcolor="white", pad="0.3", nodesep="0.35", ranksep="0.55"];',
            '  node [shape=box, style=rounded, fontname="Helvetica", color="#dadce0", margin="0.16,0.12"];',
            '  edge [color="#5f6368", arrowsize="0.7"];',
        ]
        if title:
            lines.append(f"  label={json.dumps(title, ensure_ascii=False)};")
            lines.append('  labelloc="t"; fontname="Helvetica";')
        lines.extend(
            f"  {name} [label={json.dumps(label, ensure_ascii=False)}];"
            for name, label in nodes
        )
        for source, target, label, storage in edges:
            attributes = ([f"label={json.dumps(label)}"] if label else []) + (["style=dashed"] if storage else [])
            suffix = f" [{', '.join(attributes)}]" if attributes else ""
            lines.append(f"  {source} -> {target}{suffix};")
        lines.append("}")
    return "\n".join(lines) + "\n"


def _graph_text(model: fx.GraphModule, format: str) -> str:
    buffers = dict(model.named_buffers(remove_duplicate=False))
    # The diagnostic anchor only enables gradient checks; it is not model data.
    # Match buffers (including qualified names in compositions), never inputs or
    # parameters, and leave the executable FX graph untouched.
    nodes = [node for node in model.graph.nodes if not (
        node.op == "get_attr" and node.target in buffers
        and str(node.target).rsplit(".", 1)[-1] == "_wgt_gradient_anchor"
    )]
    identifiers = {node: f"n{index}" for index, node in enumerate(nodes)}
    return _render_graph(
        [(identifiers[node], _label(model, node)) for node in nodes],
        [(identifiers[parent], identifiers[node], "", False)
         for node in nodes for parent in node.all_input_nodes if parent in identifiers],
        format,
    )


def _instruction_label(item: dict, shapes: dict[str, list[int]]) -> str:
    op = item["op"]
    if op == "read":
        label = f"{item['out']} = read {item['parameter']}"
    elif op == "write":
        label = f"{item['parameter']} <- {item['value']}"
    elif op == "branch":
        label = f"branch {item['condition']} > 0"
    elif op == "inference_skip":
        label = f"inference skip when {item['probability']} == {int(item['when_one'])}"
    else:
        arguments = []
        for key, value in item.items():
            if key in {"op", "out"}:
                continue
            if key == "data" and len(value) > 6:
                value = f"{len(value)} constants"
            arguments.append(f"{key}={value}")
        label = f"{op}({', '.join(arguments)})"
        if "out" in item:
            label = f"{item['out']} = {label}"
    if item.get("out") in shapes:
        label += f" : {shapes[item['out']]}"
    return label


def _ir_text(definition: dict, format: str, entry: str | None) -> str:
    """Draw static control flow and storage dependencies, without running IR."""
    kind = definition["kind"]
    if kind == "operations":
        plan = definition["operations"]
        programs = [(item["schema"]["name"], item["program"], item["schema"]) for item in plan["entries"]]
        parameters, states = plan["parameters"], plan["states"]
    else:
        program = definition["program"]
        programs = [("forward", program, None)]
        if program.get("loss") is not None:
            programs.append(("loss", program["loss"], None))
        parameters, states = program["parameters"], []
        entry = entry or "forward"
    if entry is not None:
        choices = [name for name, _, _ in programs]
        if entry not in choices:
            raise ValueError(f"unknown WGT entry {entry!r}; available: {', '.join(choices)}")
        programs = [item for item in programs if item[0] == entry]
    nodes, edges = [], []
    storage = {}
    for storage_kind, specs in (("parameter", parameters), ("state", states)):
        for spec in specs:
            identifier = f"storage_{len(storage)}"
            storage[spec["name"]] = identifier
            nodes.append((identifier, f"{storage_kind} {spec['name']}: {spec['shape']}"))
    model_name = definition.get("name", "WeightLangModel")
    for number, (name, program, schema) in enumerate(programs):
        prefix = f"entry_{number}"
        shapes = {register["name"]: register["shape"] for register in program["registers"]}
        signature = name
        if schema is not None:
            ports = ", ".join(f"{port['name']}: f32[{port['width']}]" for port in schema["inputs"])
            signature = f"{name}({ports}) -> " + (f"f32[{schema['output']}]" if schema["output"] else "void")
        nodes.append((prefix, f"{model_name}.{signature}"))
        instructions = program["instructions"]
        labels = {item["name"]: f"{prefix}_{index}" for index, item in enumerate(instructions) if item["op"] == "label"}
        if instructions:
            edges.append((prefix, f"{prefix}_0", "", False))
        for index, item in enumerate(instructions):
            identifier = f"{prefix}_{index}"
            nodes.append((identifier, _instruction_label(item, shapes)))
            op = item["op"]
            if op == "branch":
                edges.extend((identifier, labels[item[key]], label, False)
                             for key, label in (("positive", "yes"), ("otherwise", "no")))
            elif op == "jump":
                edges.append((identifier, labels[item["target"]], "jump", False))
            else:
                if op == "inference_skip":
                    edges.append((identifier, labels[item["target"]], "skip", False))
                if op != "return" and index + 1 < len(instructions):
                    edges.append((identifier, f"{prefix}_{index + 1}", "", False))
            if op == "write":
                edges.append((identifier, storage[item["parameter"]], "write", True))
            if op == "read":
                edges.append((storage[item["parameter"]], identifier, "read", True))
            if op == "linear":
                for key in ("weight", "bias"):
                    if item.get(key) in storage:
                        edges.append((storage[item[key]], identifier, "read", True))
    title = f"{model_name} — WGT compiler IR; solid edges: control flow; dashed edges: storage"
    return _render_graph(nodes, edges, format, title)


def visualize(
    model: nn.Module,
    path: str | PathLike[str] | None = None,
    *,
    format: str | None = None,
    overwrite: bool = False,
    entry: str | None = None,
) -> str | bytes | Path:
    """Render tensor dependencies or WGT control flow as Mermaid, DOT, SVG, or PNG.

    Static FX modules show tensor dependencies. WGT modules with runtime control
    flow or persistent state show the compiler's complete control-flow graph and
    storage reads/writes, preserving WGT names and including untaken branches.
    ``entry`` selects a typed operation or a forward model's ``"loss"``; without
    it, all typed operations or the model's forward graph are shown.

    With no path, return text (Mermaid, DOT, SVG) or PNG bytes. With a path,
    write exactly that file and return its ``Path``; the parent must exist and
    replacing a file requires ``overwrite=True``. Infer the format from a path's
    suffix, otherwise default to ``mermaid``. Mermaid (``.mmd``) and DOT need
    only PyTorch. SVG and PNG also need the Python ``graphviz`` package and the system
    Graphviz ``dot`` executable. No viewer is opened and no example is executed
    for either an existing FX graph or a compiler IR diagram. Arbitrary PyTorch
    modules use symbolic FX tracing and retain FX's control-flow limitations.
    """
    destination = Path(path) if path is not None else None
    suffix = destination.suffix.lstrip(".").lower() if destination else ""
    format = (format or suffix or "mermaid").lower()
    format = {"mmd": "mermaid", "gv": "dot"}.get(format, format)
    if format not in {"mermaid", "dot", "svg", "png"}:
        raise ValueError("format must be mermaid, dot, svg, or png")
    if destination is not None and destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass overwrite=True to replace it")
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, nn.Module):
        model = original
    definition = getattr(model, "definition", None)
    graph_format = "mermaid" if format == "mermaid" else "dot"
    if (isinstance(definition, dict) and definition.get("kind") in {"forward", "operations"}
            and (entry is not None or not isinstance(model, fx.GraphModule))):
        result: str | bytes = _ir_text(definition, graph_format, entry)
    else:
        if entry not in (None, "forward"):
            raise ValueError("entry selects a WGT operation or loss; this module only exposes its forward graph")
        result = _graph_text(trace(model), graph_format)
    if format in {"svg", "png"}:
        try:
            import graphviz
        except ImportError as error:
            raise ImportError(
                "SVG/PNG rendering requires graphviz; run 'pip install graphviz' "
                "or install './python[visualize]' from a WeightLang checkout."
            ) from error
        try:
            rendered = graphviz.Source(result).pipe(format=format)
        except graphviz.ExecutableNotFound as error:
            raise RuntimeError(
                "SVG/PNG rendering requires the system Graphviz 'dot' executable. "
                "Install Graphviz or use format='mermaid' or format='dot'."
            ) from error
        result = rendered.decode("utf-8") if format == "svg" else rendered
    if destination is None:
        return result
    mode = ("w" if overwrite else "x") + ("b" if isinstance(result, bytes) else "")
    with destination.open(mode, **({} if isinstance(result, bytes) else {"encoding": "utf-8"})) as output:
        output.write(result)
    return destination
