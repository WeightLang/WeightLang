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
    using FX's normal rules. Data-dependent Python control flow is unsupported.
    """
    if isinstance(model, fx.GraphModule):
        if concrete_args:
            raise ValueError("concrete_args applies to symbolic tracing; this model already has an FX graph")
        return model
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


def _graph_text(model: fx.GraphModule, format: str) -> str:
    nodes = list(model.graph.nodes)
    identifiers = {node: f"n{index}" for index, node in enumerate(nodes)}
    if format == "mermaid":
        lines = ["flowchart TD"]
        lines.extend(f'  {identifiers[node]}["{_mermaid_label(_label(model, node))}"]' for node in nodes)
        lines.extend(
            f"  {identifiers[parent]} --> {identifiers[node]}"
            for node in nodes for parent in node.all_input_nodes
        )
    else:
        lines = [
            "digraph WeightLang {",
            '  graph [rankdir=TB, bgcolor="white", pad="0.3", nodesep="0.35", ranksep="0.55"];',
            '  node [shape=box, style=rounded, fontname="Helvetica", color="#dadce0", margin="0.16,0.12"];',
            '  edge [color="#5f6368", arrowsize="0.7"];',
        ]
        lines.extend(
            f"  {identifiers[node]} [label={json.dumps(_label(model, node), ensure_ascii=False)}];"
            for node in nodes
        )
        lines.extend(
            f"  {identifiers[parent]} -> {identifiers[node]};"
            for node in nodes for parent in node.all_input_nodes
        )
        lines.append("}")
    return "\n".join(lines) + "\n"


def visualize(
    model: nn.Module,
    path: str | PathLike[str] | None = None,
    *,
    format: str | None = None,
    overwrite: bool = False,
) -> str | bytes | Path:
    """Render actual FX dependencies as Mermaid, DOT, SVG, or PNG.

    With no path, return text (Mermaid, DOT, SVG) or PNG bytes. With a path,
    write exactly that file and return its ``Path``; the parent must exist and
    replacing a file requires ``overwrite=True``. Infer the format from a path's
    suffix, otherwise default to ``mermaid``. Mermaid (``.mmd``) and DOT need
    only PyTorch. SVG and PNG also need the Python ``graphviz`` package and the system
    Graphviz ``dot`` executable. No viewer is opened and no example is executed
    for an existing FX graph.
    """
    destination = Path(path) if path is not None else None
    suffix = destination.suffix.lstrip(".").lower() if destination else ""
    format = (format or suffix or "mermaid").lower()
    format = {"mmd": "mermaid", "gv": "dot"}.get(format, format)
    if format not in {"mermaid", "dot", "svg", "png"}:
        raise ValueError("format must be mermaid, dot, svg, or png")
    if destination is not None and destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass overwrite=True to replace it")
    graph = trace(model)
    result: str | bytes = _graph_text(graph, "mermaid" if format == "mermaid" else "dot")
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
