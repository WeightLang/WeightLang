"""Activation experiments, first-order attribution and intervention-validated circuits."""
from dataclasses import dataclass, field
from copy import deepcopy
import math
import hashlib
import json
import struct

import torch


@dataclass
class Activation:
    name: str
    instruction: int
    occurrence: int
    shape: tuple
    before: torch.Tensor
    after: torch.Tensor
    site: dict = field(default_factory=dict)


@dataclass
class Intervention:
    instruction: int
    occurrence: int
    value: torch.Tensor


@dataclass
class Attribution:
    site: dict
    occurrence: int
    value: torch.Tensor
    gradient: torch.Tensor
    contribution: torch.Tensor


@dataclass
class AttributionReport:
    output: torch.Tensor
    input_gradient: torch.Tensor
    events: list[Attribution]
    context: dict = field(default_factory=dict)


@dataclass
class Circuit:
    output: torch.Tensor
    nodes: list[Activation]
    edges: list[dict]
    changed_paths: list[int]
    work: int
    context: dict = field(default_factory=dict)

    def to_dict(self):
        """Return JSON-compatible data with tensor geometry retained."""
        def activation(event):
            return {**vars(event), "shape": list(event.shape),
                    "before": event.before.cpu().tolist(), "after": event.after.cpu().tolist()}
        return {"output": self.output.cpu().tolist(), "nodes": [activation(e) for e in self.nodes],
                "edges": self.edges, "changed_paths": self.changed_paths, "work": self.work,
                "context": self.context}

    def mermaid(self):
        """Return a renderable graph of measured total effects."""
        lines = ["flowchart LR", 'output["Output target"]']
        for i, event in enumerate(self.nodes):
            label = f'{event.site.get("scope", "")}/{event.name} #{event.occurrence}'
            label = "".join(c if c.isalnum() or c in "/._- #" else " " for c in label)
            lines.append(f'n{i}["{label}"]')
        for edge in self.edges:
            target = "output" if edge["target"] is None else f'n{edge["target"]}'
            effect = "path removed" if edge["effect"] is None else f'{edge["effect"]:.5g}'
            lines.append(f'n{edge["source"]} -->|"{effect}"| {target}')
        return "\n".join(lines)


class Research:
    def __init__(self, *, replacements=None, interventions=(), max_events=10000,
                 max_bytes=64 * 1024 * 1024, differentiable=False):
        if type(max_events) is not int or max_events < 0 or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("capture budgets must be nonnegative integers")
        self.replacements = dict(replacements or {})
        self.interventions = list(interventions)
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.differentiable = differentiable
        self.events = []
        self.values = []
        self.work = 0
        self._bytes = 0
        self._counts = {}
        self._targets = {}
        for patch in self.interventions:
            if (not isinstance(patch, Intervention) or type(patch.instruction) is not int
                    or type(patch.occurrence) is not int or patch.occurrence < 0):
                raise ValueError("invalid intervention target")
            key = (patch.instruction, patch.occurrence)
            if key in self._targets:
                raise ValueError("duplicate intervention target")
            self._targets[key] = patch.value

    def visit(self, site, value):
        size = value.numel() * value.element_size() * (4 if self.differentiable else 2)
        if len(self.events) >= self.max_events or size > self.max_bytes - self._bytes:
            raise ValueError("research capture budget exceeded")
        pc = site["instruction"]
        occurrence = self._counts.get(pc, 0)
        self._counts[pc] = occurrence + 1
        before = value.detach().clone()
        replacement = self._targets.get((pc, occurrence), self.replacements.get(site["name"]))
        if replacement is not None:
            value = replacement.detach().clone()
        if self.differentiable and not value.requires_grad:
            value = value.detach().requires_grad_(True)
        self.events.append(Activation(site["name"], pc, occurrence, tuple(value.shape),
                                      before, value.detach().clone(), deepcopy(site)))
        if self.differentiable:
            self.values.append(value)
        self._bytes += size
        return value


def _tensor_payload(tensor):
    cpu = tensor.detach().to(device="cpu").contiguous().clone()
    try:
        return cpu.view(torch.uint8).numpy().tobytes()
    except RuntimeError:
        return bytes(cpu.untyped_storage())


def _canonical_numbers(value):
    if type(value) is float:
        return {"$f64": struct.pack(">d", value).hex()}
    if isinstance(value, dict):
        return {key: _canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_numbers(item) for item in value]
    return value


def _context(model, value, seed):
    schema = deepcopy(model.definition["program"])
    for parameter in schema["parameters"]:
        parameter["data"] = []
    architecture = json.dumps(_canonical_numbers(schema), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    state = hashlib.sha256()
    for name, tensor in sorted(model._parameters_by_name().items()):
        encoded = name.encode()
        state.update(struct.pack("<Q", len(encoded)))
        state.update(encoded)
        state.update(struct.pack("<Q", tensor.ndim))
        for dim in tensor.shape:
            state.update(struct.pack("<Q", dim))
        state.update(str(tensor.dtype).encode())
        state.update(_tensor_payload(tensor))
    return {"model_fingerprint": hashlib.sha256(architecture).hexdigest(),
            "state_fingerprint": state.hexdigest(), "input": value.detach().cpu().tolist(),
            "seed": None if seed is None else seed.detach().cpu().tolist()}


def _validate_value(replacement, expected, value):
    if (not isinstance(replacement, torch.Tensor) or tuple(replacement.shape) != expected
            or replacement.dtype != value.dtype or replacement.device != value.device
            or not torch.isfinite(replacement).all()):
        raise ValueError("replacement must match activation shape, dtype and device and be finite")


def _execute(model, value, session, max_work, memory_bytes):
    from .program import ProgramModule, _limits
    if not isinstance(model, ProgramModule):
        raise TypeError("research requires a WGT program with observe sites")
    if tuple(value.shape) != model.input_shape:
        raise ValueError("research requires one unbatched input")
    sites = model._executor.observations
    names = {s["name"] for s in sites.values()}
    if session.replacements.keys() - names:
        raise ValueError("unknown observation replacement")
    for site in sites.values():
        if site["name"] in session.replacements:
            _validate_value(session.replacements[site["name"]],
                            model._executor.shapes[site["register"]], value)
    for patch in session.interventions:
        if patch.instruction not in sites:
            raise ValueError("unknown intervention instruction")
        _validate_value(patch.value, model._executor.shapes[sites[patch.instruction]["register"]], value)
    maximum, memory = _limits(model, max_work, memory_bytes)
    result = model._executor.run(value, model._parameters_by_name(), differentiable=session.differentiable,
                                 max_work=maximum, memory_bytes=memory, research=session)
    visited = {(e.instruction, e.occurrence) for e in session.events}
    if session._targets.keys() - visited:
        raise ValueError("intervention target was not executed")
    session.work = result.work
    return result.output


@torch.no_grad()
def inspect(model, value, *, replacements=None, interventions=(), max_events=10000,
            max_bytes=64 * 1024 * 1024, max_work=None, memory_bytes=None):
    """Capture a single snapshot execution, discarding all forward writes."""
    session = Research(replacements=replacements, interventions=interventions,
                       max_events=max_events, max_bytes=max_bytes)
    from .program import ProgramModule
    if not isinstance(model, ProgramModule):
        raise TypeError("inspect requires a WGT program with observe sites")
    with model._execution_lock:
        output = _execute(model, value, session, max_work, memory_bytes)
    return output, session.events


def attribute(model, value, seed, *, max_events=10000, max_bytes=64 * 1024 * 1024,
              max_work=None, memory_bytes=None):
    """Compute an output VJP and activation-times-gradient for each dynamic boundary."""
    from .program import ProgramModule
    if not isinstance(model, ProgramModule):
        raise TypeError("attribute requires a WGT program with observe sites")
    _validate_value(seed, model.output_shape, value)
    session = Research(max_events=max_events, max_bytes=max_bytes, differentiable=True)
    with model._execution_lock, torch.enable_grad():
        x = value.detach().clone().requires_grad_(True)
        output = _execute(model, x, session, max_work, memory_bytes)
        tensors = [x, *session.values]
        active = [tensor for tensor in tensors if tensor.requires_grad]
        gradients = {}
        if output.requires_grad:
            computed = torch.autograd.grad(output, active, grad_outputs=seed, allow_unused=True)
            gradients = {id(t): g for t, g in zip(active, computed)}
        values = [gradients.get(id(t)) for t in tensors]
        values = [torch.zeros_like(t) if g is None else g for t, g in zip(tensors, values)]
        if any(not torch.isfinite(g).all() for g in values):
            raise ValueError("nonfinite attribution gradient")
        events = []
        for event, gradient in zip(session.events, values[1:]):
            contribution = event.before * gradient
            if not torch.isfinite(contribution).all():
                raise ValueError("nonfinite attribution")
            events.append(Attribution(event.site, event.occurrence, event.before,
                                      gradient.detach().clone(), contribution.detach()))
        context = _context(model, value, seed)
    return AttributionReport(output.detach(), values[0].detach(), events, context)


@torch.no_grad()
def circuit(model, value, seed, *, max_nodes=32, max_work=None,
            max_bytes=64 * 1024 * 1024, memory_bytes=None, epsilon=1e-6):
    """Measure total causal influences by ablating one dynamic event per experiment.

    Output edges are signed changes in dot(output, seed).
    Internal edges report L2 changes in downstream pre-intervention activations.
    Missing downstream events have effect=None. Changed paths are recorded explicitly.
    """
    from .program import ProgramModule, _limits
    if not isinstance(model, ProgramModule):
        raise TypeError("circuit requires a WGT program with observe sites")
    if type(max_nodes) is not int or not 1 <= max_nodes <= 128:
        raise ValueError("circuit nodes must be 1..128")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid circuit threshold")
    _validate_value(seed, model.output_shape, value)
    maximum, memory = _limits(model, max_work, memory_bytes)
    baseline = Research(max_events=max_nodes, max_bytes=max_bytes)
    with model._execution_lock:
        output = _execute(model, value, baseline, maximum, memory)
        work = baseline.work
        edges, changed_paths = [], []
        identities = [(e.instruction, e.occurrence) for e in baseline.events]
        for source, event in enumerate(baseline.events):
            experiment = Research(max_bytes=max_bytes, interventions=[
                Intervention(event.instruction, event.occurrence, torch.zeros_like(event.before))])
            if work >= maximum:
                raise ValueError("circuit work budget exceeded")
            patched = _execute(model, value, experiment, maximum - work, memory)
            work += experiment.work
            measured = {(e.instruction, e.occurrence): e for e in experiment.events}
            if identities != [(e.instruction, e.occurrence) for e in experiment.events]:
                changed_paths.append(source)
            effect = ((output - patched) * seed).sum().item()
            if not math.isfinite(effect):
                raise ValueError("nonfinite circuit effect")
            if abs(effect) > epsilon:
                edges.append({"source": source, "target": None, "effect": effect})
            for target in range(source + 1, len(baseline.events)):
                downstream = baseline.events[target]
                after = measured.get((downstream.instruction, downstream.occurrence))
                effect = None
                if after is not None:
                    effect = (downstream.before.double() - after.before.double()).norm().item()
                    if not math.isfinite(effect):
                        raise ValueError("nonfinite circuit effect")
                if effect is None or effect > epsilon:
                    edges.append({"source": source, "target": target, "effect": effect})
        context = _context(model, value, seed)
    return Circuit(output, baseline.events, edges, changed_paths, work, context)
