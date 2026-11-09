"""Typed WGT operations with shared PyTorch parameters and persistent buffers."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from collections.abc import Mapping
import math
from threading import RLock
from typing import Any

import torch
from torch import nn

from .program import ProgramExecutor, validate_parameter
from .runtime import _MAX_ELEMENTS, _attribute, _positive, _shape, _tensor


def _limit(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class OperationsModule(nn.Module):
    """A typed-operation model; calls share one parameter and state store.

    Use ``call("advance", tensor)`` or ``advance(tensor)`` for a WGT operation.
    Names already used by ``nn.Module`` remain accessible through ``call``.
    Inputs are positional, finite vectors matching the declared widths and the
    model's device/dtype. Each invocation commits all checked writes together;
    execution, validation, or work-budget failures leave storage unchanged.

    Typed operations follow the native WGT contract: no automatic or cross-state
    differentiation. Parameters have ``requires_grad=False`` and inputs asking
    for gradients are rejected. ``state_dict`` contains both parameters and
    state buffers; ``reset()`` exists only when the WGT model declares it.
    ``max_work`` and ``memory_bytes`` bound each call's native IR execution.
    """

    def __init__(
        self,
        definition: dict[str, Any],
        *,
        max_work: int = 1_000_000,
        memory_bytes: int = 256 * 1024**2,
    ) -> None:
        super().__init__()
        if definition.get("format") != "weightlang.torch.v1" or definition.get("kind") != "operations":
            raise ValueError("expected a typed-operation WeightLang compiler export")
        if definition.get("interface") is not None:
            raise ValueError("typed operations declare their own signatures, not a model interface")
        self.definition = deepcopy(definition)
        self.max_work = _limit(max_work, "max_work")
        self.memory_bytes = _limit(memory_bytes, "memory_bytes")
        self.parameter_names: dict[str, str] = {}
        self.state_names: dict[str, str] = {}
        self._specs: dict[str, dict] = {}
        self._entries: dict[str, tuple[dict, ProgramExecutor]] = {}
        self._candidate_updatable: dict[str, bool] = {}
        self._operation_lock = RLock()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        try:
            self._initialize(self.definition["operations"])
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(f"malformed typed-operation compiler export: {error}") from error

    def _initialize(self, plan: dict) -> None:
        entries = plan["entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("typed model requires at least one operation")
        names = [entry["schema"]["name"] for entry in entries]
        if (any(not isinstance(name, str) or not name.isidentifier() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("operation names must be unique identifiers")
        # Reserve operation attributes before registering storage, so a weight
        # with the same WGT name cannot hide an otherwise callable operation.
        self._operation_names = frozenset(names)
        count = 0
        for key, mapping in (("parameters", self.parameter_names), ("states", self.state_names)):
            for spec in plan[key]:
                name = spec["name"]
                if not isinstance(name, str) or not name.isidentifier() or name in self._specs:
                    raise ValueError(f"duplicate or invalid operation storage identity: {name!r}")
                shape = _shape(spec["shape"], f"storage {name}")
                count += math.prod(shape)
                if count > _MAX_ELEMENTS:
                    raise ValueError("operation storage element limit exceeded")
                if type(spec.get("frozen", False)) is not bool or type(spec.get("probability", False)) is not bool:
                    raise ValueError(f"storage {name} frozen/probability metadata must be boolean")
                if key == "states" and (len(shape) != 1 or shape[0] > 16384 or spec.get("frozen")
                                         or spec.get("probability") or spec.get("learn") is not None):
                    raise ValueError(f"state {name} must be a mutable, plain f32 vector")
                value = _tensor(spec["data"], shape, f"storage {name}")
                validate_parameter(value, spec)
                attribute = _attribute(self, name, "state" if key == "states" else "parameter")
                if key == "states":
                    self.register_buffer(attribute, value)
                else:
                    self.register_parameter(attribute, nn.Parameter(value, requires_grad=False))
                mapping[name] = attribute
                self._specs[name] = spec
        outer = self.definition.get("parameters", [])
        expected = [{key: spec[key] for key in ("name", "shape", "frozen", "learn") if key in spec}
                    for spec in plan["parameters"]]
        normalized = lambda specs: {
            spec["name"]: (spec["shape"], spec.get("frozen", False), spec.get("learn")) for spec in specs
        }
        if len(outer) != len(expected) or normalized(outer) != normalized(expected):
            raise ValueError("operation parameter schemas do not match initialized storage")
        for schema in outer:
            name = schema["name"]
            spec = self._specs[name]
            eligible = not spec.get("frozen", False) and not spec.get("probability", False)
            # Old compiler exports lack the public flag, but their initialized
            # storage describes both exclusions. Never infer from frozen alone.
            updatable = schema.get("candidate_updatable", eligible)
            if type(updatable) is not bool or (updatable and not eligible):
                raise ValueError(f"parameter {name} has inconsistent candidate_updatable metadata")
            self._candidate_updatable[name] = updatable
        for entry in entries:
            schema, program = entry["schema"], entry["program"]
            name = schema["name"]
            ports = schema["inputs"]
            port_names = [port["name"] for port in ports]
            if (any(not isinstance(port, str) or not port.isidentifier() for port in port_names)
                    or len(set(port_names)) != len(port_names)):
                raise ValueError(f"operation {name} has invalid or duplicate input names")
            width = sum(_positive(port["width"], f"{name} input width", 16384) for port in ports)
            if width > 16384:
                raise ValueError(f"operation {name} aggregate input exceeds 16384 values")
            output = schema["output"]
            if output is not None:
                _positive(output, f"{name} output width", 16384)
            if program.get("loss") is not None:
                raise ValueError("typed operations do not support loss or cross-state differentiation")
            registers = {register["name"]: register["shape"] for register in program["registers"]}
            if registers.get(program["input"]) != [max(width, 1)]:
                raise ValueError(f"operation {name} packed input does not match its signature")
            for instruction in program["instructions"]:
                if instruction["op"] == "return" and registers.get(instruction["value"]) != [output or 1]:
                    raise ValueError(f"operation {name} return does not match its signature")
            reads, writes = set(), set()
            for instruction in program["instructions"]:
                if instruction["op"] == "read" and instruction["parameter"] in self.state_names:
                    reads.add(instruction["parameter"])
                if instruction["op"] == "write":
                    target = instruction["parameter"]
                    if target not in self._specs:
                        raise ValueError(f"operation {name} writes unknown storage {target!r}")
                    spec = self._specs[target]
                    if spec.get("frozen") or spec.get("probability"):
                        raise ValueError(f"operation {name} cannot update frozen or probability storage {target!r}")
                    writes.add(target)
            if name == "forward" and writes:
                raise ValueError("forward must not update state or parameters; use a named update operation")
            actual_effects = {
                "reads_state": sorted(reads),
                "writes_state": sorted(writes & self.state_names.keys()),
                "writes_parameters": sorted(writes & self.parameter_names.keys()),
            }
            if schema["effects"] != actual_effects:
                raise ValueError(f"operation {name} effects do not match its instructions")
            for spec in program["parameters"]:
                stored = self._specs.get(spec["name"])
                if stored is None or any(spec.get(key, False if key in ("frozen", "probability") else None)
                                         != stored.get(key, False if key in ("frozen", "probability") else None)
                                         for key in ("shape", "frozen", "probability", "learn")):
                    raise ValueError(f"operation {name} storage schema mismatch")
            executor = ProgramExecutor(program, parameter_specs=list(self._specs.values()), transactional=True)
            self._entries[name] = (schema, executor)

    @property
    def operation_schemas(self) -> dict[str, dict]:
        """A copy of each WGT signature and its statically computed effects."""
        return {name: deepcopy(schema) for name, (schema, _) in self._entries.items()}

    @property
    def parameter_schema(self) -> dict[str, dict]:
        """Storage domains and external-update eligibility, keyed by WGT name."""
        return {
            name: {"name": name, "shape": list(self._specs[name]["shape"]),
                   "frozen": bool(self._specs[name].get("frozen", False)),
                   "probability": bool(self._specs[name].get("probability", False)),
                   "learn": deepcopy(self._specs[name].get("learn")),
                   "candidate_updatable": self._candidate_updatable[name]}
            for name in self.parameter_names
        }

    def parameter_values(self) -> dict[str, torch.Tensor]:
        """Return detached parameter snapshots under their original WGT names."""
        with self._operation_lock:
            return {name: getattr(self, attribute).detach().clone()
                    for name, attribute in self.parameter_names.items()}

    def genome_schema(self):
        """Describe fixed-topology genetic search without creating a population."""
        from .evolution import GenomeSchema
        return GenomeSchema.from_model(self)

    def update_parameters(self, candidates: Mapping[str, torch.Tensor]) -> None:
        """Atomically apply a nonempty named subset of parameter candidates.

        Each tensor must already match the model's shape, dtype and device.
        Integer and bool domains require hard discrete values; this method does
        not round, project, copy across devices, or update persistent states.
        Validation and publication share the operation lock with ``call``.
        """
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError("parameter candidates must be a nonempty name-to-tensor mapping")
        with self._operation_lock, torch.no_grad():
            storage = self._storage()
            if any(value.requires_grad for value in storage.values()):
                raise ValueError("typed operations do not support autograd; keep their parameters frozen to gradients")
            prepared: dict[str, torch.Tensor] = {}
            for name, value in candidates.items():
                if name in prepared:
                    raise ValueError(f"duplicate candidate parameter {name!r}")
                if name not in self.parameter_names:
                    raise ValueError(f"unknown candidate parameter {name!r}; states are not parameter candidates")
                if not self._candidate_updatable[name]:
                    raise ValueError(f"candidate cannot update frozen, probability or ineligible parameter {name!r}")
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"candidate parameter {name} must be a torch.Tensor")
                if value.requires_grad:
                    raise ValueError("parameter candidates must not require autograd")
                if value.device != storage[name].device or value.dtype != storage[name].dtype:
                    raise ValueError(f"candidate parameter {name} must match model dtype/device")
                candidate = value.detach().clone()
                validate_parameter(candidate, self._specs[name], discrete=True)
                prepared[name] = candidate
            for name, candidate in prepared.items():
                storage[name].copy_(candidate)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self.__dict__.get("_operation_names", ()) and not name.startswith("_"):
                return partial(self.call, name)
            raise

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.pop("_operation_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._operation_lock = RLock()

    def _storage(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, attribute)
                for name, attribute in (self.parameter_names | self.state_names).items()}

    def call(
        self,
        name: str,
        *inputs: torch.Tensor,
        max_work: int | None = None,
    ) -> torch.Tensor | None:
        """Invoke one operation atomically; optionally override its work budget."""
        if name not in self._entries:
            raise ValueError(f"unknown model operation {name!r}; available: {', '.join(self._entries)}")
        schema, executor = self._entries[name]
        if len(inputs) != len(schema["inputs"]):
            raise ValueError(f"operation {name} expects {len(schema['inputs'])} input vectors, got {len(inputs)}")
        with self._operation_lock:
            storage = self._storage()
            reference = next(iter(storage.values()), self._device_anchor)
            for value, port in zip(inputs, schema["inputs"]):
                if not isinstance(value, torch.Tensor) or value.shape != (port["width"],):
                    raise ValueError(f"operation {name} input {port['name']} requires shape ({port['width']},)")
                if not value.is_floating_point() or value.dtype != reference.dtype or value.device != reference.device:
                    raise ValueError(f"operation {name} input {port['name']} must match model dtype/device")
                if value.requires_grad:
                    raise ValueError("typed operations do not support autograd or cross-state differentiation")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"operation {name} input {port['name']} must be finite")
            if any(value.requires_grad for value in storage.values()):
                raise ValueError("typed operations do not support autograd; keep their parameters frozen to gradients")
            budget = _limit(self.max_work if max_work is None else max_work, "max_work")
            memory = _limit(self.memory_bytes, "memory_bytes")
            with torch.no_grad():
                packed = torch.cat(inputs) if inputs else reference.new_zeros(1)
                result = executor.run(packed, storage, differentiable=False, max_work=budget, memory_bytes=memory)
                if result.output.shape != (schema["output"] or 1,) or not bool(torch.isfinite(result.output).all()):
                    raise ValueError(f"operation {name} returned an invalid or nonfinite vector")
                # Complete validation before the first in-place update. Reads in
                # the executor see its candidates, while live storage is untouched.
                for target in result.writes:
                    validate_parameter(result.parameters[target], self._specs[target], discrete=True)
                    if (result.parameters[target].device != storage[target].device
                            or result.parameters[target].dtype != storage[target].dtype):
                        raise ValueError(f"operation {name} changed storage dtype/device")
                output = result.output.detach().clone() if schema["output"] is not None else None
                for target in result.writes:
                    storage[target].copy_(result.parameters[target])
                return output

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | None:
        """Invoke the declared typed forward without modifying model storage."""
        if "forward" not in self._entries:
            raise ValueError("this model has no forward operation; use model.call(name, *inputs)")
        return self.call("forward", *inputs)

    def extra_repr(self) -> str:
        return f"operations={list(self._entries)}, states={list(self.state_names)}, autograd=False"
