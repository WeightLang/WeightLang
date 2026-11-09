"""Immutable weight snapshots and explicit, complete construction plans."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import torch
from safetensors.torch import load as decode_tensors, save as encode_tensors

from .initialization import checked_binding, checked_bindings, materialize, parameter_specs


FORMAT = "weightlang.weights.v1"
_PAYLOAD = "weights.safetensors"
_MAX_BYTES = 64 * 1024**2
_MAX_MANIFEST = 1024**2


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate weight manifest field")
    return result


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"weight file exceeds {limit} bytes: {path.name}")
    return content


def _table(tensors: Mapping[str, torch.Tensor]) -> list[dict]:
    from .runtime import _shape
    result, total = [], 0
    if any(not isinstance(name, str) or not name for name in tensors):
        raise ValueError("weight names must be nonempty strings")
    for name, value in sorted(tensors.items()):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.dtype != torch.float32:
            raise ValueError(f"weight {name} must be a dense F32 tensor; convert explicitly")
        _shape(list(value.shape), f"weight {name}")
        if any(size > 16384 for size in value.shape):
            raise ValueError(f"weight {name} axis exceeds 16384")
        total += value.numel() * 4
        if total > _MAX_BYTES:
            raise ValueError("weight snapshot exceeds 64 MiB")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"weight {name} contains nonfinite values")
        result.append({"name": name, "shape": list(value.shape), "dtype": "F32"})
    return result


@dataclass(frozen=True, init=False)
class WeightSet:
    """A value snapshot. Returned tensors are copies and cannot mutate its revision.

    Only parameters are stored: runtime state, optimizers and RNG state belong
    to separate checkpoint contracts. The revision hashes the exact payload.
    """

    _payload: bytes = field(repr=False)
    _metadata: str = field(repr=False)

    def __init__(self, tensors: Mapping[str, torch.Tensor], *, provenance: dict | None = None):
        table = _table(tensors)
        values = {name: value.detach().to(device="cpu").contiguous().clone() for name, value in tensors.items()}
        payload = encode_tensors(values)
        if len(payload) > _MAX_BYTES:
            raise ValueError("weight snapshot exceeds 64 MiB")
        metadata = {"format": FORMAT, "revision": hashlib.sha256(payload).hexdigest(), "payload": _PAYLOAD,
                    "tensors": table, "provenance": provenance if provenance is not None else {}}
        if not isinstance(metadata["provenance"], dict):
            raise TypeError("weight provenance must be a JSON object")
        encoded = _json(metadata)
        if len(encoded.encode()) > _MAX_MANIFEST:
            raise ValueError("weight manifest exceeds 1 MiB")
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_metadata", encoded)

    def __repr__(self) -> str:
        return f"WeightSet(revision={self.revision!r}, parameters={len(self.names)})"

    @property
    def revision(self) -> str:
        return self.manifest["revision"]

    @property
    def manifest(self) -> dict:
        return json.loads(self._metadata)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.manifest["tensors"])

    def __getitem__(self, name: str) -> torch.Tensor:
        return decode_tensors(self._payload)[name].clone()

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in decode_tensors(self._payload).items()}

    def ref(self, name: str) -> WeightRef:
        if name not in self.names:
            raise KeyError(f"unknown weight {name!r}")
        return WeightRef(self, name)

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> WeightSet:
        """Capture logical WGT parameters; state buffers are deliberately excluded."""
        model = getattr(model, "_orig_mod", model)
        names = getattr(model, "parameter_names", None)
        if not isinstance(names, dict) or not isinstance(getattr(model, "definition", None), dict):
            raise TypeError("from_model requires a WeightLang module")
        lock = getattr(model, "_operation_lock", getattr(model, "_execution_lock", None))
        with lock if lock is not None else nullcontext():
            values = checked_bindings(model.definition, {name: model.get_parameter(key) for name, key in names.items()})
            return cls(values, provenance={"operation": "capture", "model": model.definition.get("name")})

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, revision: str | None = None) -> WeightSet:
        """Read a snapshot and optionally require a specific, full SHA-256 pin."""
        root = Path(path)
        metadata = json.loads(_read_bounded(root / "manifest.json", _MAX_MANIFEST), object_pairs_hook=_object)
        if (not isinstance(metadata, dict)
                or set(metadata) != {"format", "revision", "payload", "tensors", "provenance"}
                or metadata["format"] != FORMAT or metadata["payload"] != _PAYLOAD
                or not isinstance(metadata["revision"], str)
                or re.fullmatch(r"[0-9a-f]{64}", metadata["revision"]) is None
                or not isinstance(metadata["provenance"], dict)):
            raise ValueError("invalid weight snapshot manifest")
        if revision is not None and revision != metadata["revision"]:
            raise ValueError("weight revision pin mismatch")
        payload = _read_bounded(root / _PAYLOAD, _MAX_BYTES)
        if hashlib.sha256(payload).hexdigest() != metadata["revision"]:
            raise ValueError("weight payload SHA-256 mismatch")
        if not isinstance(metadata["tensors"], list):
            raise ValueError("invalid weight tensor manifest")
        from .runtime import _shape
        for item in metadata["tensors"]:
            if (not isinstance(item, dict) or set(item) != {"name", "shape", "dtype"}
                    or not isinstance(item["name"], str) or item["dtype"] != "F32"
                    or not isinstance(item["shape"], list)):
                raise ValueError("invalid weight tensor manifest fields")
            _shape(item["shape"], "manifest tensor shape")
        if _table(decode_tensors(payload)) != sorted(metadata["tensors"], key=lambda item: item["name"]):
            raise ValueError("weight tensor manifest does not match payload")
        result = object.__new__(cls)
        object.__setattr__(result, "_payload", payload)
        object.__setattr__(result, "_metadata", _json(metadata))
        return result

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Publish a complete new snapshot. Existing destinations are never replaced.

        The manifest is the atomic publication marker: readers reject a directory
        until its complete payload and manifest are present.
        """
        root = Path(path)
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
            stage = Path(temporary)
            for name, content in ((_PAYLOAD, self._payload), ("manifest.json", (self._metadata + "\n").encode())):
                with (stage / name).open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            root.mkdir()  # Atomic reservation; rejects files, empty directories and prior snapshots.
            try:
                os.link(stage / _PAYLOAD, root / _PAYLOAD)
                os.link(stage / "manifest.json", root / "manifest.json")
            except BaseException:
                shutil.rmtree(root)
                raise
        return root


@dataclass(frozen=True)
class WeightRef:
    """One logical parameter in one fixed snapshot; it never follows file changes."""

    snapshot: WeightSet
    name: str

    @property
    def revision(self) -> str:
        return self.snapshot.revision

    def tensor(self) -> torch.Tensor:
        return self.snapshot[self.name]

    def source(self) -> dict:
        return {"revision": self.revision, "name": self.name}


WeightValue = torch.Tensor | WeightRef | Callable[[tuple[int, ...]], torch.Tensor]


class Plan:
    """Explicit assignments for a complete target schema, without model execution.

    ``base`` explicitly inherits a fully compatible snapshot. Architecture
    migration uses no base and names every destination through copy/set/reuse.
    Initializer callbacks receive the target shape and run only during build.
    """

    def __init__(self, definition: dict, *, base: WeightSet | None = None):
        self.definition = deepcopy(definition)
        self._specs = {spec["name"]: spec for spec in parameter_specs(definition)}
        self._values: dict[str, WeightValue] = {}
        self._origins: dict[str, dict] = {}
        self._replaced: set[str] = set()
        self._base = base
        if base is not None:
            checked_bindings(definition, base.tensors())
            for name in self._specs:
                self._values[name] = base.ref(name)
                self._origins[name] = {"operation": "copy", **base.ref(name).source()}

    def _prepare(self, name: str, value: WeightValue, sources: Iterable[WeightRef] = ()) -> tuple[WeightValue, dict]:
        if name not in self._specs:
            raise KeyError(f"unknown target parameter {name!r}")
        if isinstance(value, WeightRef):
            checked_binding(value.tensor(), self._specs[name], f32=True)
            return value, {"operation": "copy", **value.source()}
        if callable(value):
            origin = {"operation": "initialize", "function": getattr(value, "__qualname__", type(value).__name__)}
            prepared = value
        else:
            prepared = checked_binding(value, self._specs[name], f32=True)
            origin = {"operation": "explicit_tensor"}
        recorded = [reference.source() for reference in sources]
        if recorded:
            origin["sources"] = recorded
        return prepared, origin

    def set(self, name: str, value: WeightValue, *, sources: Iterable[WeightRef] = ()) -> Plan:
        if name in self._values:
            raise ValueError(f"parameter {name} is already assigned; use replace explicitly")
        prepared, origin = self._prepare(name, value, sources)
        self._values[name], self._origins[name] = prepared, origin
        return self

    def set_many(self, values: Mapping[str, WeightValue]) -> Plan:
        """Validate the whole assignment batch before adding any destination."""
        if set(values) & self._values.keys():
            raise ValueError("duplicate parameter assignment in batch")
        prepared = {name: self._prepare(name, value) for name, value in values.items()}
        for name, (value, origin) in prepared.items():
            self._values[name], self._origins[name] = value, origin
        return self

    def copy(self, name: str, source: WeightRef) -> Plan:
        if not isinstance(source, WeightRef):
            raise TypeError("copy requires a fixed WeightRef")
        return self.set(name, source)

    def reuse(self, source: WeightSet, names: Iterable[str]) -> Plan:
        names = [names] if isinstance(names, str) else list(names)
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter in reuse selection")
        return self.set_many({name: source.ref(name) for name in names})

    def replace(self, name: str, value: WeightValue, *, sources: Iterable[WeightRef] = ()) -> Plan:
        if name not in self._values:
            raise ValueError(f"replace requires an existing assignment: {name}")
        if name in self._replaced:
            raise ValueError(f"parameter {name} was already replaced")
        prepared, origin = self._prepare(name, value, sources)
        self._values[name], self._origins[name] = prepared, origin
        self._replaced.add(name)
        return self

    def initialize(self, names: Iterable[str] | None = None) -> Plan:
        """Explicitly select declared recipes; omitted names selects all missing slots."""
        names = (list(self._specs.keys() - self._values.keys()) if names is None
                 else [names] if isinstance(names, str) else list(names))
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter in initialization selection")
        values = {}
        for name in names:
            spec = self._specs[name]
            if (spec.get("initializer") or {}).get("op") == "unbound" or (not spec.get("initializer") and not spec.get("data")):
                raise ValueError(f"parameter {name} has no declared initializer")
            values[name] = lambda shape, spec=deepcopy(spec): materialize(spec)
        self.set_many(values)
        for name in names:
            self._origins[name] = {"operation": "declared_initializer"}
        return self

    def preview(self) -> dict:
        missing = sorted(self._specs.keys() - self._values.keys())
        return {"complete": not missing, "missing": missing,
                "base_revision": self._base.revision if self._base is not None else None,
                "assignments": [{"name": name, "shape": list(self._specs[name]["shape"]),
                                 "deferred": callable(self._values[name]), **deepcopy(self._origins[name])}
                                for name in sorted(self._values)]}

    def build(self) -> WeightSet:
        report = self.preview()
        if report["missing"]:
            raise ValueError(f"unassigned target parameters: {', '.join(report['missing'])}")
        values = {}
        for name, value in self._values.items():
            if isinstance(value, WeightRef):
                value = value.tensor()
            elif callable(value):
                value = value(tuple(self._specs[name]["shape"]))
            values[name] = checked_binding(value, self._specs[name], f32=True)
        return WeightSet(values, provenance={"operation": "construct", "base_revision": report["base_revision"],
                                            "assignments": deepcopy(self._origins)})


def restore(definition: dict, weights: WeightSet, *, device: str | torch.device = "auto",
            dtype: torch.dtype | None = None) -> torch.nn.Module:
    """Construct directly from complete bindings, without running parameter recipes.

    A new instance initializes its declared runtime state separately. Optimizer
    history is never read, synthesized or copied from a weight snapshot.
    """
    from .api import select_device
    from .runtime import build_module
    if not isinstance(weights, WeightSet):
        raise TypeError("restore requires a WeightSet")
    return build_module(definition, bindings=weights.tensors()).to(device=select_device(device), dtype=dtype)
