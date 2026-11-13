"""Explicit, trusted PyTorch-target models compiled by the shared Rust frontend.

These definitions contain generated Python. The native model loader never runs
this target. Applications bind declared host dependencies explicitly.
"""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .compiler import compile as compile_model

FORMAT = "weightlang.torch_module.v1"
_SNAPSHOTS = ContextVar("wgt_torch_module_snapshots", default=None)


def _build(definition, namespace):
    if (
        definition.get("format") != "weightlang.torch.v1"
        or definition.get("kind") != "torch_module"
    ):
        raise ValueError("expected a compiled PyTorch-target model definition")
    plan = definition["torch_module"]
    missing = set(plan["externs"]) - namespace.keys()
    if missing:
        raise ValueError(f"missing declared host dependencies: {sorted(missing)}")
    scope = {name: namespace[name] for name in plan["externs"]}
    scope.update(Module=nn.Module, __name__=plan["host"])
    # Only compiler-generated Python is evaluated. WGT parsing and expression
    # validation belong to the Rust frontend, never a parallel Python parser.
    exec(compile(plan["code"], "<compiled WGT torch model>", "exec"), scope)
    result = {}
    for name in [*plan["models"], *plan["functions"]]:
        cls = scope[name]
        if name in plan["models"] and (
            not isinstance(cls, type) or not issubclass(cls, nn.Module)
        ):
            raise ValueError(f"{name} must define a torch.nn.Module")
        cls._wgt_definition = deepcopy(definition)
        cls._wgt_source = definition["source"]
        cls._wgt_dependencies = {
            key: value
            for key, value in namespace.items()
            if key in plan["externs"] and hasattr(value, "_wgt_definition")
        }
        result[name] = cls
    return result


def classes_from_ir(definition, namespace, *, trusted=False):
    """Bind already compiled architecture to an explicit host namespace."""
    if not trusted:
        raise ValueError("PyTorch-target definitions require trusted=True")
    return _build(definition, namespace)


def bind_models(path, namespace, *, compiler=None):
    """Compile a local project model and bind its declared host dependencies."""
    snapshots = _SNAPSHOTS.get()
    definition = (
        snapshots.get(namespace.get("__name__")) if snapshots is not None else None
    )
    return _build(
        definition
        if definition is not None
        else compile_model(path, compiler=compiler),
        namespace,
    )


def _encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": [type(value).__module__, type(value).__qualname__],
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("model configuration keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported model configuration value: {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$tuple"}:
        return tuple(_decode(item) for item in value["$tuple"])
    if set(value) == {"$dataclass", "fields"}:
        module, qualified = value["$dataclass"]
        cls = importlib.import_module(module)
        for part in qualified.split("."):
            cls = getattr(cls, part)
        if not is_dataclass(cls):
            raise ValueError("saved configuration type must be a dataclass")
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    return {key: _decode(item) for key, item in value.items()}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_module(model, path, *, kwargs, metadata=None, origin="edited", training=None):
    """Save complete source/IR snapshots and tensor values to a new directory.

    Parameter aliases are copied on disk, then tied again by the model's source.
    The artifact records frozen flags, constructor data and host file digests.
    Optimizer state belongs to the application's training checkpoint.
    """
    cls = type(model)
    if not hasattr(cls, "_wgt_definition"):
        raise TypeError("expected a compiled PyTorch-target model")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    definitions = {}
    hosts = {}
    bindings = {}
    visiting = set()

    def collect(candidate):
        definition = candidate._wgt_definition
        host = definition["torch_module"]["host"]
        if host in definitions:
            if definitions[host] != definition:
                raise ValueError(f"conflicting source snapshots for {host}")
            return
        if host in visiting:
            raise ValueError(f"cyclic model dependency: {host}")
        visiting.add(host)
        bindings[host] = {}
        for name, dependency in candidate._wgt_dependencies.items():
            collect(dependency)
            bindings[host][name] = [
                dependency._wgt_definition["torch_module"]["host"],
                dependency.__name__,
            ]
        visiting.remove(host)
        definitions[host] = definition
        module = importlib.import_module(host)
        hosts[host] = _sha(module.__file__)
        for name in definition["torch_module"]["externs"]:
            value = getattr(module, name, None)
            owner = inspect.getmodule(value)
            filename = getattr(owner, "__file__", None)
            if (
                filename
                and str(filename).endswith(".py")
                and owner.__name__.split(".")[0] == host.split(".")[0]
            ):
                hosts[owner.__name__] = _sha(filename)

    collect(cls)
    # Fully encode and validate metadata before reserving any destination.
    values = dict(model.state_dict())
    constants = {
        name: value
        for name, value in model.named_buffers(remove_duplicate=False)
        if name not in values
    }
    for name, value in constants.items():
        key = "__wgt_buffer__." + name
        if key in values:
            raise ValueError(f"reserved artifact tensor name: {key}")
        values[key] = value
    manifest = {
        "nonpersistent_buffers": list(constants),
        "format": FORMAT,
        "model": cls.__name__,
        "host": cls._wgt_definition["torch_module"]["host"],
        "kwargs": _encode(kwargs),
        "definitions": definitions,
        "bindings": bindings,
        "host_sha256": hosts,
        "frozen": {
            name: not parameter.requires_grad
            for name, parameter in model.named_parameters()
        },
        "application_metadata": metadata or {},
        "origin": origin,
        "training": training,
        "weights": {"file": "weights.safetensors"},
        "training_mode": model.training,
    }
    json.dumps(manifest, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    reserved = False
    try:
        save_file(
            {
                name: value.detach().cpu().contiguous().clone()
                for name, value in values.items()
            },
            str(staging / "weights.safetensors"),
        )
        manifest["weights"]["sha256"] = _sha(staging / "weights.safetensors")
        (staging / "source.wgt").write_text(cls._wgt_source, encoding="utf-8")
        manifest["source_sha256"] = _sha(staging / "source.wgt")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # mkdir provides exclusive ownership, including concurrent publishers.
        path.mkdir()
        reserved = True
        for name in ("weights.safetensors", "source.wgt", "manifest.json"):
            os.replace(staging / name, path / name)
    except BaseException:
        if reserved:
            shutil.rmtree(path)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return ModuleArtifact(path, manifest, model)


class ModuleArtifact:
    def __init__(self, path, manifest, module):
        self.path, self.manifest, self.module = Path(path), manifest, module


def load_module(path, *, trusted=False):
    """Load a complete target-specific artifact with explicit executable trust."""
    if not trusted:
        raise ValueError("PyTorch-target artifacts require trusted=True")
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(
            "unsupported artifact format; legacy torch-dynamic artifacts need explicit migration"
        )
    if manifest["weights"]["file"] != "weights.safetensors":
        raise ValueError("invalid artifact weight filename")
    if (
        _sha(path / "weights.safetensors") != manifest["weights"]["sha256"]
        or _sha(path / "source.wgt") != manifest["source_sha256"]
    ):
        raise ValueError("artifact source or weight digest mismatch")
    token = _SNAPSHOTS.set(manifest["definitions"])
    try:
        for name, digest in manifest["host_sha256"].items():
            host = importlib.import_module(name)
            if _sha(host.__file__) != digest:
                raise ValueError(f"host dependency digest mismatch: {name}")
        exports = {}
        for host, definition in manifest["definitions"].items():
            module = importlib.import_module(host)
            namespace = dict(vars(module))
            for name, (dependency_host, dependency_name) in manifest["bindings"][
                host
            ].items():
                namespace[name] = exports[dependency_host][dependency_name]
            exports[host] = _build(definition, namespace)
        cls = exports[manifest["host"]][manifest["model"]]
        values = load_file(str(path / "weights.safetensors"))
        buffers = {
            name: values.pop("__wgt_buffer__." + name)
            for name in manifest["nonpersistent_buffers"]
        }
        model = restore_parameters(
            cls,
            values,
            buffers=buffers,
            kwargs=_decode(manifest["kwargs"]),
            frozen=manifest["frozen"],
            training=manifest["training_mode"],
        )
    finally:
        _SNAPSHOTS.reset(token)
    return ModuleArtifact(path, manifest, model)


def restore_parameters(
    cls, values, *, kwargs, frozen=None, training=False, buffers=None
):
    """Bind a complete saved parameter/buffer set without drawing initial weights.

    Constructors must support PyTorch's meta device. Explicitly tied parameter
    identities are retained, including duplicate state_dict entries.
    """
    if not hasattr(cls, "_wgt_definition"):
        raise TypeError("expected a compiled PyTorch-target model class")
    with torch.device("meta"):
        model = cls(**kwargs)
    expected_buffers = {
        name: value
        for name, value in model.named_buffers(remove_duplicate=False)
        if name not in model.state_dict()
    }
    buffers = {} if buffers is None else buffers
    if buffers.keys() != expected_buffers.keys():
        raise ValueError("saved model requires complete nonpersistent buffer bindings")
    for name, value in buffers.items():
        expected = expected_buffers[name]
        floating_conversion = value.is_floating_point() and expected.is_floating_point()
        if value.shape != expected.shape or (
            value.dtype != expected.dtype and not floating_conversion
        ):
            raise ValueError(f"nonpersistent buffer contract mismatch: {name}")
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, value)
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        for name in names[1:]:
            if (
                names[0] in values
                and name in values
                and not torch.equal(values[names[0]], values[name])
            ):
                raise ValueError(f"inconsistent values for tied parameters: {names}")
    if any(
        not bool(torch.isfinite(value).all())
        for value in [*values.values(), *buffers.values()]
        if value.is_floating_point()
    ):
        raise ValueError("saved model contains nonfinite values")
    model.load_state_dict(values, strict=True, assign=True)
    for names in aliases.values():
        owner = model.get_parameter(names[0])
        for name in names[1:]:
            parent, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(parent), attribute, owner)
    parameters = dict(model.named_parameters())
    if frozen is not None:
        if parameters.keys() != frozen.keys() or any(
            type(value) is not bool for value in frozen.values()
        ):
            raise ValueError("artifact frozen-parameter schema mismatch")
        for name, is_frozen in frozen.items():
            parameters[name].requires_grad_(not is_frozen)
    model.train(training)
    return model
