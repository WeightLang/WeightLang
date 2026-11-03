"""Experimental dynamic tensor programs, compiled to the local Torch backend.

This is a trusted-source language frontend, NOT a sandbox. Host bindings are
explicit dependencies. Checking parses only; loading executes the tensor program.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
from dataclasses import dataclass, is_dataclass, fields
import hashlib
import importlib
import json
from pathlib import Path
import re
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .syntax import WgtError

HEADER = "wgt 0.3\nbackend torch-dynamic\n"
FORMAT = "weightlang.tensor-program"
_BUNDLE = ContextVar("weightlang_tensor_bundle", default=None)

# Constructor names are part of this frontend, not user supplied Python imports.
OPERATORS = {
    "linear": nn.Linear, "embedding": nn.Embedding, "layernorm": nn.LayerNorm,
    "gru": nn.GRU, "gru_cell": nn.GRUCell, "conv1d": nn.Conv1d,
    "gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh,
    "sequential": nn.Sequential, "parameter": nn.Parameter,
    "transformer_encoder": nn.TransformerEncoder,
    "transformer_layer": nn.TransformerEncoderLayer,
    "multihead_attention": nn.MultiheadAttention,
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


@dataclass
class TensorProgram:
    source: str
    path: str
    host: str
    exports: tuple[str, ...]
    tree: ast.Module

    def classes(self, bindings):
        namespace = dict(bindings)
        namespace.update(OPERATORS)
        namespace.update(torch=torch, F=F, Module=nn.Module)
        namespace["__name__"] = self.host
        exec(compile(self.tree, self.path, "exec"), namespace)
        result = {}
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[node.name] = namespace[node.name]
                continue
            if not isinstance(node, ast.ClassDef):
                continue
            cls = namespace[node.name]
            cls._wgt_source = self.source
            cls._wgt_host = self.host
            cls._wgt_path = self.path
            cls._wgt_bindings = {name: bindings[name] for name in self.exports}
            cls._wgt_functions = {n: v for n, v in vars(cls).items() if callable(v)}
            result[node.name] = cls
        return result


def parse_tensor_program(source, path="<tensor-program>"):
    if not source.startswith(HEADER):
        raise WgtError("expected wgt 0.3 and backend torch-dynamic", path=path)
    lines = source.splitlines()
    if len(lines) < 4 or not lines[2].startswith("host ") or not (lines[3] == "extern" or lines[3].startswith("extern ")):
        raise WgtError("tensor program requires host and extern declarations", path=path)
    host = lines[2][5:]
    if not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", host):
        raise WgtError("invalid host module", path=path)
    exports = tuple(filter(None, (s.strip() for s in lines[3][7:].split(","))))
    if any(not name.isidentifier() or name.startswith("__") for name in exports):
        raise WgtError("invalid extern binding", path=path)
    lowered = ["", "", "", ""]
    for line in lines[4:]:
        line = re.sub(r"^(\s*)module (\w+)(.*):$", r"\1class \2\3:", line)
        line = re.sub(r"^(\s*)setup\(", r"\1def __init__(", line)
        line = re.sub(r"^(\s*)fn (\w+)\(", r"\1def \2(", line)
        line = re.sub(r"^(\s*)async fn (\w+)\(", r"\1async def \2(", line)
        lowered.append(line)
    try:
        tree = ast.parse("\n".join(lowered), filename=path)
    except SyntaxError as exc:
        raise WgtError(exc.msg, exc.lineno, path) from None
    if not tree.body or any(not isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body):
        raise WgtError("only module and fn declarations are allowed at top level", path=path)
    names = [node.name for node in tree.body]
    if len(names) != len(set(names)):
        raise WgtError("duplicate module declaration", path=path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise WgtError("imports/global/nonlocal are not tensor-program statements; use extern", node.lineno, path)
        if isinstance(node, ast.Name) and node.id in {"exec", "eval", "compile", "open", "__import__"}:
            raise WgtError("unsupported tensor-program operation", node.lineno, path)
    # Future annotations avoid evaluating host type annotations at compile time.
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    # classes() only iterates real module declarations.
    return TensorProgram(source, path, host, exports, tree)


def bind_tensor_program(path, bindings):
    path = Path(path)
    source = path.read_text()
    parsed = parse_tensor_program(source, str(path))
    bundle = _BUNDLE.get()
    if bundle is not None and parsed.host in bundle:
        parsed = parse_tensor_program(bundle[parsed.host], str(path))
    missing = set(parsed.exports) - bindings.keys()
    if missing:
        raise WgtError(f"missing host bindings: {sorted(missing)}", path=str(path))
    return parsed.classes({name: bindings[name] for name in parsed.exports} | {"__builtins__": __builtins__})


def _source_bundle(module):
    result = {}
    for child in module.modules():
        for cls in type(child).__mro__:
            if "_wgt_source" in vars(cls):
                result[cls._wgt_host] = cls._wgt_source
    return result


def _encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {"$dataclass": type(value).__module__ + ":" + type(value).__name__,
                "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if "$dataclass" in value:
            module, name = value["$dataclass"].split(":")
            cls = getattr(importlib.import_module(module), name)
            if not is_dataclass(cls):
                raise WgtError("constructor record is not a dataclass")
            return cls(**_decode(value["fields"]))
        if "$tuple" in value:
            return tuple(_decode(v) for v in value["$tuple"])
        return {k: _decode(v) for k, v in value.items()}
    return value


def _construct(sources, host, name, args, kwargs):
    # Compile bundled sources against installed host adapters, then link every
    # compiled namespace to the bundled classes. Never use mutable local WGT.
    classes = {}
    for key, source in sources.items():
        program = parse_tensor_program(source)
        if program.host != key:
            raise WgtError("bundle host mismatch")
        adapter = importlib.import_module(key)
        bindings = {n: getattr(adapter, n) for n in program.exports}
        classes[key] = program.classes(bindings)
    by_name = {name: cls for group in classes.values() for name, cls in group.items()}
    # Class inheritance across source files must resolve against bundled bases.
    for _ in range(len(classes) + 1):
        for key, source in sources.items():
            program = parse_tensor_program(source)
            adapter = importlib.import_module(key)
            bindings = {n: by_name.get(n, getattr(adapter, n)) for n in program.exports}
            classes[key] = program.classes(bindings)
            by_name.update(classes[key])
    for group in classes.values():
        for cls in group.values():
            for value in (vars(cls).values() if isinstance(cls, type) else [cls]):
                function = getattr(value, "__func__", value)
                namespace = getattr(function, "__globals__", None)
                if namespace is not None:
                    namespace.update(by_name)
    return classes[host][name](*_decode(args), **_decode(kwargs))


@dataclass
class TensorArtifact:
    path: Path
    manifest: dict
    module: nn.Module

    @property
    def name(self):
        return self.manifest["model"]

    @property
    def untrained_tensors(self):
        return [n for n, m in self.manifest["weights"]["tensors"].items() if m["origin"] == "random-init"]

    def predict(self, *args, **kwargs):
        with torch.inference_mode():
            return self.module(*args, **kwargs)

    def save(self, path, *, training=None):
        if Path(path).resolve().is_relative_to(self.path.resolve()):
            raise WgtError("output must be outside the baseline model directory")
        baseline = load_tensor_artifact(self.path, warn_untrained=False, trusted=True)
        if baseline.manifest != self.manifest:
            raise WgtError("baseline changed")
        if _source_bundle(self.module) != _source_bundle(baseline.module):
            raise WgtError("tensor-program source changed")
        before = baseline.module.state_dict()
        for name, parameter in self.module.named_parameters():
            if not dict(baseline.module.named_parameters())[name].requires_grad and (parameter.requires_grad or not torch.equal(parameter.cpu(), before[name])):
                raise WgtError(f"frozen parameter changed: {name}")
        return save_tensor_module(self.module, path, args=_decode(self.manifest["args"]), kwargs=_decode(self.manifest["kwargs"]),
                                  training=training, origin="edited", sources=self.manifest["sources"], metadata=self.manifest.get("application_metadata"))


def save_tensor_module(module, path, *, args=(), kwargs=None, training=None, origin="random-init", sources=None, metadata=None):
    from safetensors.torch import save_file
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise WgtError("output already exists; choose a new model directory")
    kwargs = kwargs or {}
    args, kwargs = _encode(list(args)), _encode(kwargs)
    json.dumps([args, kwargs, training, metadata], allow_nan=False)
    if training is not None and (not isinstance(training, dict) or not training.get("method")):
        raise WgtError("training requires a method")
    sources = sources or _source_bundle(module)
    if not sources:
        raise WgtError("module was not compiled from a tensor program")
    for child in module.modules():
        if child._forward_hooks or child._forward_pre_hooks or "forward" in child.__dict__:
            raise WgtError("cannot save patched forwards or hooks")
    for child in module.modules():
        for base in type(child).__mro__:
            for name, function in vars(base).get("_wgt_functions", {}).items():
                if vars(base).get(name) is not function:
                    raise WgtError("tensor program method was replaced")
    state = {k: v.detach().cpu().contiguous().clone() for k, v in module.state_dict().items()}
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise WgtError("nonfinite weights")
    cls = type(module)
    with torch.random.fork_rng(devices=[]):
        reference = _construct(sources, cls._wgt_host, cls.__name__, args, kwargs)
    def structure(value):
        return [(name, type(child).__name__, child.extra_repr()) for name, child in value.named_modules()]
    if structure(module) != structure(reference):
        raise WgtError("module configuration differs from tensor program")
    expected = reference.state_dict()
    if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype for k in state):
        raise WgtError("state differs from declared constructor")
    parameters = dict(module.named_parameters())
    manifest = {"format": FORMAT, "format_version": 1, "model": cls.__name__, "host": cls._wgt_host,
                "args": list(args), "kwargs": kwargs, "sources": sources, "application_metadata": metadata or {},
                "source_hashes": {k: digest(v.encode()) for k, v in sources.items()}, "training": training,
                "weights": {"file": "weights.safetensors", "tensors": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype), "origin": "trained" if training else origin,
                        "requires_grad": parameters[k].requires_grad if k in parameters else False}
                    for k, v in state.items()}}}
    import inspect
    hosts = {}
    for host, source in sources.items():
        adapter = importlib.import_module(host)
        dependencies = [adapter]
        for name in parse_tensor_program(source).exports:
            value = getattr(adapter, name)
            owner = getattr(value, "__module__", "")
            if owner.startswith(host.split(".")[0] + "."):
                dependencies.append(importlib.import_module(owner))
        for dependency in dependencies:
            file = inspect.getsourcefile(dependency)
            if file:
                hosts[dependency.__name__] = digest(Path(file).read_bytes())
    manifest["host_hashes"] = hosts
    output.mkdir(parents=True, exist_ok=False)
    try:
        save_file(state, str(output / "weights.safetensors"))
        manifest["weights"]["sha256"] = digest((output / "weights.safetensors").read_bytes())
        (output / "source.wgt").write_text(cls._wgt_source)
        manifest["source_sha256"] = digest((output / "source.wgt").read_bytes())
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    except Exception:
        import shutil
        shutil.rmtree(output)
        raise
    return load_tensor_artifact(output, warn_untrained=False, trusted=True)


def load_tensor_artifact(path, *, warn_untrained=True, device="cpu", trusted=False):
    if not trusted:
        raise WgtError("tensor programs execute trusted source; pass trusted=True explicitly")
    from safetensors.torch import load_file
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("format_version") != 1:
        raise WgtError("unsupported tensor program artifact")
    for name, source in manifest["sources"].items():
        if digest(source.encode()) != manifest["source_hashes"][name]:
            raise WgtError("bundled source hash mismatch")
    if digest((root / "source.wgt").read_bytes()) != manifest["source_sha256"]:
        raise WgtError("source hash mismatch")
    if (root / "source.wgt").read_text() != manifest["sources"][manifest["host"]]:
        raise WgtError("entry source differs from bundle")
    if manifest["weights"]["file"] != "weights.safetensors" or digest((root / "weights.safetensors").read_bytes()) != manifest["weights"]["sha256"]:
        raise WgtError("weights hash mismatch")
    import inspect
    for host, expected_hash in manifest.get("host_hashes", {}).items():
        adapter = importlib.import_module(host)
        file = inspect.getsourcefile(adapter)
        if file is None or digest(Path(file).read_bytes()) != expected_hash:
            raise WgtError(f"host adapter changed: {host}")
    with torch.random.fork_rng(devices=[]):
        module = _construct(manifest["sources"], manifest["host"], manifest["model"], manifest["args"], manifest["kwargs"])
    state = load_file(str(root / "weights.safetensors"))
    expected = module.state_dict()
    if state.keys() != expected.keys() or state.keys() != manifest["weights"]["tensors"].keys():
        raise WgtError("tensor keys mismatch")
    for key, value in state.items():
        meta = manifest["weights"]["tensors"][key]
        if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise WgtError(f"tensor contract mismatch: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise WgtError(f"nonfinite tensor: {key}")
    module.load_state_dict(state, strict=True)
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(manifest["weights"]["tensors"][name]["requires_grad"])
    module.to(device).eval()
    artifact = TensorArtifact(root, manifest, module)
    if warn_untrained and artifact.untrained_tensors:
        warnings.warn("tensor program contains initialized, untrained weights", stacklevel=2)
    return artifact


def build_tensor_program(source, out_dir="weights", *, model=None, kwargs=None, device="cpu"):
    path = Path(source)
    program = parse_tensor_program(path.read_text(), str(path))
    adapter = importlib.import_module(program.host)
    classes = program.classes({n: getattr(adapter, n) for n in program.exports})
    names = [n for n, value in classes.items() if isinstance(value, type) and issubclass(value, nn.Module)]
    if model is None:
        if len(names) != 1:
            raise WgtError("select --tensor-model from " + ", ".join(names))
        model = names[0]
    if model not in names:
        raise WgtError(f"unknown tensor module {model}")
    kwargs = kwargs or {}
    module = classes[model](**kwargs).to(device)
    return save_tensor_module(module, Path(out_dir) / model, kwargs=kwargs)
