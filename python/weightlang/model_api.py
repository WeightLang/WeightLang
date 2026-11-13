"""Default-native model entrypoints; PyTorch is an explicit backend."""
from copy import deepcopy
from pathlib import Path
import json
import os
import stat

from .model import Model


class TorchModel(Model):
    """A WGT object delegating execution to an explicitly selected torch module."""
    _wgt_backend = "torch"

    def __init__(self, module):
        self.module = module

    @classmethod
    def from_source(cls, source, **options):
        from .api import loads as build
        return cls(build(source, **options))

    @classmethod
    def from_file(cls, path, **options):
        from .api import load as build
        return cls(build(path, **options))

    @property
    def definition(self):
        return deepcopy(self.module.definition)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def __getattr__(self, name):
        return getattr(self.module, name)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def to_torch(self):
        return self.module

    def close(self):
        pass

    def __enter__(self):
        return self

    def train_step(self, *args, **kwargs):
        raise TypeError("backend='torch' uses host PyTorch optimizers")

    train_recurrent = train_step
    set_parameters = train_step


def _backend(backend):
    if backend not in {"native", "torch"}:
        raise ValueError("backend must be 'native' or 'torch'; there is no automatic fallback")


def from_ir(definition, *, backend="native", **options):
    _backend(backend)
    if backend == "torch":
        from .api import from_ir as build
        return TorchModel(build(definition, **options))
    return Model(definition, **options)


def loads(source, *, backend="native", compiler=None, dimensions=None, input_shapes=None, timeout=30, **options):
    from .compiler import compile_source
    _backend(backend)
    definition = compile_source(source, compiler=compiler, dimensions=dimensions, input_shapes=input_shapes, timeout=timeout, target="wgt" if backend == "native" else "torch")
    if backend == "native":
        options["timeout"] = timeout
    return from_ir(definition, backend=backend, **options)


def load(path, *, backend="native", compiler=None, dimensions=None, input_shapes=None, timeout=30, **options):
    _backend(backend)
    if backend == "torch":
        from .api import load as build
        return TorchModel(build(path, compiler=compiler, dimensions=dimensions, input_shapes=input_shapes, timeout=timeout, **options))
    from .compiler import compile
    path = Path(path)
    if path.suffix == ".wgt":
        definition = compile(path, compiler=compiler, dimensions=dimensions, input_shapes=input_shapes, timeout=timeout, target="wgt")
    elif path.suffix == ".json":
        if dimensions or input_shapes:
            raise ValueError("saved definitions cannot be rebound")
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("compiled definition exceeds 16 MiB")
        descriptor=os.open(path,os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(descriptor,"rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("compiled definitions require a regular file")
            data=stream.read(16 * 1024 * 1024 + 1)
        if len(data)>16 * 1024 * 1024:
            raise ValueError("compiled definition exceeds 16 MiB")
        definition = json.loads(data)
    else:
        raise ValueError("native Model loads .wgt or compiled .json; use backend='torch' for SDK .pt files")
    return Model(definition, timeout=timeout, **options)


def save(model, path, **options):
    if not isinstance(model, TorchModel):
        raise TypeError("save currently requires an explicit torch WGT Model; native parameters() are detached snapshots, not optimizer checkpoints")
    from .api import save as write
    return write(model.module, path, **options)

class DynamicModel(Model):
    """One native session, with immutable-source shape specialization."""
    def __init__(self, compile_shape, *, input_shapes, **options):
        self._compile_shape = compile_shape
        self._shapes = {name: tuple(shape) for name, shape in input_shapes.items()}
        definition = compile_shape(self._shapes)
        self._fingerprint = definition.get("template_fingerprint")
        if definition.get("kind") != "forward" or any(op.get("op") == "write" for op in definition.get("program", {}).get("instructions", [])):
            raise ValueError("dynamic native models require pure shared forward without persistent writes")
        if not isinstance(self._fingerprint, str) or len(self._fingerprint) != 64 or any(c not in '0123456789abcdef' for c in self._fingerprint):
            raise ValueError("dynamic Model requires compiler template fingerprints")
        super().__init__(definition, **options)

    def specialize(self, input_shapes):
        self._ensure_native_process()
        with self._lock:
            if self._closed:
                raise ConnectionError("WGT model is closed")
            shapes = {name: tuple(shape) for name, shape in input_shapes.items()}
            if shapes == self._shapes:
                return self
            definition = self._compile_shape(shapes)
            if definition.get("template_fingerprint") != self._fingerprint:
                raise ValueError("dynamic model source changed; reload explicitly")
            result = self._call("RebindDefinition", source=definition["source"])
            self._definition = deepcopy(definition)
            self._definition["forward_interface"] = result["forward_interface"]
            self._shapes = shapes
            self.execution_backend = result.get("backend")
            return self

    def forward(self, *inputs, **named):
        from .model import _tensor
        self._ensure_native_process()
        with self._lock:
            ports = self._definition.get("forward_interface")
            names = [p["name"] for p in ports["inputs"]] if ports else list(self._shapes)
            if named:
                if inputs or set(named) != set(names):
                    raise TypeError(f"provide exactly inputs {names}")
                inputs = tuple(named[name] for name in names)
            if len(inputs) != len(names):
                raise TypeError(f"forward requires {len(names)} inputs")
            values = [_tensor(value) for value in inputs]
            self.specialize({name: shape for name, (shape, _) in zip(names, values)})
            if ports:
                return super().forward(*inputs)
            return self._call("Predict", input=values[0][1])["output"]


def load_dynamic(path, *, input_shapes, backend="native", dimensions=None, compiler=None, timeout=30, **options):
    _backend(backend)
    if backend == "torch":
        from .dynamic import load_dynamic as build
        return TorchModel(build(path, input_shapes=input_shapes, dimensions=dimensions, compiler=compiler, timeout=timeout, **options))
    from .compiler import compile, _compiler
    executable = _compiler(compiler)
    source = Path(path).expanduser().resolve()
    return DynamicModel(lambda shapes: compile(source, compiler=executable, dimensions=dimensions, input_shapes=shapes, timeout=timeout, target="wgt"), input_shapes=input_shapes, timeout=timeout, **options)


def loads_dynamic(source, *, input_shapes, backend="native", dimensions=None, compiler=None, timeout=30, **options):
    _backend(backend)
    if backend == "torch":
        from .dynamic import loads_dynamic as build
        return TorchModel(build(source, input_shapes=input_shapes, dimensions=dimensions, compiler=compiler, timeout=timeout, **options))
    from .compiler import compile_source, _compiler
    executable = _compiler(compiler)
    return DynamicModel(lambda shapes: compile_source(source, compiler=executable, dimensions=dimensions, input_shapes=shapes, timeout=timeout, target="wgt"), input_shapes=input_shapes, timeout=timeout, **options)


def compile(path, *, target="wgt", **options):
    """Compile a backend-independent WGT definition, never create an instance."""
    from .compiler import compile as build
    return build(path, target=target, **options)


def compile_source(source, *, target="wgt", **options):
    from .compiler import compile_source as build
    return build(source, target=target, **options)
