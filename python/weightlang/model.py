"""First-class WGT models. Native execution owns a real wgtime session."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
import socket
import struct
import threading


class ServiceError(RuntimeError):
    """A complete error reply; the connection remains usable."""


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field in wgtime reply")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant in wgtime reply: {value}")


class NativeConnection:
    """Bounded, UID-checked IPC; incomplete exchanges close, never replay."""
    MAX_FRAME = 16 * 1024 * 1024

    def __init__(self, endpoint=None, *, timeout=30, expected_uid=None):
        if not hasattr(socket, "SO_PEERCRED"):
            raise OSError("native WGT execution currently requires Linux Unix sockets")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self.endpoint = os.fspath(endpoint or os.environ.get("WGTIME_ENDPOINT", "/run/wgtime/service.sock"))
        try:
            try:
                self._socket.connect(self.endpoint)
            except OSError as exc:
                raise ConnectionError(f"wgtime unavailable at {self.endpoint}; start the matching service or explicitly choose backend='torch'") from exc
            uid = struct.unpack("3i", self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
            if uid != (os.geteuid() if expected_uid is None else expected_uid):
                raise PermissionError("wgtime service identity mismatch")
            hello = self.call("Hello", protocol=1)
            if not isinstance(hello, dict) or type(hello.get("protocol")) is not int or hello["protocol"] != 1:
                raise ValueError("invalid wgtime handshake")
        except BaseException:
            self.close()
            raise

    def _read(self, count):
        result = bytearray()
        while len(result) < count:
            block = self._socket.recv(count - len(result))
            if not block:
                raise ConnectionError("wgtime closed an incomplete exchange")
            result.extend(block)
        return bytes(result)

    def call(self, operation, **fields):
        payload = json.dumps({operation: fields} if fields else operation, allow_nan=False, separators=(",", ":")).encode()
        if len(payload) > self.MAX_FRAME:
            raise ValueError("IPC frame exceeds 16 MiB")
        if self._pid != os.getpid():
            raise ConnectionError("wgtime connection is inherited across fork")
        with self._lock:
            if self._socket is None:
                raise ConnectionError("wgtime connection is closed or inherited across fork")
            try:
                self._socket.sendall(struct.pack("<I", len(payload)) + payload)
                size, = struct.unpack("<I", self._read(4))
                if size > self.MAX_FRAME:
                    raise ValueError("wgtime reply exceeds 16 MiB")
                reply = json.loads(self._read(size), object_pairs_hook=_json_object, parse_constant=_invalid_constant)
                if (not isinstance(reply, dict) or set(reply) != {"value", "error"}
                        or (reply["value"] is None) == (reply["error"] is None)
                        or (reply["error"] is not None and not isinstance(reply["error"], str))):
                    raise ValueError("invalid wgtime reply")
            except BaseException:
                self.close()
                raise
            if reply["error"] is not None:
                raise ServiceError(reply["error"])
            return reply["value"]

    def close(self):
        if self._pid != os.getpid():
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            return
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None


@dataclass(frozen=True)
class Parameter:
    """Detached native parameter value; mutate explicitly with set_parameters."""
    name: str
    shape: tuple[int, ...]
    data: tuple[float, ...]


def _tensor(value, _depth=0, _remaining=None):
    if _remaining is None:
        _remaining = [16384]
    if _depth > 8:
        raise ValueError("native tensors support at most eight axes")
    if getattr(value, "requires_grad", False):
        raise TypeError("native WGT does not join PyTorch autograd; use train_step or backend='torch'")
    if hasattr(value, "device") and str(value.device) != "cpu":
        raise TypeError("native WGT inputs require CPU values")
    if hasattr(value, "dtype") and str(value.dtype) not in {"float32", "torch.float32"}:
        raise TypeError("native WGT inputs require f32")
    if hasattr(value, "shape") and (len(value.shape) > 8 or math.prod(value.shape) > 16384):
        raise ValueError("native tensors support at most 16384 elements and eight axes")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) > 16384:
            raise ValueError("native tensor axis exceeds 16384")
        parts = [_tensor(item, _depth + 1, _remaining) for item in value]
        if not parts or any(shape != parts[0][0] for shape, _ in parts):
            raise ValueError("native tensors must be nonempty and rectangular")
        return (len(parts), *parts[0][0]), [x for _, data in parts for x in data]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("native inputs must contain finite f32 values")
    _remaining[0] -= 1
    if _remaining[0] < 0:
        raise ValueError("native tensor exceeds 16384 elements")
    # Reject f32 overflow before transmitting rather than silently rounding to inf.
    try:
        number, = struct.unpack("<f", struct.pack("<f", value))
    except (OverflowError, struct.error) as exc:
        raise ValueError("native input exceeds f32 range") from exc
    if not math.isfinite(number):
        raise ValueError("native input exceeds f32 range")
    return (), [number]


def _reshape(values, shape):
    if len(shape) == 1:
        return values
    width = math.prod(shape[1:])
    return [_reshape(values[i:i + width], shape[1:]) for i in range(0, len(values), width)]


class Model:
    """Customizable WGT object, defaulting to native Rust/wgtime execution.

    Source compilation is separate from this explicit instance construction.
    Use a context manager or close(). Native inputs/outputs are finite f32
    lists; train_step runs the service optimizer, not torch.autograd. Subclasses
    may override forward for host-side preparation; native tensor code stays WGT.
    """
    _wgt_backend = "native"

    def __init__(self, definition, *, endpoint=None, timeout=30, memory_bytes=64 * 1024 * 1024, optimizer=None, device="cpu", dtype=None, work_limit=None):
        if str(device) not in {"cpu", "auto"} or (dtype is not None and str(dtype) not in {"float32", "torch.float32"}):
            raise ValueError("native execution supports CPU/f32; choose backend='torch' for other devices/dtypes")
        if not isinstance(definition, dict) or definition.get("kind") not in {"forward", "sequential", "operations"}:
            raise ValueError("native Model requires shared WGT IR; target torch uses modules.bind_models(..., backend='torch')")
        if definition.get("format") not in {"weightlang.wgt.v1", "weightlang.torch.v1"}:
            raise ValueError("expected a versioned WGT model definition")
        if not isinstance(definition.get("source"), str):
            raise ValueError("native model definition must include its concrete WGT source")
        if type(memory_bytes) is not int or memory_bytes <= 0:
            raise ValueError("memory_bytes must be a positive integer")
        if optimizer not in {None, "sgd", "adam"}:
            raise ValueError("native optimizer must be None, 'sgd' or 'adam'")
        if work_limit is not None and (type(work_limit) is not int or not 1 <= work_limit <= 100_000_000 or definition["kind"] != "forward"):
            raise ValueError("work_limit requires a forward model and integer in 1..100000000")
        self._compact_parameters = sum(math.prod(p["shape"]) for p in definition.get("parameters",[])) > 100_000
        self.work_limit = None
        self._definition = deepcopy(definition)
        self._lock = threading.RLock()
        self._client = NativeConnection(endpoint, timeout=timeout)
        self._closed = False
        try:
            result = self._client.call("InitializeDefinition", source=definition["source"], memory=memory_bytes, optimizer=optimizer)
            if not isinstance(result, dict) or type(result.get("handle")) is not int or result["handle"] <= 0:
                raise ValueError("invalid native model handle")
            if definition.get("forward_interface") is not None and "forward_interface" not in result:
                raise ValueError("wgtime must export the authoritative forward interface; update the service")
            self._definition["forward_interface"] = result.get("forward_interface")
            self._handle = result["handle"]
            self.execution_backend = result.get("backend")
            if work_limit is not None:
                self.set_work_limit(work_limit)
        except BaseException:
            self._client.close()
            self._closed = True
            raise

    @property
    def backend(self):
        return self._wgt_backend

    @property
    def definition(self):
        return deepcopy(self._wgt_definition if self.backend == "torch" else self._definition)

    @classmethod
    def from_source(cls, source, *, compiler=None, dimensions=None, input_shapes=None, timeout=30, **options):
        from .compiler import compile_source
        return cls(compile_source(source, compiler=compiler, dimensions=dimensions, input_shapes=input_shapes, timeout=timeout, target="wgt"), timeout=timeout, **options)

    @classmethod
    def from_file(cls, path, *, compiler=None, dimensions=None, input_shapes=None, timeout=30, **options):
        from .compiler import compile
        return cls(compile(path, compiler=compiler, dimensions=dimensions, input_shapes=input_shapes, timeout=timeout, target="wgt"), timeout=timeout, **options)

    def _call(self, operation, **fields):
        if self.backend == "torch":
            raise TypeError("Torch-specific models use PyTorch host optimizers and methods")
        if self._closed:
            raise ConnectionError("WGT model is closed")
        return self._client.call(operation, handle=self._handle, **fields)

    def _ensure_native_process(self):
        if self.backend != "native":
            raise TypeError("backend='torch' uses PyTorch host methods")
        if self._closed or self._client._socket is None or self._client._pid != os.getpid():
            raise ConnectionError("WGT model is closed or inherited across fork")

    def forward(self, *inputs, **named):
        self._ensure_native_process()
        with self._lock:
            ports = self._definition.get("forward_interface")
            if ports:
                names = [port["name"] for port in ports["inputs"]]
                if named:
                    if inputs or set(named) != set(names):
                        raise TypeError(f"provide exactly the named inputs {names}, or all positionally")
                    inputs = tuple(named[name] for name in names)
                if len(inputs) != len(names):
                    raise TypeError(f"forward requires {len(names)} inputs")
                packed = []
                for value, port in zip(inputs, ports["inputs"]):
                    shape, data = _tensor(value)
                    if list(shape) != port["shape"]:
                        raise ValueError(f"input {port['name']} requires shape {port['shape']}")
                    packed.extend(data)
                values = self._call("Predict", input=packed)["output"]
                return tuple(_reshape(values[p["offset"]:p["offset"] + p["width"]], p["shape"]) for p in ports["outputs"])
            if named or len(inputs) != 1:
                raise TypeError("flat native forward requires one positional input")
            shape, data = _tensor(inputs[0])
            if len(shape) != 1:
                raise ValueError("flat native forward requires one f32 vector")
            if self._definition["kind"] == "operations":
                return self.call("forward", inputs[0])
            return self._call("Predict", input=data)["output"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def call(self, name, *inputs):
        self._ensure_native_process()
        with self._lock:
            values = [_tensor(value) for value in inputs]
            if any(len(shape) != 1 for shape, _ in values):
                raise ValueError("typed operations require fixed-width f32 vectors")
            return self._call("CallOperation", name=name, inputs=[data for _, data in values])["output"]

    def parameters(self):
        self._ensure_native_process()
        with self._lock:
            if self._compact_parameters:
                return tuple(Parameter(name,tuple(shape),tuple(v[0] for v in struct.iter_unpack(">f",bytes.fromhex(data)))) for name,shape,data in self._call("ParameterValuesHex"))
            return tuple(Parameter(name, tuple(shape), tuple(data)) for name, shape, data in self._call("ParameterValues"))

    def set_parameters(self, values):
        self._ensure_native_process()
        with self._lock:
            if self._compact_parameters:
                candidates=[(p.name,list(p.shape),b"".join(struct.pack(">f",v) for v in p.data).hex()) for p in values]
                return self._call("UpdateParametersHex",candidates=candidates)["revision"]
            candidates = [(p.name, list(p.shape), list(p.data)) for p in values]
            return self._call("UpdateParameters", candidates=candidates)["revision"]

    def set_work_limit(self, work):
        """Set the owned native forward call/training work budget, not weights."""
        self._ensure_native_process()
        if type(work) is not int or not 1 <= work <= 100_000_000:
            raise ValueError("work limit must be an integer in 1..100000000")
        with self._lock:
            self.work_limit = self._call("SetWorkLimit", work=work)["work_limit"]
            return self.work_limit

    def set_optimizer(self, optimizer):
        if self.backend != "native":
            raise TypeError("backend='torch' uses host PyTorch optimizers")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("native optimizer must be 'sgd' or 'adam'")
        self._ensure_native_process()
        with self._lock:
            return self._call("SetOptimizer", optimizer=optimizer)["revision"]

    def _check_policy(self, straight_through):
        if type(straight_through) is not bool:
            raise ValueError("straight_through must be bool")
        if not straight_through and any((p.get("learn") or {}).get("kind") in {"bool", "int"} for p in self._definition.get("program", {}).get("parameters", [])):
            raise ValueError("bool/int parameters require explicit straight_through=True")

    def train_step(self, inputs, targets, *, rate, loss="mse", clip=None, straight_through=False):
        self._ensure_native_process()
        with self._lock:
            rows = [_tensor(value) for value in inputs]
            if any(len(shape) != 1 for shape, _ in rows):
                raise ValueError("native training uses packed input vectors")
            self._check_policy(straight_through)
            return self._call("TrainStepWithPolicy", inputs=[data for _, data in rows], targets=targets, rate=rate, loss=loss, clip=clip, straight_through=straight_through)["loss"]

    def train_recurrent(self, inputs, targets, *, input_offset, output_offset, state_width, supervised_offset, supervised_width, rate, clip=None, straight_through=False):
        self._ensure_native_process()
        with self._lock:
            rows = [_tensor(value) for value in inputs]
            if any(len(shape) != 1 for shape, _ in rows):
                raise ValueError("native recurrent training uses packed input vectors")
            self._check_policy(straight_through)
            return self._call("TrainRecurrent", inputs=[data for _, data in rows], targets=targets, input_offset=input_offset, output_offset=output_offset, state_width=state_width, supervised_offset=supervised_offset, supervised_width=supervised_width, rate=rate, clip=clip)["loss"]

    def close(self):
        if self.backend == "torch":
            return
        if self._client._pid != os.getpid() or self._client._socket is None:
            self._client.close()
            self._closed = True
            return
        with self._lock:
            if not self._closed:
                try:
                    self._call("Drop")
                finally:
                    self._closed = True
                    self._client.close()

    def __enter__(self):
        if self.backend == "torch":
            return self
        self._ensure_native_process()
        return self

    def __exit__(self, *_):
        self.close()

    def to_torch(self):
        if self.backend == "torch":
            return self
        raise TypeError("choose backend='torch' when constructing a model; changing backend does not transfer live state")
