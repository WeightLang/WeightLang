"""Native CPU/GPU inference with measured device selection; no silent Torch execution fallback.

The JSON transport allocates. Only the Rust Session::run core has a zero-heap-
allocation contract. Use the Rust API directly when the entire hot path needs it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def lower(source, parameters, *, config=None):
    """Lower a static vector WGT chain with explicit flat f32 parameter values.

    No initialization, imports, data loaders or training callbacks are executed.
    Parameter values may instead be native IR storage/expression dictionaries.
    Mapping insertion order determines expression declaration order.
    """
    from .syntax import parse_file, WgtError

    program = parse_file(source, check_training=False, config=config)
    if program.imports or program.inherit or program.derivation:
        raise WgtError("explicit native parameters cannot override import/derive declarations")
    return _lower_program(program, parameters)


def _structure(program):
    from .syntax import WgtError
    unsupported = (program.derivation or program.adapters or program.weight_types or
                   program.weight_assignments or getattr(program, "precision", {}) or program.dimensions or
                   (program.returns and program.returns != {program.output.name: program.layers[-1].name}) or
                   len(program.inputs) > 1 or len(program.outputs) > 1)
    if unsupported:
        raise WgtError("weightlang-runtime does not support this model contract; use backend='torch' explicitly")
    if program.input.dtype != "f32" or len(program.input.shape) != 1 or program.output.dtype != "f32" or len(program.output.shape) != 1:
        raise WgtError("native backend requires one static f32 vector input/output; use backend='torch' for other contracts")
    shapes, layers = {}, []

    def visit(items, prefix=""):
        previous = None
        for layer in items:
            if layer.results or layer.returns or (layer.inputs and layer.inputs != [previous] and not (previous is None and layer.inputs in (["input"], [program.input.name]))):
                raise WgtError(f"native backend requires a sequential chain: {prefix}{layer.name}; use backend='torch'")
            path = prefix + layer.name
            if layer.op == "seq":
                visit(layer.children, path + ".")
            elif layer.children:
                raise WgtError(f"native backend unsupported structure: {layer.op}; use backend='torch'")
            elif layer.op == "linear":
                if layer.args.get("tie_embedding", 0):
                    raise WgtError("native backend does not support tie_embedding")
                weight = path + ".weight"
                shapes[weight] = [layer.args["out"], layer.args["in"]]
                bias = None
                if layer.args.get("bias", 1):
                    bias = path + ".bias"
                    shapes[bias] = [layer.args["out"]]
                layers.append(dict(op="linear", weight=weight, bias=bias))
            elif layer.op in {"relu", "tanh", "sigmoid", "identity"}:
                layers.append(dict(op=layer.op))
            else:
                raise WgtError(f"native backend unsupported operator: {layer.op} at {path}; use backend='torch'")
            previous = layer.name
    visit(program.layers)
    return shapes, layers


def _lower_program(program, parameters):
    from .syntax import WgtError
    shapes, layers = _structure(program)
    records = []
    for name, value in parameters.items():
        if name in program.parameter_shares:
            raise WgtError(f"provide the owner, not shared destination {name}")
        if isinstance(value, dict):
            record = dict(value)
            if "name" in record:
                raise WgtError("parameter mapping key is its identity; do not supply name")
            record.setdefault("shape", shapes.get(name))
        else:
            if name not in shapes:
                raise WgtError(f"auxiliary parameter {name} needs an explicit shape")
            record = dict(shape=shapes[name], kind="resident", data=list(value))
        if name in shapes and record["shape"] != shapes[name]:
            raise WgtError(f"native parameter shape mismatch: {name}")
        records.append(dict(name=name, **record))
    available = set(parameters)
    for destination, identity in program.parameter_shares.items():
        owner = program.parameter_ids[identity]
        if owner not in available:
            raise WgtError(f"missing native parameter owner: {owner}")
        records.append(dict(name=destination, shape=shapes[destination], kind="view", source=owner, start=0))
        available.add(destination)
    missing = set(shapes) - available
    if missing:
        raise WgtError(f"missing native parameter values: {sorted(missing)}")
    # Match freezing against canonical source parameter paths, as apply_freezing does.
    frozen = {name: True for imp in program.imports if imp.freeze
              for name in shapes if name not in program.parameter_shares
              and (name == imp.target or name.startswith(imp.target + "."))}
    for rule in program.freezing:
        matches = [name for name in shapes if name not in program.parameter_shares
                   and (rule.target == "all" or name == rule.target or name.startswith(rule.target + "."))]
        if not matches:
            raise WgtError(f"freeze selector {rule.target!r} matches no parameters")
        frozen.update({name: rule.frozen for name in matches})
    for record in records:
        if record["name"] in frozen:
            record["frozen"] = frozen[record["name"]]
    # Rust validates every expression, shape, reference, file range and budget.
    return dict(version=1, input=program.input.shape[0], parameters=records, layers=layers)


class NativeSession:
    """Persistent wgt-native subprocess. Use as a context manager for deterministic close.

    The IR path's parent resolves relative parameter files. Disk sources must remain
    immutable for the lifetime of the session. f32 inference, single vector only. Auto measures the available CPU/GPU candidates.
    """

    def __init__(self, model, *, budget_bytes, executable=None, device="auto", strict_zero_alloc=False):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("budget_bytes must be a nonnegative integer")
        binary = _executable(executable)
        if binary is None:
            raise RuntimeError("build cargo build -p weightlang-runtime, then set WEIGHTLANG_NATIVE")
        if type(device) is not str or type(strict_zero_alloc) is not bool:
            raise TypeError("device must be a string and strict_zero_alloc must be a bool")
        command = [str(binary), str(Path(model).resolve()), str(budget_bytes), "--device", device]
        if strict_zero_alloc:
            command.append("--strict-zero-alloc")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            self.report = self._receive()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_source(cls, source, parameters, *, budget_bytes, executable=None, config=None, device="auto", strict_zero_alloc=False):
        """Lower and prepare without retaining an artifact or accessing training data.

        File-backed parameter paths must be absolute in this in-memory convenience API.
        Use an explicit IR file for relative paths or large embedded resident values.
        """
        model = lower(source, parameters, config=config)
        return cls.from_ir(model, budget_bytes=budget_bytes, executable=executable, device=device, strict_zero_alloc=strict_zero_alloc)

    @classmethod
    def from_ir(cls, model, *, budget_bytes, executable=None, device="auto", strict_zero_alloc=False):
        """Prepare an in-memory native IR; external parameter file paths must be absolute."""
        for record in model["parameters"]:
            if record.get("kind") == "file" and not Path(record["path"]).is_absolute():
                raise ValueError("from_source requires absolute weight file paths")
        # Control data only. Large model payloads belong in disk-backed parameter files.
        encoded = json.dumps(model, allow_nan=False)
        if len(encoded) > 8 * 1024 * 1024:
            raise ValueError("native source control data exceeds 8 MiB; use file-backed parameters")
        scratch = Path(tempfile.gettempdir()) / "weightlang"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-", dir=scratch) as directory:
            path = Path(directory) / "model.json"
            path.write_text(encoded)
            return cls(path, budget_bytes=budget_bytes, executable=executable, device=device, strict_zero_alloc=strict_zero_alloc)

    def _receive(self):
        line = self._process.stdout.readline()
        if not line:
            error = self._process.stderr.read()
            self._process.wait()
            raise RuntimeError(error.strip() or "native backend closed unexpectedly")
        return json.loads(line)

    def __call__(self, values):
        if getattr(values, "requires_grad", False):
            raise ValueError("native inference has no autograd; use backend='torch' for training")
        if hasattr(values, "tolist"):
            values = values.tolist()
        return self._exchange(list(values))

    def _exchange(self, request):
        if self._process.poll() is not None:
            raise RuntimeError("native session is closed")
        encoded = json.dumps(request, allow_nan=False)
        self._process.stdin.write(encoded + "\n")
        self._process.stdin.flush()
        return self._receive()

    def _command(self, **request):
        response = self._exchange(request)
        if "error" in response:
            raise ValueError(response["error"])
        return response["ok"]

    def parameter_info(self):
        """Ordered names, shapes and effective writability; no parameter payloads."""
        return self._command(op="parameters")

    def read_parameter(self, name, *, start=0, length=None):
        """Read a flat f32 slice; length=None reads through the parameter end."""
        return self._command(op="read", name=name, start=start, len=length)

    def write_parameters(self, writes):
        """Apply [{name, values, start=0}, ...] after validating the entire batch.

        Resident storage and its views are writable unless frozen; files and add
        expressions are read-only. Overlapping aliases are rejected. Returns the
        session revision. Updates become visible on the next CPU/GPU forward.
        """
        return self._command(op="write", writes=list(writes))

    def predict(self, values):
        """Host-side batching of independent vector calls, not a batched GPU kernel."""
        if isinstance(values, dict):
            name = getattr(self, "input_name", None)
            if set(values) != {name}:
                raise ValueError("native predict expects the single declared input")
            values = values[name]
        if getattr(values, "requires_grad", False):
            raise ValueError("native inference has no autograd; use backend='torch'")
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], (list, tuple)):
            return [self(row) for row in values]
        return self(values)

    def close(self):
        process = self._process
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def available_devices(*, executable=None):
    """List detected CPU/GPU candidates; execution still requires model validation."""
    binary = _executable(executable)
    if binary is None:
        raise RuntimeError("wgt-native not found; build the native runtime and set WEIGHTLANG_NATIVE")
    result = subprocess.run([str(binary), "--devices"], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _executable(executable=None):
    configured = executable or os.environ.get("WEIGHTLANG_NATIVE") or shutil.which("wgt-native")
    if configured:
        return configured
    root = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / ("wgt-native.exe" if os.name == "nt" else "wgt-native")
        if candidate.is_file():
            return str(candidate)
    return None
