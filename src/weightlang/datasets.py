"""Pinned file datasets and deferred Python loaders."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import io
import json
import math

from .syntax import WgtError


def _rows(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("inputs and targets must be arrays of flat rows")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("each sample must be a nonempty flat row")
        if any(isinstance(v, (bool, str, list, dict)) for v in row):
            raise ValueError("dataset values must be numbers")
        result.append([float(v) for v in row])
    return result


def assign(ds, value):
    if not isinstance(value, dict) or set(value) != {"inputs", "targets"}:
        raise ValueError("loader must return a mapping with inputs and targets")
    inputs, targets = _rows(value["inputs"]), _rows(value["targets"])
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have equal row counts")
    ds.inputs, ds.targets, ds.loaded = inputs, targets, True


def load_file(ds, input_width, source_path, stack=()):
    from .syntax import _FileParser, _Loader

    path = ((Path(source_path).parent if source_path else Path.cwd()) / ds.file).resolve()
    try:
        if path in stack:
            raise ValueError("cyclic data file reference")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read data file {ds.file}: {exc.strerror}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ds.sha256:
            raise ValueError(f"data file {ds.file} has sha256 {digest}, source pins {ds.sha256}")
        fmt = ds.format or path.suffix.lstrip(".").lower()
        if ds.skip and fmt not in {"csv", "tsv", "jsonl"}:
            raise ValueError("skip only applies to csv, tsv and jsonl")
        if ds.dataset and fmt != "wgt":
            raise ValueError("dataset selection only applies to wgt files")
        if fmt in {"csv", "tsv"}:
            lines = raw.decode("utf-8").splitlines()[ds.skip:]
            rows = [row for row in csv.reader(
                (line for line in lines if line.strip() and not line.lstrip().startswith("#")),
                delimiter="," if fmt == "csv" else "\t")]
            try:
                rows = [list(map(float, row)) for row in rows]
            except ValueError:
                raise ValueError("data files must contain only numbers") from None
            if any(len(r) <= input_width for r in rows):
                raise ValueError(f"expected {input_width} inputs followed by targets")
            value = {"inputs": [r[:input_width] for r in rows], "targets": [r[input_width:] for r in rows]}
        elif fmt == "json":
            value = json.loads(raw)
        elif fmt == "jsonl":
            records = [json.loads(line) for line in raw.decode("utf-8").splitlines()[ds.skip:] if line.strip()]
            if any(not isinstance(r, dict) or set(r) != {"inputs", "targets"} for r in records):
                raise ValueError("JSONL rows require inputs and targets")
            value = {key: [r[key] for r in records] for key in ("inputs", "targets")}
        elif fmt == "npz":
            import numpy as np
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                value = {key: arrays[key].tolist() for key in ("inputs", "targets")}
        elif fmt == "wgt":
            parser = _FileParser(raw.decode("utf-8"), str(path), _Loader(), library=False)
            parser.data_only = True
            parser.run()
            if parser.version != "0.3":
                raise ValueError("standalone data files require wgt 0.3")
            selected = ds.dataset or ds.name
            if selected not in parser.datasets:
                raise ValueError(f"data file has no dataset {selected!r}; select with dataset=")
            nested = parser.datasets[selected]
            if nested.python or nested.entrypoint:
                raise ValueError("declare Python loaders in the model file")
            if nested.file:
                load_file(nested, input_width, str(path), (*stack, path))
            value = {"inputs": nested.inputs, "targets": nested.targets}
        else:
            raise ValueError(f"unsupported data format {fmt!r}")
        assign(ds, value)
    except Exception as exc:
        raise WgtError(f"cannot load dataset {ds.name!r}: {exc}", ds.line, source_path) from exc


@dataclass(frozen=True)
class DataContext:
    name: str
    source_dir: Path
    input_shape: tuple
    output_shape: tuple
    input_dtype: str
    seed: int


def materialize(program):
    from .plugins import load_callback
    from .syntax import check

    for ds in program.datasets.values():
        if ds.python or ds.entrypoint:
            callback, record = load_callback(ds.python, ds.entrypoint, "weightlang.data_loaders", program.path, "load")
            context = DataContext(ds.name, Path(program.path).resolve().parent,
                                  program.input.shape, program.output.shape, program.input.dtype, program.seed)
            try:
                assign(ds, callback(context))
                ds.loader_record = record
            except Exception as exc:
                raise WgtError(f"data loader {ds.name!r} failed: {exc}", ds.line, program.path) from exc
        elif ds.file and not ds.loaded:
            load_file(ds, math.prod(program.input.shape), program.path)
    check(program)
