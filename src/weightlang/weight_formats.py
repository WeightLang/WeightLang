"""Local checkpoint readers. Inventories reference payloads; they never execute graphs."""

from __future__ import annotations

import importlib
import json
import math
import zipfile
from pathlib import Path

from .syntax import WgtError

EXTENSIONS = {".pt", ".pth", ".bin", ".ckpt", ".gguf", ".onnx", ".npy", ".npz"}
MAX_METADATA = 16 * 1024 * 1024


def optional(package: str):
    try:
        return importlib.import_module(package)
    except ImportError:
        raise WgtError(f"{package} support requires: pip install 'weightlang[formats]'") from None


def json_metadata(value):
    """Only JSON data, bounded and without object stringification or tensor materialization."""

    def visit(item, depth=0):
        if depth > 32:
            raise ValueError("metadata nesting exceeds 32 levels")
        if item is None or type(item) in (bool, str, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            return {k: visit(v, depth + 1) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(v, depth + 1) for v in item]
        raise ValueError("not finite JSON data")

    result = visit(value)
    if len(json.dumps(result, ensure_ascii=False).encode()) > MAX_METADATA:
        raise ValueError("metadata exceeds 16 MiB")
    return result


def source_record(path: Path, encoding: str, metadata=None):
    return {
        "uri": str(path.resolve()),
        "size": path.stat().st_size,
        "encoding": encoding,
        "payload_verified": False,
        "metadata": metadata or {},
    }


def document(sources, tensors, config=None):
    result = {
        "format": "weightlang.tensor_inventory",
        "version": 2,
        "executable": False,
        "revision": None,
        "config": config,
        "sources": sources,
        "tensors": tensors,
        "limitations": [
            "Original tensor payloads remain in referenced files; payload integrity is not verified.",
            "Source metadata is unverified data, not executable instructions.",
            "Inventory only: no native inference graph or output format conversion.",
        ],
    }
    return result


def pytorch(path: Path):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    # Older restricted unpicklers had known unsafe behavior; never fall back to unrestricted loading.
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise WgtError("PyTorch checkpoint inspection requires torch>=2.6")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if any(n.endswith("/constants.pkl") or n == "constants.pkl" for n in archive.namelist()):
                raise WgtError("TorchScript archives are unsupported; export a state_dict instead")
    try:
        with FakeTensorMode():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise WgtError(f"restricted PyTorch checkpoint load failed (no unsafe fallback): {exc}") from None
    if not isinstance(checkpoint, dict):
        raise WgtError("expected a tensor state_dict or checkpoint dictionary")
    candidates = [
        k
        for k in ("state_dict", "model_state_dict", "model")
        if isinstance(checkpoint.get(k), dict) and any(isinstance(v, torch.Tensor) for v in checkpoint[k].values())
    ]
    if len(candidates) > 1:
        raise WgtError("multiple state dictionaries found; select/export one explicitly")
    root = candidates[0] if candidates else None
    state = checkpoint[root] if root else checkpoint
    tensors, metadata, omitted = {}, {}, []
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise WgtError("state dictionary keys must be strings")
        if not isinstance(tensor, torch.Tensor):
            if root:
                raise WgtError(f"non-tensor entry in state dictionary: {name}")
            continue
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise WgtError(f"unsupported sparse/quantized PyTorch tensor: {name}")
        tensors[name] = {
            "source": 0,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "locator": {
                "kind": "state_dict",
                "path": [root, name] if root else [name],
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
            },
        }
    if not tensors:
        raise WgtError("checkpoint contains no tensor state dictionary")
    for key, value in checkpoint.items():
        if key == root or (root is None and key in tensors):
            continue
        try:
            if not isinstance(key, str):
                raise ValueError("non-string metadata key")
            metadata[key] = json_metadata(value)
        except ValueError:
            omitted.append(str(key) if isinstance(key, (str, int)) else "<non-string key>")
    state_metadata = None
    if hasattr(state, "_metadata"):
        try:
            state_metadata = json_metadata(state._metadata)
        except ValueError:
            omitted.append("state_dict._metadata")
    source = source_record(path, "pytorch", json_metadata(metadata))
    source["metadata_omitted"] = omitted
    if state_metadata is not None:
        source["state_dict_metadata"] = state_metadata
    return document([source], tensors)


def gguf(path: Path):
    library = optional("gguf")
    reader = library.GGUFReader(str(path), mode="r")
    try:
        metadata = {key: field.contents() for key, field in reader.fields.items() if not key.startswith("GGUF.")}
        source = source_record(path, "gguf", json_metadata(metadata))
        source["byte_order"] = reader.endianess.name.lower()
        tensors = {}
        spans = []
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise WgtError(f"duplicate GGUF tensor: {tensor.name}")
            start, end = int(tensor.data_offset), int(tensor.data_offset + tensor.n_bytes)
            if start < reader.data_offset or end > source["size"]:
                raise WgtError(f"GGUF tensor outside payload: {tensor.name}")
            # GGUF dimensions are fastest-axis-first; use conventional outermost-axis-first in .wgt.
            tensors[tensor.name] = {
                "source": 0,
                "dtype": "gguf." + tensor.tensor_type.name,
                "shape": [int(n) for n in reversed(tensor.shape)],
                "locator": {"kind": "byte_range", "offset": start, "length": end - start},
            }
            spans.append((start, end))
        cursor = reader.data_offset
        for start, end in sorted(spans):
            if start < cursor:
                raise WgtError("GGUF contains overlapping tensor ranges")
            cursor = end
        return document([source], tensors)
    finally:
        # All returned values are plain Python metadata, independent of the mmap.
        reader.data._mmap.close()


def onnx(path: Path):
    library = optional("onnx")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise WgtError("ONNX protobuf exceeds 256 MiB; export ONNX external-data weights first")
    model = library.load_model(str(path), load_external_data=False)

    def properties(entries):
        result = {}
        for entry in entries:
            if entry.key in result:
                raise WgtError(f"duplicate ONNX property: {entry.key}")
            result[entry.key] = entry.value
        return result

    metadata = {
        "metadata_props": properties(model.metadata_props),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": model.model_version,
        "doc_string": model.doc_string,
        "ir_version": model.ir_version,
        "opset_import": [{"domain": p.domain, "version": p.version} for p in model.opset_import],
    }
    sources = [source_record(path, "onnx", json_metadata(metadata))]
    tensors = {}
    source_indices = {}
    if model.graph.sparse_initializer or model.functions:
        raise WgtError("ONNX sparse initializers and local functions are not supported")
    entries = [
        (t.name, t, {"kind": "onnx_tensor", "path": ["graph", "initializer", i]})
        for i, t in enumerate(model.graph.initializer)
    ]
    for i, node in enumerate(model.graph.node):
        for j, attr in enumerate(node.attribute):
            if attr.type in (
                library.AttributeProto.GRAPH,
                library.AttributeProto.GRAPHS,
                library.AttributeProto.SPARSE_TENSOR,
                library.AttributeProto.SPARSE_TENSORS,
            ):
                raise WgtError("ONNX subgraphs and sparse tensor attributes are not supported")
            if attr.type == library.AttributeProto.TENSOR:
                entries.append(
                    (
                        f"@node/{i}/{attr.name}",
                        attr.t,
                        {"kind": "onnx_tensor", "path": ["graph", "node", i, "attribute", j, "t"]},
                    )
                )
            elif attr.type == library.AttributeProto.TENSORS:
                for k, tensor in enumerate(attr.tensors):
                    entries.append(
                        (
                            f"@node/{i}/{attr.name}/{k}",
                            tensor,
                            {"kind": "onnx_tensor", "path": ["graph", "node", i, "attribute", j, "tensors", k]},
                        )
                    )
    for name, tensor, locator in entries:
        if not name or name in tensors:
            raise WgtError(f"missing or duplicate ONNX tensor name: {name!r}")
        source_index = 0
        dtype = library.TensorProto.DataType.Name(tensor.data_type)
        if dtype == "UNDEFINED" or any(n < 0 for n in tensor.dims):
            raise WgtError(f"invalid ONNX tensor type/shape: {name}")
        bits = {
            "FLOAT": 32,
            "DOUBLE": 64,
            "FLOAT16": 16,
            "BFLOAT16": 16,
            "INT8": 8,
            "UINT8": 8,
            "BOOL": 8,
            "INT16": 16,
            "UINT16": 16,
            "INT32": 32,
            "UINT32": 32,
            "INT64": 64,
            "UINT64": 64,
            "COMPLEX64": 64,
            "COMPLEX128": 128,
            "INT4": 4,
            "UINT4": 4,
        }.get(dtype)
        if dtype.startswith("FLOAT8"):
            bits = 8
        elif dtype.startswith("FLOAT4"):
            bits = 4
        if bits is None and dtype != "STRING":
            raise WgtError(f"unsupported ONNX tensor encoding: {dtype}")
        expected_bytes = (math.prod(tensor.dims) * bits + 7) // 8 if bits else None
        if tensor.HasField("raw_data") and len(tensor.raw_data) != expected_bytes:
            raise WgtError(f"ONNX raw tensor byte size mismatch: {name}")
        if tensor.data_location == library.TensorProto.EXTERNAL:
            external = properties(tensor.external_data)
            location = external.get("location", "")
            if not location or Path(location).is_absolute() or ".." in Path(location).parts:
                raise WgtError("ONNX external data must be inside the model directory")
            external_path = (path.parent / location).resolve()
            if not external_path.is_relative_to(path.parent.resolve()):
                raise WgtError("ONNX external data escapes model directory")
            if external_path not in source_indices:
                source_indices[external_path] = len(sources)
                sources.append(source_record(external_path, "onnx_external"))
            source_index = source_indices[external_path]
            offset = int(external.get("offset", 0))
            length = int(external.get("length", sources[source_index]["size"] - offset))
            if offset < 0 or length < 0 or offset + length > sources[source_index]["size"]:
                raise WgtError(f"ONNX external range exceeds file: {name}")
            if length != expected_bytes:
                raise WgtError(f"ONNX external tensor byte size mismatch: {name}")
            locator = {"kind": "byte_range", "offset": offset, "length": length}
        else:
            library.checker.check_tensor(tensor)
        tensors[name] = {
            "source": source_index,
            "shape": list(tensor.dims),
            "dtype": "onnx." + dtype,
            "locator": locator,
            "metadata": properties(tensor.metadata_props),
        }
    config = {
        "graph_summary": {
            "name": model.graph.name,
            "inputs": [p.name for p in model.graph.input],
            "outputs": [p.name for p in model.graph.output],
            "nodes": [
                {
                    "name": n.name,
                    "op_type": n.op_type,
                    "domain": n.domain,
                    "inputs": list(n.input),
                    "outputs": list(n.output),
                }
                for n in model.graph.node
            ],
        }
    }
    return document(sources, tensors, config)


def numpy_file(path: Path):
    from numpy.lib import format as fmt

    def header(stream):
        version = fmt.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = fmt.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = fmt.read_array_header_2_0(stream)
        else:
            raise WgtError(f"unsupported NPY version: {version}")
        if dtype.hasobject or dtype.fields or dtype.kind not in "biufc":
            raise WgtError("NumPy object, structured and non-numeric arrays are unsupported")
        return list(shape), fortran, dtype, stream.tell()

    source = source_record(path, "numpy")
    tensors = {}
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            shape, fortran, dtype, offset = header(stream)
        length = math.prod(shape) * dtype.itemsize
        if offset + length != source["size"]:
            raise WgtError("NPY payload size mismatch")
        tensors[path.stem] = {
            "source": 0,
            "dtype": "numpy." + dtype.str,
            "shape": shape,
            "locator": {"kind": "byte_range", "offset": offset, "length": length, "order": "F" if fortran else "C"},
        }
    else:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if not entry.filename.endswith(".npy"):
                    raise WgtError(f"unsupported NPZ member: {entry.filename}")
                name = entry.filename[:-4]
                if name in tensors:
                    raise WgtError(f"duplicate NPZ tensor: {name}")
                with archive.open(entry) as stream:
                    shape, fortran, dtype, offset = header(stream)
                if offset + math.prod(shape) * dtype.itemsize != entry.file_size:
                    raise WgtError(f"NPZ member size mismatch: {name}")
                tensors[name] = {
                    "source": 0,
                    "dtype": "numpy." + dtype.str,
                    "shape": shape,
                    "locator": {"kind": "npz_member", "member": entry.filename, "order": "F" if fortran else "C"},
                }
    return document([source], tensors)


def read_local(path: Path):
    readers = {
        ".pt": pytorch,
        ".pth": pytorch,
        ".bin": pytorch,
        ".ckpt": pytorch,
        ".gguf": gguf,
        ".onnx": onnx,
        ".npy": numpy_file,
        ".npz": numpy_file,
    }
    try:
        return readers[path.suffix.lower()](path.resolve())
    except WgtError:
        raise
    except Exception as exc:
        raise WgtError(f"cannot read {path.suffix} checkpoint: {exc}") from None


def validate_v2(doc):
    if not doc["sources"]:
        raise WgtError("inventory has no sources")
    for source in doc["sources"]:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("uri"), str)
            or type(source.get("size")) is not int
            or source["size"] < 0
            or source.get("encoding") not in {"pytorch", "gguf", "onnx", "onnx_external", "numpy"}
            or source.get("payload_verified") is not False
            or not isinstance(source.get("metadata"), dict)
        ):
            raise WgtError("invalid inventory source")
        try:
            json_metadata(source["metadata"])
        except ValueError as exc:
            raise WgtError(f"invalid source metadata: {exc}") from None
    for name, tensor in doc["tensors"].items():
        index, shape, dtype, locator = (tensor.get(k) for k in ("source", "shape", "dtype", "locator"))
        if (
            type(index) is not int
            or not 0 <= index < len(doc["sources"])
            or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(locator, dict)
        ):
            raise WgtError(f"invalid tensor: {name}")
        source = doc["sources"][index]
        kind = locator.get("kind")
        if kind == "byte_range":
            offset, length = locator.get("offset"), locator.get("length")
            if (
                type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length < 0
                or offset + length > source["size"]
            ):
                raise WgtError(f"invalid byte range: {name}")
        elif kind in ("state_dict", "onnx_tensor"):
            path = locator.get("path")
            expected = "pytorch" if kind == "state_dict" else "onnx"
            if (
                source["encoding"] != expected
                or not isinstance(path, list)
                or not path
                or any(type(p) not in (str, int) for p in path)
            ):
                raise WgtError(f"invalid logical tensor reference: {name}")
        elif kind == "npz_member":
            if source["encoding"] != "numpy" or not isinstance(locator.get("member"), str):
                raise WgtError(f"invalid NPZ reference: {name}")
        else:
            raise WgtError(f"unsupported tensor reference: {name}")
    return doc


def read_collection(path: Path):
    from .decompile import _json

    config = None
    if path.is_dir():
        config_file = path / "config.json"
        if config_file.exists():
            config = _json(config_file.read_bytes())
        index = path / "pytorch_model.bin.index.json"
        if index.exists():
            path = index
        else:
            candidates = sorted(p for p in path.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
            if len(candidates) != 1:
                raise WgtError(
                    "model directory must contain exactly one supported checkpoint or a PyTorch shard index; select a file explicitly"
                )
            path = candidates[0]
    if not path.name.endswith(".bin.index.json"):
        doc = read_local(path)
        if config is not None:
            doc["config"] = config
        return doc
    index = _json(path.read_bytes())
    mapping = index.get("weight_map")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())
    ):
        raise WgtError("invalid PyTorch shard index")
    sources, tensors, actual = [], {}, {}
    for filename in sorted(set(mapping.values())):
        if Path(filename).name != filename or "\\" in filename or Path(filename).suffix not in (".bin", ".pt", ".pth"):
            raise WgtError("invalid PyTorch shard filename")
        shard = pytorch(path.parent / filename)
        for name, tensor in shard["tensors"].items():
            if name in tensors:
                raise WgtError(f"duplicate tensor across shards: {name}")
            tensors[name] = {**tensor, "source": len(sources)}
            actual[name] = filename
        sources.extend(shard["sources"])
    if actual != mapping:
        raise WgtError("PyTorch shard index does not match actual tensors")
    return document(sources, tensors, config)
