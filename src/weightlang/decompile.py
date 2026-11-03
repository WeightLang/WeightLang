"""Conservative reverse import: executable architecture or explicit tensor inventory.

No unrestricted pickle loading, Python execution or inferred forward graph.
The external dialect references original bytes; it is not an inference program.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import urllib.request
from pathlib import Path
from urllib.parse import quote

from .syntax import WgtError, parse

HEADER = "wgt tensors 1"
MAX_HEADER = 16 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise WgtError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique)


def _get(url: str, limit: int = MAX_HEADER) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise WgtError("remote metadata exceeds size limit")
    return data


def _range(url: str, start: int, end: int) -> tuple[bytes, int]:
    # Distinct URLs prevent caches from reusing a response for a different Range.
    request = urllib.request.Request(f"{url}?wgt_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, end):
            raise WgtError("server did not honor the requested byte range; refusing full download")
        raw = response.read(end - start + 2)
        if len(raw) != end - start + 1:
            raise WgtError("truncated or oversized range response")
        return raw, int(match[3])


def _header(source: str, remote: bool) -> tuple[dict, dict]:
    if remote:
        prefix, size = _range(source, 0, 7)
    else:
        path = Path(source)
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
    if len(prefix) != 8:
        raise WgtError("truncated safetensors prefix")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= MAX_HEADER or length + 8 > size:
        raise WgtError("invalid or oversized safetensors header")
    if remote:
        raw, size2 = _range(source, 8, 7 + length)
        if size2 != size:
            raise WgtError("remote file size changed between requests")
    else:
        with Path(source).open("rb") as stream:
            stream.seek(8)
            raw = stream.read(length)
    header = _json(raw)
    if not isinstance(header, dict):
        raise WgtError("safetensors header must be an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise WgtError("invalid safetensors metadata")
    tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
    ranges = []
    for name, meta in tensors.items():
        if not isinstance(meta, dict):
            raise WgtError(f"invalid tensor metadata: {name}")
        shape, dtype, offsets = meta.get("shape"), meta.get("dtype"), meta.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(dtype, str)
            or dtype not in DTYPE_BYTES
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
        ):
            raise WgtError(f"invalid shape, offsets or unsupported dtype: {name}")
        start, end = offsets
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype] or end > size - length - 8:
            raise WgtError(f"tensor byte size mismatch: {name}")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise WgtError("safetensors contains overlapping tensors or gaps")
        cursor = end
    if cursor != size - length - 8:
        raise WgtError("safetensors contains unindexed trailing bytes")
    return tensors, {
        "uri": source,
        "size": size,
        "data_start": length + 8,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_verified": False,
        "metadata": metadata,
    }


def _safe_shard(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
        raise WgtError(f"invalid shard filename: {name!r}")
    return name


def tensor_inventory(source: str) -> dict:
    from .weight_formats import EXTENSIONS, read_collection

    if not source.startswith("hf://"):
        path = Path(source)
        if (
            path.suffix.lower() in EXTENSIONS
            or path.name.endswith(".bin.index.json")
            or (
                path.is_dir()
                and not list(path.glob("*.safetensors"))
                and not (path / "model.safetensors.index.json").exists()
            )
        ):
            return read_collection(path)
    config = None
    revision = None
    weight_map = None
    if source.startswith("hf://"):
        spec = source[5:]
        repo, sep, revision = spec.partition("@")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise WgtError("expected hf://owner/model[@revision]")
        revision = revision if sep else "main"
        info = _json(_get(f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}"))
        revision = info["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise WgtError("Hugging Face did not return an immutable commit")
        base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        available = {f["rfilename"] for f in info["siblings"]}
        if "config.json" in available:
            config = _json(_get(base + "config.json"))
        if "model.safetensors.index.json" in available:
            weight_map = _json(_get(base + "model.safetensors.index.json"))["weight_map"]
            names = sorted(set(weight_map.values()))
        elif "model.safetensors" in available:
            names = ["model.safetensors"]
        else:
            raise WgtError("repository has no standard safetensors checkpoint")
        files = [base + _safe_shard(name) for name in names]
        remote = True
    else:
        path = Path(source).resolve()
        if path.is_dir():
            config_path = path / "config.json"
            if config_path.exists():
                config = _json(config_path.read_bytes())
            index = path / "model.safetensors.index.json"
            if index.exists():
                path = index
            else:
                files = sorted(str(p) for p in path.glob("*.safetensors"))
        if path.name.endswith(".safetensors.index.json"):
            weight_map = _json(path.read_bytes())["weight_map"]
            files = [str(path.parent / _safe_shard(n)) for n in sorted(set(weight_map.values()))]
        elif path.is_file() and path.suffix == ".safetensors":
            files = [str(path)]
        elif not path.is_dir():
            raise WgtError("expected .safetensors, shard index, model directory, model.py or hf://owner/model")
        remote = False
    if not files:
        raise WgtError("no safetensors weights found")
    sources, tensors, actual_map = [], {}, {}
    for number, file in enumerate(files):
        header, metadata = _header(file, remote)
        sources.append(metadata)
        for name, meta in header.items():
            if name in tensors:
                raise WgtError(f"duplicate tensor across shards: {name}")
            tensors[name] = {**meta, "source": number}
            actual_map[name] = file.rsplit("/", 1)[-1]
    if weight_map is not None and actual_map != weight_map:
        raise WgtError("shard index does not match actual tensor names and locations")
    return {
        "format": "weightlang.tensor_inventory",
        "version": 1,
        "executable": False,
        "revision": revision,
        "config": config,
        "sources": sources,
        "tensors": tensors,
        "limitations": [
            "Tensor bytes remain in referenced files; payloads were not verified.",
            "Weights do not determine a forward graph. Native execution is unavailable.",
        ],
    }


def render_inventory(document: dict) -> str:
    lines = [f"wgt tensors {document['version']}", "# External tensor references; NOT an executable WeightLang model."]
    for key in ("format", "version", "executable", "revision", "config", "limitations", "sources"):
        lines.append(f"{key} " + json.dumps(document[key], ensure_ascii=False, sort_keys=True))
    for name, meta in sorted(document["tensors"].items()):
        lines.append("tensor " + json.dumps({"name": name, **meta}, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


def render_architecture(document: dict) -> str:
    """Describe shapes only; group identical numbered modules without implying sharing."""
    modules = {}
    standalone = {}
    for name, meta in sorted(document["tensors"].items()):
        spec = {"shape": meta["shape"], "dtype": meta["dtype"]}
        match = re.fullmatch(r"(.+)\.([0-9]+)\.(.+)", name)
        if match:
            prefix, index, suffix = match.groups()
            modules.setdefault(prefix, {}).setdefault(index, {})[suffix] = spec
        else:
            standalone[name] = spec
    groups = []
    for prefix, instances in modules.items():
        schemas = {}
        for index, members in instances.items():
            key = json.dumps(members, sort_keys=True)
            schemas.setdefault(key, []).append(index)
        for schema, indices in schemas.items():
            groups.append({"module": prefix + ".{i}",
                           "indices": sorted(indices, key=lambda i: (int(i), i)),
                           "members": json.loads(schema)})
    description = {
        "executable": False,
        "config": document["config"],
        "modules": groups,
        "tensors": standalone,
    }
    return ("wgt architecture 1\n"
            "# Structure only: no weight values, locations or checkpoint imports.\n"
            "# Each index is a separate module; equal shapes do not imply shared weights.\n"
            "# Tensor shapes/config do not recover a complete executable forward graph.\n"
            + json.dumps(description, ensure_ascii=False, indent=2) + "\n")


def parse_architecture(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0] != "wgt architecture 1":
        raise WgtError("expected 'wgt architecture 1'")
    doc = _json("\n".join(line for line in lines[1:] if not line.startswith("#")))
    if (not isinstance(doc, dict) or set(doc) != {"executable", "config", "modules", "tensors"}
            or doc["executable"] is not False or not isinstance(doc["modules"], list)
            or not isinstance(doc["tensors"], dict)):
        raise WgtError("invalid architecture description")
    names = set()

    def member(name, spec):
        if (not isinstance(name, str) or name in names or not isinstance(spec, dict)
                or set(spec) != {"shape", "dtype"} or not isinstance(spec["shape"], list)
                or any(type(n) is not int or n < 0 for n in spec["shape"])
                or not isinstance(spec["dtype"], str) or not spec["dtype"]):
            raise WgtError("invalid or duplicate architecture member")
        names.add(name)

    for name, spec in doc["tensors"].items():
        member(name, spec)
    for group in doc["modules"]:
        if (not isinstance(group, dict) or set(group) != {"module", "indices", "members"}
                or not isinstance(group["module"], str) or group["module"].count("{i}") != 1
                or not isinstance(group["indices"], list) or not group["indices"]
                or not isinstance(group["members"], dict) or not group["members"]):
            raise WgtError("invalid architecture module")
        for index in group["indices"]:
            if not isinstance(index, str) or not re.fullmatch(r"[0-9]+", index):
                raise WgtError("invalid architecture module index")
            for suffix, spec in group["members"].items():
                member(group["module"].replace("{i}", index) + "." + suffix, spec)
    return doc


def parse_inventory(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0] not in (HEADER, "wgt tensors 2"):
        raise WgtError("expected 'wgt tensors 1' or 'wgt tensors 2'")
    doc = {"tensors": {}}
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        key, _, raw = line.partition(" ")
        value = _json(raw)
        if key == "tensor":
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                raise WgtError("invalid tensor record")
            name = value.pop("name")
            if name in doc["tensors"]:
                raise WgtError(f"duplicate tensor: {name}")
            doc["tensors"][name] = value
        elif (
            key in {"format", "version", "executable", "revision", "config", "limitations", "sources"}
            and key not in doc
        ):
            doc[key] = value
        else:
            raise WgtError(f"unknown or duplicate inventory field: {key}")
    if (
        doc.get("format") != "weightlang.tensor_inventory"
        or type(doc.get("version")) is not int
        or doc["version"] not in (1, 2)
        or lines[0] != f"wgt tensors {doc['version']}"
        or doc.get("executable") is not False
        or not isinstance(doc.get("sources"), list)
    ):
        raise WgtError("invalid tensor inventory metadata")
    if set(doc) != {"format", "version", "executable", "revision", "config", "limitations", "sources", "tensors"}:
        raise WgtError("missing tensor inventory fields")
    if doc["version"] == 2:
        from .weight_formats import validate_v2

        return validate_v2(doc)
    ranges = [[] for _ in doc["sources"]]
    for source in doc["sources"]:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("uri"), str)
            or type(source.get("size")) is not int
            or type(source.get("data_start")) is not int
            or not 10 <= source["data_start"] <= source["size"]
            or source.get("payload_verified") is not False
            or not isinstance(source.get("header_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", source["header_sha256"])
        ):
            raise WgtError("invalid inventory source metadata")
        if "metadata" in source and (
            not isinstance(source["metadata"], dict) or any(not isinstance(v, str) for v in source["metadata"].values())
        ):
            raise WgtError("invalid safetensors metadata")
    for name, meta in doc["tensors"].items():
        if (
            type(meta.get("source")) is not int
            or not 0 <= meta["source"] < len(doc["sources"])
            or not isinstance(meta.get("shape"), list)
            or any(type(n) is not int or n < 0 for n in meta["shape"])
            or not isinstance(meta.get("dtype"), str)
            or meta["dtype"] not in DTYPE_BYTES
        ):
            raise WgtError(f"invalid tensor record: {name}")
        offsets = meta.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] - offsets[0] != math.prod(meta["shape"]) * DTYPE_BYTES[meta["dtype"]]
        ):
            raise WgtError(f"invalid tensor offsets: {name}")
        ranges[meta["source"]].append(tuple(offsets))
    for source, spans in zip(doc["sources"], ranges):
        cursor = 0
        for start, end in sorted(spans):
            if cursor != start:
                raise WgtError("inventory contains overlapping tensors or gaps")
            cursor = end
        if cursor != source["size"] - source["data_start"]:
            raise WgtError("inventory byte ranges do not cover the source payload")
    return doc


def python_sequential(path: Path) -> str:
    """Compatibility entry point for static Python architecture conversion."""
    from .python_import import python_architecture

    return python_architecture(path)


def artifact_source(path: Path, *, references: bool = False) -> str:
    """Reconstruct architecture; optionally pin parameters to the checkpoint."""
    from .artifact import load_artifact

    if references:
        manifest = load_artifact(path, warn_untrained=False).manifest
    else:
        manifest = _json((path / "manifest.json").read_bytes())
    return source_from_manifest(manifest, path=path, references=references)


def source_from_manifest(manifest, *, path=Path("."), references=False):
    blocks = []
    counter = 0

    def layer_line(layer):
        nonlocal counter
        op = layer["op"]
        if op == "route":
            branches = []
            for child in layer["layers"]:
                recovered = layer_line(child)
                branches.append(recovered.split(": ", 1)[1])
            expr = "route(" + ", ".join(branches) + ")"
        elif op in ("seq", "residual", "scan", "recur", "scan_state", "ponder"):
            name = f"recovered_block_{counter}"
            counter += 1
            body = ["  " + layer_line(child) for child in layer["layers"]]
            if layer.get('returns'):
                body.append('  return '+' '.join(f'{field}={ref}' for field,ref in layer['returns'].items()))
            blocks.append(f"block {name}():\n" + "\n".join(body))
            if op == "scan_state":
                expr = f"scan_state({name}())"
            elif op == "scan":
                expr = f"scan(hidden={layer['args']['hidden']}, {name}())"
            elif op in {"recur", "ponder"}:
                stop = f", until={layer['args']['until']}" if "until" in layer["args"] else ""
                compact = f", compact={layer['args']['compact']}" if "compact" in layer["args"] else ""
                expr = f"{op}({layer['args']['steps']}, {name}(){stop}{compact})"
            else:
                expr = f"{name}()" if op == "seq" else f"residual({name}())"
        elif op in ("rearrange", "reduce"):
            arguments = [json.dumps(layer["args"]["pattern"])]
            arguments.extend(f"{key}={value}" for key, value in layer["args"].get("sizes", {}).items())
            if op == "reduce":
                arguments.append(f"op={layer['args']['reduction']}")
            expr = f"{op}({', '.join(arguments)})"
        else:
            args = ", ".join(f"{key}={value}" for key, value in layer["args"].items())
            expr = f"{op}({args})" if args else op
        refs = layer.get("inputs", [])
        if op in {"add", "mul", "concat", "mix", "where"}:
            expr = f"{op}({', '.join(refs)})"
        elif refs:
            expr += " from " + ", ".join(refs)
        return f"layer {layer['name']}: {expr}"

    layers = [layer_line(layer) for layer in manifest["layers"]]
    layers += [f"parameter {name} = {path}" for name,path in manifest['interface'].get('parameter_ids',{}).items()]
    layers += [f"wtype {name} = {values!r}" for name, values in manifest['interface'].get('weight_types', {}).items()]
    layers += [f"weight {spec['target']}: {spec['type']}" + ''.join(f" {key}={spec[key]}" for key in ('axis','start','stop') if key in spec) for spec in manifest['interface'].get('weight_assignments', [])]
    layers += [f"share {path} = {name}" for path,name in manifest['interface'].get('parameter_shares',{}).items()]
    layers += [f"adapter {name}: {spec.get('method', 'lora')} " + ' '.join(f'{k}={v}' for k,v in spec.items() if k != 'method') for name,spec in manifest['interface'].get('adapters',{}).items()]
    lines = [
        "wgt 0.3",
        "# Reconstructed architecture; " + ("weights inherited exactly." if references else "new weights initialize on build; no trained values."),
        *blocks,
        f"model {manifest['model']}",
    ]
    from .symbolic import display
    for name, bounds in manifest["interface"].get("dimensions", {}).items():
        lines.append(f"dim {name} min={bounds['min']} max={bounds['max']}")
    for op, left, right in manifest["interface"].get("constraints", []):
        lines.append(f"require {display(left)} {op} {display(right)}")
    for direction in ("input", "output"):
        ports = manifest["interface"].get(direction + "s", [manifest["interface"][direction]])
        for port in ports:
            axes = port.get("axes", [None] * len(port["shape"]))
            shape = ",".join(f"{axis}={size}" if axis else str(size) for axis, size in zip(axes, [display(e) for e in port.get("shape_expr", port["shape"])]))
            suffix = f" from {port['from']}" if direction == "output" and "from" in port else ""
            lines.append(f"{direction} {port['name']}: {port['dtype']}[{shape}]{suffix}")
    location = str(path.resolve())
    if any(c in location for c in ('"', "\\", "\n", "\r")):
        raise WgtError("artifact path cannot be represented in a WeightLang string")
    lines.extend([*layers, "init seed=0"])
    for layer in manifest["layers"]:
        name = layer["name"]
        if references and any(key.startswith(name + ".") for key in manifest["weights"]["tensors"]):
            lines.append(f'import {name} from "{location}" sha256={manifest["weights"]["sha256"]}')
    text = "\n".join(lines) + "\n"
    parse(text)
    return text


def decompile(source: str, output: str | Path, *, references: bool = False,
              class_name: str | None = None, init_args: dict | None = None,
              input_shape: tuple[int, ...] | None = None, input_dtype: str = "f32") -> dict:
    """Write a NEW source. Existing files and model directories are never replaced."""
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise WgtError(f"output already exists: {destination}")
    try:
        path = Path(source)
        if (class_name is not None or init_args is not None or input_shape is not None or input_dtype != "f32") and (path.suffix != ".py" or source.startswith("hf://")):
            raise WgtError("Python conversion options require a .py source")
        if path.suffix == ".py" and not source.startswith("hf://"):
            from .python_import import python_architecture

            text = python_architecture(path, class_name=class_name, init_args=init_args,
                                       input_shape=input_shape, input_dtype=input_dtype)
            report = {"kind": "architecture", "executable": True, "trained_weights": False}
        elif path.is_dir() and (path / "manifest.json").exists():
            text = artifact_source(path, references=references)
            report = {"kind": "checkpoint" if references else "architecture", "executable": True,
                      "trained_weights": False if not references else None}
        else:
            if path.is_file() and path.suffix == ".wgt":
                document = parse_inventory(path.read_text(encoding="utf-8"))
            else:
                document = tensor_inventory(source)
            text = render_inventory(document) if references else render_architecture(document)
            (parse_inventory if references else parse_architecture)(text)
            report = {
                "kind": "tensor_inventory" if references else "architecture",
                "executable": False,
                "tensors": len(document["tensors"]),
                "parameters": sum(math.prod(t["shape"]) for t in document["tensors"].values()),
                "payload_verified": False,
                "revision": document["revision"],
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(text)
        return {"output": str(destination), **report}
    except WgtError:
        raise
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
        raise WgtError(f"cannot decompile: {exc}") from None
