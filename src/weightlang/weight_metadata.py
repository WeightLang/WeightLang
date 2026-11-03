"""Read metadata or annotate a NEW safetensors file without changing tensor bytes."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from .decompile import MAX_HEADER, _header, _json, tensor_inventory
from .syntax import WgtError


def read_metadata(source: str):
    try:
        inventory = tensor_inventory(source)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WgtError(f"cannot read metadata: {exc}") from None
    return {
        "verified": False,
        "sources": [
            {
                "uri": record["uri"],
                "metadata": record.get("metadata", {}),
                "metadata_omitted": record.get("metadata_omitted", []),
            }
            for record in inventory["sources"]
        ],
    }


def write_metadata(source: str | Path, output: str | Path, updates: dict[str, str]):
    path, destination = Path(source), Path(output)
    if path.suffix != ".safetensors" or destination.suffix != ".safetensors":
        raise WgtError("metadata writing currently requires .safetensors input and output")
    if not updates or any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in updates.items()):
        raise WgtError("metadata updates must be a nonempty string-to-string mapping")
    if destination.exists() or destination.is_symlink():
        raise WgtError(f"output already exists: {destination}")
    created = False
    try:
        before = path.stat()
        _, record = _header(str(path), False)
        with path.open("rb") as original:
            length = struct.unpack("<Q", original.read(8))[0]
            raw = original.read(length)
            header = _json(raw)
            metadata = header.setdefault("__metadata__", {})
            metadata.update(updates)
            raw = json.dumps(header, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            raw += b" " * (-len(raw) % 8)
            if len(raw) > MAX_HEADER:
                raise WgtError("updated safetensors header exceeds 16 MiB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as target:
                created = True
                target.write(struct.pack("<Q", len(raw)))
                target.write(raw)
                shutil.copyfileobj(original, target, length=1024 * 1024)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise WgtError("source changed while copying")
        _header(str(destination), False)
        return {
            "output": str(destination),
            "metadata": metadata,
            "tensor_bytes_copied": record["size"] - record["data_start"],
            "source_modified": False,
        }
    except Exception as exc:
        if created:
            destination.unlink(missing_ok=True)
        if isinstance(exc, WgtError):
            raise
        raise WgtError(f"cannot write metadata: {exc}") from None
