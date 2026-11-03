"""WeightLang revision store v1: immutable, lossless content-addressed snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

from .artifact import load_artifact, sha256_file
from .syntax import WgtError

FORMAT = "weightlang.revision"
CHUNK_SIZE = 65536
FILES = ("manifest.json", "source.wgt", "weights.safetensors")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise WgtError("revision/object ID must be 64 lowercase hex characters")
    return value


class RevisionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, kind, digest):
        return self.root / kind / _id(digest)

    def _read(self, kind, digest):
        path = self._path(kind, digest)
        if path.is_symlink():
            raise WgtError(f"symlink is not a store object: {path}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise WgtError(f"missing/unreadable {kind} {digest}: {exc}") from None
        if _digest(raw) != digest:
            raise WgtError(f"{kind} sha256 mismatch: {digest}")
        return raw

    def _put(self, kind, raw):
        digest = _digest(raw)
        path = self._path(kind, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            if self._read(kind, digest) != raw:
                raise WgtError("content address collision")
            return digest, 0
        fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(tmp, path)  # publish atomically, never replace another writer
                added = len(raw)
            except FileExistsError:
                self._read(kind, digest)
                added = 0
        finally:
            Path(tmp).unlink(missing_ok=True)
        return digest, added

    def read(self, revision):
        try:
            raw = self._read("revisions", revision)
            record = json.loads(raw)
            if not isinstance(record, dict) or set(record) != {
                "format",
                "format_version",
                "precision",
                "parent",
                "files",
            }:
                raise ValueError("unexpected revision fields")
            if type(record["format_version"]) is not int:
                raise ValueError("invalid format version")
            if (
                record["format"] != FORMAT
                or record["format_version"] != 1
                or record["precision"] != "lossless"
                or raw != _canonical(record)
            ):
                raise ValueError("unsupported/noncanonical revision")
            if record["parent"] is not None:
                _id(record["parent"])
            if not isinstance(record["files"], dict) or set(record["files"]) != set(FILES):
                raise ValueError("revision requires exactly the three artifact files")
            for item in record["files"].values():
                if not isinstance(item, dict) or set(item) != {"sha256", "size", "objects"}:
                    raise ValueError("unexpected file fields")
                if not isinstance(item["objects"], list):
                    raise ValueError("objects must be an array")
                _id(item["sha256"])
                if type(item["size"]) is not int or item["size"] < 0:
                    raise ValueError("invalid file size")
                total = 0
                for obj in item["objects"]:
                    if not isinstance(obj, dict) or set(obj) != {"sha256", "size"}:
                        raise ValueError("unexpected object fields")
                    _id(obj["sha256"])
                    if type(obj["size"]) is not int or not 0 < obj["size"] <= CHUNK_SIZE:
                        raise ValueError("invalid object size")
                    total += obj["size"]
                if total != item["size"]:
                    raise ValueError("object sizes do not cover file")
            return record
        except (KeyError, TypeError, ValueError) as exc:
            raise WgtError(f"invalid revision: {exc}") from None

    def save(self, artifact, *, parent=None):
        if parent is not None:
            self.read(parent)
        root = Path(artifact)
        model = load_artifact(root, warn_untrained=False)
        if (
            model.manifest["weights"]["file"] != "weights.safetensors"
            or model.manifest["source"]["file"] != "source.wgt"
        ):
            raise WgtError("revision v1 requires standard artifact filenames")
        before = {name: sha256_file(root / name) for name in FILES}
        files, added = {}, 0
        for name in FILES:
            path = root / name
            size = path.stat().st_size
            boundaries = [size]
            if name == "weights.safetensors":
                with path.open("rb") as stream:
                    length = struct.unpack("<Q", stream.read(8))[0]
                    if length > 16 * 1024 * 1024 or length + 8 > size:
                        raise WgtError("invalid safetensors header length")
                    header = json.loads(stream.read(length))
                # Reset chunk boundaries at tensor boundaries, preserving reuse
                # when a preceding tensor or the header changes size.
                boundaries = [8 + length] + sorted(
                    8 + length + value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"
                )
                if boundaries[-1] != size:
                    raise WgtError("safetensors file is not fully covered")
            objects, cursor = [], 0
            with path.open("rb") as stream:
                for end in boundaries:
                    while cursor < end:
                        raw = stream.read(min(CHUNK_SIZE, end - cursor))
                        if not raw:
                            raise WgtError("artifact changed while saving")
                        digest, count = self._put("objects", raw)
                        added += count
                        objects.append({"sha256": digest, "size": len(raw)})
                        cursor += len(raw)
            files[name] = {"sha256": before[name], "size": size, "objects": objects}
        if any(sha256_file(root / name) != before[name] for name in FILES):
            raise WgtError("artifact changed while saving")
        record = {"format": FORMAT, "format_version": 1, "precision": "lossless", "parent": parent, "files": files}
        # Verify the bytes actually captured, not only the current source files.
        self._verify_objects(record)
        revision, count = self._put("revisions", _canonical(record))
        return {
            "revision": revision,
            "parent": parent,
            "new_bytes": added + count,
            "artifact_bytes": sum(item["size"] for item in files.values()),
        }

    def _verify_objects(self, record, output=None):
        for name, item in record["files"].items():
            digest = hashlib.sha256()
            stream = (output / name).open("xb") if output is not None else None
            try:
                for obj in item["objects"]:
                    raw = self._read("objects", obj["sha256"])
                    if len(raw) != obj["size"]:
                        raise WgtError("object size mismatch")
                    digest.update(raw)
                    if stream is not None:
                        stream.write(raw)
            finally:
                if stream is not None:
                    stream.close()
            if digest.hexdigest() != item["sha256"]:
                raise WgtError(f"reconstructed {name} sha256 mismatch")

    def checkout(self, revision, output):
        record = self.read(revision)
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise WgtError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".checkout-", dir=output.parent) as tmp:
            stage = Path(tmp) / "model"
            stage.mkdir()
            self._verify_objects(record, stage)
            load_artifact(stage, warn_untrained=False)
            # Reserve the destination rather than replacing an existing model.
            output.mkdir()
            try:
                for name in FILES:
                    shutil.move(str(stage / name), output / name)
            except BaseException:
                shutil.rmtree(output)
                raise
        return {"revision": revision, "artifact": str(output)}

    def verify(self, revision):
        # Disk-backed scratch in the store, never put potentially large models in /tmp.
        with tempfile.TemporaryDirectory(prefix=".verify-", dir=self.root) as tmp:
            self.checkout(revision, Path(tmp) / "model")
        return {"revision": revision, "verified": True}
