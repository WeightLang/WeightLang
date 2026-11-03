> **历史接口说明 / Historical interfaces:** 当前源码已清退全部 Python 实现和测试。本文中的 Python API、`.py` 脚本和旧 CLI 示例已停用；仅 Rust 实现部分继续有效。参见[原生入口与迁移边界](../weightc-migration.md)。

# WeightLang Revision Store v1

Status: implemented, experimental interoperability contract. MUST/MUST NOT are
normative. Artifact formats 1–3 remain unchanged. No lossy encodings are permitted.

## Layout and identities

```
STORE/
  objects/<sha256>       raw byte objects, 1..65536 bytes
  revisions/<sha256>     canonical UTF-8 JSON revision records
```

IDs MUST be 64 lowercase hexadecimal SHA-256 digests of the exact file bytes.
Readers MUST verify object and revision hashes before using them. Object filenames
have no suffix. Stored object files MUST NOT be symlinks. Identical raw bytes
share an object regardless of their source tensor or filename. Writers MUST NOT
replace an existing addressed object; publication uses an exclusive atomic link
from a fully written temporary file. Temporary `.pending-*` files are not objects.
A failed save may leave unreferenced valid objects; v1 does not garbage-collect.
No durability guarantee against sudden power loss of directory entries is made.

## Revision record

See [JSON Schema](revision-v1.schema.json). Example structure (IDs abbreviated):

```json
{
  "format": "weightlang.revision",
  "format_version": 1,
  "precision": "lossless",
  "parent": "<revision sha256 or null>",
  "files": {
    "manifest.json": {"sha256": "...", "size": 123, "objects": [{"sha256": "...", "size": 123}]},
    "source.wgt": {"sha256": "...", "size": 42, "objects": [{"sha256": "...", "size": 42}]},
    "weights.safetensors": {"sha256": "...", "size": 456, "objects": [{"sha256": "...", "size": 456}]}
  }
}
```

A record MUST use exactly the top-level and per-file/object fields in the schema.
JSON canonicalization: recursively lexicographically sorted keys, no whitespace
between tokens, ASCII escaping of non-ASCII characters (Python `ensure_ascii=True`
conventions), no NaN/Infinity, and no trailing newline. Strings in v1's record are
ASCII; integers are nonnegative decimal without leading zeroes. Thus no floating
point or Unicode normalization convention is needed. The revision ID is SHA-256
of this encoding. No timestamps are included; saving the same bytes with the same
parent produces the same ID. Changing only the parent changes the revision ID.

`files` MUST contain exactly `manifest.json`, `source.wgt`, and
`weights.safetensors`. Each file is the concatenation of its ordered object bytes.
Object sizes MUST match actual bytes; their sum MUST equal file size; concatenation
MUST match the file SHA-256. All reconstructed files MUST pass the standard
artifact hash, architecture, provenance, dtype and tensor validation. Original
header padding, tensor bytes, source and manifest formatting MUST be preserved.

## Chunking and sharing

Writers split source and manifest into consecutive chunks of at most 65536 bytes.
For safetensors, treat the 8-byte length plus JSON header as one segment, and
each tensor's contiguous data as a subsequent segment in file-offset order.
Split each segment independently into consecutive chunks of at most 65536 bytes.
Do not emit empty objects. Readers need only the ordered object references;
they MUST NOT depend on recomputing segmentation. Headers can change independently
without shifting the chunks of unchanged tensors. Changed chunks are stored as
new bytes, not arithmetic floating-point deltas. This guarantees bitwise recovery,
including signed zero. It does not guarantee savings when most parameters change.

## Parent, verification and export

A parent is null or one existing revision ID in the same store. Writers MUST check
the parent's record hash/format before publishing a child. Parent is historical
metadata, not a reconstruction dependency, and does not imply weight inheritance
or architecture compatibility. Each record is a complete file/object reference
tree: loading a child MUST NOT recursively replay its ancestors. A child remains
loadable if its parent record is unavailable but its own objects are retained.
Preserving ancestry for audit requires retaining parent records separately.

Readers MUST reject unknown format versions or precision modes, malformed IDs,
missing/tampered objects, invalid sizes, and mismatched reconstructed file hashes.
Export MUST refuse existing destinations, verify the complete artifact before
publishing it, and MUST NOT modify the store. Interrupted export can leave an
incomplete destination; normal artifact verification MUST reject it. There are
no named mutable branches, automatic cleanup, signatures, encryption, network
resolution, compression or optimizer states in v1. Hashes prove integrity, not
publisher identity. A portable artifact export is the interoperation boundary
for existing Python/Rust readers and package publishers.
