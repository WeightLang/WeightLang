# Package distribution (design only — not implemented)

Status: proposal. `weightc` 0.1 only imports from local artifact directories.
Nothing here is implemented, and no network access happens during builds.

## Goals

- Identify packages by GitHub repository path, as Go does.
- Builds are reproducible: a version resolves to the same bytes forever.
- Every downloaded byte is checked against a recorded hash before use.
- Large weight files do not have to live in Git history.
- Loading third-party weights never executes third-party code.

## Package identity and versions

A package is a repository (optionally a subdirectory) with a `wgt.mod` at its root:

```
module github.com/acme/logic-gates
wgt 0.2

require github.com/acme/features v1.4.0
```

Versions are semantic-version Git tags (`v1.4.0`). As in Go, a breaking major
version changes the path (`github.com/acme/features/v2`), and dependency
selection uses minimal version selection: the build uses the highest *required*
version of each module, never "latest".

Source would reference artifacts by package path plus artifact name:

```wgt
import hidden from "github.com/acme/logic-gates/xor_mlp" layer=hidden freeze
```

The version comes from `wgt.mod`, not from the import line, so one file cannot
silently mix versions of the same package.

## Lock file and integrity

`weightc` writes `wgt.sum`, one line per fetched item:

```
github.com/acme/logic-gates v1.2.0 commit 3f9c...e1        # tag resolved to commit
github.com/acme/logic-gates v1.2.0 xor_mlp/manifest.json sha256:94ab...
github.com/acme/logic-gates v1.2.0 xor_mlp/weights.safetensors sha256:7969...
```

- Tags can be moved; the commit SHA recorded on first resolution is
  authoritative. A tag that later points elsewhere is a hard error.
- Each artifact's manifest and weights are pinned by sha256 (the same hash the
  0.1 `import ... sha256=` option already verifies). Byte-stable artifact
  writing (see `artifact.md`) is a prerequisite and is already in place.
- `wgt.sum` is committed. A later checksum database or transparency log (like
  `sum.golang.org`) can detect a publisher serving different bytes to
  different users; it is out of scope until the basic flow works.

## Large files

Git stores source and small files: `.wgt`, `wgt.mod`, published `manifest.json`.
Weights are referenced by hash and URL from a published manifest:

```json
"weights": {
  "file": "weights.safetensors",
  "sha256": "7969...",
  "size": 476,
  "urls": ["https://github.com/acme/logic-gates/releases/download/v1.2.0/xor_mlp.safetensors"]
}
```

- Default host: GitHub Release assets attached to the version tag (per-asset
  size limit applies; very large models would need sharding or another host).
- Mirrors may be listed; any URL is acceptable because the hash, not the host,
  decides validity.
- Git LFS is allowed but not required (bandwidth quotas, and LFS pointers in
  Git history do not by themselves give the lock-file guarantees above).
- Local cache is content-addressed: `~/.cache/weightlang/sha256/<hash>`.
  Identical weights shared by many packages are stored once.

## Safety and compatibility

- Only `safetensors` weights and the declarative manifest are accepted; no
  pickle, no custom Python layers from packages.
- Imports check layer signatures (op and sizes) before copying, as 0.1 already does.
- Published manifests must carry a license field; `weightc` reports the license
  of every imported tensor. Policy (allow/deny lists) is a later decision.
- Provenance (`origin`, `upstream_origin`) travels with imported tensors, so a
  package cannot launder random weights as trained ones.

## Planned commands

`weightc mod init <path>`, `weightc get <path>@<version>`, `weightc mod verify`,
`weightc publish` (build, hash, produce the release asset list). Fetching is an
explicit step; `weightc build` would only read the cache and fail if something
is missing or does not match `wgt.sum`.

## Open questions

- Whether artifacts are published prebuilt or rebuilt from source by consumers
  (rebuild is reproducible only on matching PyTorch/hardware).
- Signing of release assets in addition to hashes.
- Private repositories and authentication.
- How datasets, which may be large or restricted, are referenced and hashed.
