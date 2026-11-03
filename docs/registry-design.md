# Git package registry

The native client supports the default [WeightLang registry](https://github.com/WeightLang/registry).
This is a Git index of reviewed metadata; model assets stay in publisher GitHub
Releases. It is not a Cargo-compatible registry or a remote source-library loader.

## Consumer workflow

```sh
wgt add weightlang/gates
wgt add weightlang/gates --version v0.2.0
wgt update weightlang/gates
wgt update --all  # all dependencies, retaining each locked registry
wgt remove weightlang/gates
wgt fetch --locked
wgt fetch --offline --locked
```

Names are lowercase `owner/name`. The namespace must match the canonical module's
GitHub owner. `--registry OWNER/REPO` selects another Git index. No fallback to a
different registry is performed. The highest numeric non-yanked registered
version is selected by default; exact versions are also supported. All records
are stable numeric versions; legacy GitHub prerelease assets may be explicitly
registered by maintainers. GitHub release discovery for direct URLs still
excludes prereleases.

The manifest stores the canonical module path and exact version, preserving
existing Python consumers. Source weight imports still use that full path:

```wgt
import hidden from "github.com/WeightLang/WeightLang/packages/gates/xor_mlp" layer=hidden freeze
```

`wgt.sum` additionally records `registry`, `registry-commit`, `registry-name`,
and `registry-record` (SHA256 of the parsed record's canonical pretty JSON).
It also pins the source commit, release index and every artifact file as before.
A registry record seeds expected hashes **before** downloading artifacts.
Existing pins reject replacement contents. Re-adding the same version retains
its original registry commit. A name cannot silently switch module or registry.
Use the short name to update registered dependencies; direct URLs cannot bypass
that provenance. Builtin `import std/nn` does not use this registry.

The bare Git cache is under `<WEIGHTLANG_CACHE>/registries/<repository hash>`.
Online add/update fetches the default branch; offline resolution requires an
exact version and a cached index, plus already locked/cached artifacts. Build,
sync and exact artifact fetch use existing locks, without chasing index updates.
Yanking blocks new selections but preserves existing locked builds and downloads.

## Publisher workflow

```sh
wgt build -o weights
wgt package --version v0.1.0 --artifacts weights
# Push the exact source commit and package tag shown by packaging first.
wgt publish --index dist/wgt.index.json --name your-owner/model --license MIT --edition 0.3
```

`publish` creates a stable Release with `--latest=false`, checks remote artifacts,
then submits a registry PR using the authenticated `gh` account. The account must
have push permission on the source repository. Ordinary contributors get a fork;
registry writers also submit via PR. Local Git branches and working files are
not switched or committed by this command. Registration does not auto-merge.

`--name` defaults to `<source owner>/<last module component>`, lowercased.
`--license` is required unless `--no-register` is supplied; it is a publisher
claim for maintainer review, not an automatic license audit. `--edition` defaults
to 0.3 and declares the package's language compatibility. Use the actual edition
for older packages. Only current exact direct weight packages are supported.

- `--dry-run`: prepare/report actions without upload or PR; no registry network access.
- `--no-register`: upload only, preserving standalone Release workflows.
- `--register-only`: verify and submit an existing release after a partial failure;
  never overwrite its assets. Release upload and PR creation are separate phases.
- `--offline`: forbids real publication; `--locked` also forbids mutations.

Existing releases can be registered independently:

```sh
wgt register github.com/your-owner/repo/packages/model --name your-owner/model \
  --version v0.1.0 --license MIT --edition 0.3
```

`register --dry-run` verifies the remote release and prints the record without
creating a PR. Identical already-merged registrations return the existing record
URL; identical open submissions return their PR. Failed PR creation can be retried
using the deterministic submission branch. Different existing records are rejected.
A new fork may need time to become available; retry a failed submission.

## Index contract and review

One JSON file per version lives at `packages/<owner>/<name>/vX.Y.Z.json`:

```json
{
  "format_version": 1,
  "name": "owner/model",
  "module": "github.com/owner/models/model",
  "version": "v0.1.0",
  "source_commit": "<40 lowercase hex>",
  "index_sha256": "<64 lowercase hex>",
  "license": "MIT",
  "edition": "0.3",
  "yanked": false
}
```

The release index URL is derived from the canonical module and version; records
cannot redirect asset downloads to arbitrary hosts. Version records are immutable
except a one-way `yanked: false -> true` transition. Ownership is reviewed against
the source repository; namespace spelling alone is not authentication. The client
permission check is useful but cannot replace server-side maintainer review.

The index repository includes a validator and a read-only PR check which executes
the validator from the trusted base commit, checks record invariants and verifies
remote release hashes. It never runs publisher code. The Rust client additionally
validates safetensors layout, model contracts and origins. Maintainers review
ownership and license before merging. There are no publisher signatures, mirrors,
transitive model dependencies or version-range solving in this implementation.

Git keeps review and maintenance simple. A future sparse HTTP transport can reuse
these per-package records. Cargo also separates index metadata from downloads and
supports Git and sparse protocols: https://doc.rust-lang.org/cargo/reference/registry-index.html
