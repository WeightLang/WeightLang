# Packages

The native Rust **wgt** CLI now offers Cargo-like project and package commands.
See [wgt](wgt.md). It uses the formats below and keeps existing weightc consumers
compatible. The new executable is on main, not in the published 0.2.0 wheel.

Status: a first, deliberately small version works end to end — publishing a
package as a GitHub release, fetching it with hash checks, and building
offline against it. Items under [Not implemented yet](#not-implemented-yet) are
still design.

The first published package is `github.com/WeightLang/WeightLang/packages/gates`
(`packages/gates/` in this repository, tag `packages/gates/v0.1.0`), consumed by
`examples/remote_reuse/`.

## Latest stable selection (native CLI)

`wgt add github.com/WeightLang/WeightLang/packages/gates` selects the highest
stable published package version, verifies it, then writes an exact version and
hashes. `--version latest` is equivalent; `--version v0.1.0` selects explicitly.
`wgt update MODULE` explicitly refreshes that pin. Builds never follow latest.
Only releases matching the package subdirectory are considered; draft and
prerelease releases are excluded. Offline discovery is rejected. Existing exact
pins, cache verification and `wgt.sum` immutability checks remain in force.
See [native CLI](wgt.md) for options and [Git registry](registry-design.md)
for `wgt add owner/name` and `wgt publish` registration PRs.

Standard libraries are bundled: `import std/nn` (wgt 0.3) needs neither `add`
nor a manifest dependency. This imports computation definitions, not weights.

## Package identity and versions

Terminology: a **weight package** distributes model artifacts; a source **library**
contains reusable `block` declarations. The historical `module` field identifies
the package path, not a source file. See [project hierarchy](project-layout.md).

A package is a directory with a `wgt.mod` inside a GitHub repository:

```
module github.com/WeightLang/WeightLang/packages/gates
wgt 0.1
```

The module path is `github.com/<owner>/<repo>[/<subdir>]` and must match the
package's location in the repository. Versions are Git tags
`v<major>.<minor>.<patch>`; for a package in a subdirectory the tag is prefixed
with the subdirectory (`packages/gates/v0.1.0`), as Go does.

A consumer lists requirements in its own `wgt.mod`:

```
module github.com/WeightLang/WeightLang/examples/remote_reuse
wgt 0.1

require github.com/WeightLang/WeightLang/packages/gates v0.1.0
```

and imports `<module>/<artifact>`:

```wgt
import hidden from "github.com/WeightLang/WeightLang/packages/gates/xor_mlp" layer=hidden freeze
```

The version comes only from `wgt.mod`, so one program cannot mix versions of a
package. Import paths starting with `./`, `../` or `/` are local artifact
directories; anything else is a package path.

## Publishing

```sh
weightc publish packages/gates --version v0.1.0 -o dist
```

`publish` checks that the package directory matches the module path and has no
uncommitted changes, records `HEAD`, builds every `.wgt` in the package, refuses
artifacts with untrained tensors unless `--allow-untrained` is given, and writes
to `dist/`:

- `<artifact>.manifest.json`, `<artifact>.weights.safetensors`, `<artifact>.source.wgt`
- `wgt.index.json`: module, version, tag, source commit, and for each artifact
  its source file and the sha256 and size of each asset.

It then prints the commands to tag that commit and upload the assets as a GitHub
release (`git tag …`, `gh release create … --verify-tag dist/*`). Weights live in
release assets, not in Git history.

## Fetching and the lock file

```sh
weightc get examples/remote_reuse
```

For each `require`:

1. Resolve the tag to a commit with `git ls-remote` (annotated tags are peeled).
2. Download `wgt.index.json` from the release; it must name the same module,
   version and commit.
3. Download every asset, rejecting any whose size or sha256 differs from the index.
4. Download the artifact's `.wgt` from `raw.githubusercontent.com` **at that
   commit** and require it to be byte-identical to `source.wgt`. This ties the
   prebuilt weights to the tagged source; it does not prove the weights were
   produced by that source (see below).
5. Run the full artifact verification (`load_artifact`) and move the result into
   the cache: `$WEIGHTLANG_CACHE` or `~/.cache/weightlang/pkg/<module>@<version>/<artifact>/`.
6. Write `wgt.sum`:

```
github.com/WeightLang/WeightLang/packages/gates v0.1.0 commit <40-hex sha>
github.com/WeightLang/WeightLang/packages/gates v0.1.0 wgt.index.json sha256:<hex>
github.com/WeightLang/WeightLang/packages/gates v0.1.0 xor_mlp/manifest.json sha256:<hex>
github.com/WeightLang/WeightLang/packages/gates v0.1.0 xor_mlp/source.wgt sha256:<hex>
github.com/WeightLang/WeightLang/packages/gates v0.1.0 xor_mlp/weights.safetensors sha256:<hex>
```

Commit `wgt.sum`. If it already exists, every value fetched later must equal it:
a moved tag or a replaced release asset is a hard error, not an update. The first
fetch is trust-on-first-use.

`weightc build` never uses the network. A package import is resolved through
`wgt.mod`, must have entries in `wgt.sum`, and every cached file is re-hashed
against `wgt.sum` before loading. The built artifact's manifest records the
module, version and commit in `imports[].package`.

## Safety

- Only safetensors weights and declarative manifests are loaded; no pickle, no
  package-provided Python.
- Imports check layer op and sizes before copying.
- Provenance (`origin`, `upstream_origin`) travels with imported tensors, so
  random weights cannot be passed off as trained through a package.
- Downloads are size-limited by the index before hashing.

## Known gaps in this version

- The publisher builds the weights; consumers trust them. Checking the source
  binding (step 4) does not prove the weights came from that source. Rebuilding
  and comparing hashes would, but only on the same PyTorch version and hardware.
- Trust is per-consumer (TOFU + `wgt.sum`). There is no checksum database or
  transparency log, and assets are not signed.
- Release assets can be deleted by the repository owner; there is no mirror or proxy.

## Not implemented yet

- Packages that themselves `require` other packages, and minimal version
  selection across a dependency graph. `publish` refuses packages with requirements.
- Major-version path suffixes (`/v2`) enforcement.
- Mirror URLs and hosts other than GitHub releases; sharding of very large
  weights (GitHub limits the size of each release asset).
- A content-addressed blob store shared across packages (the cache is per
  module version today).
- License fields in published manifests and license reporting for imports.
- Private repositories and authentication.
- Referencing large or restricted datasets.
