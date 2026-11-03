# wgt: native project and package tooling

> 当前命令已拆分：`build` 生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例使用 `weightc legacy-build`；详见[分发包、元数据与迁移规范](bundles.md)。


`weightc` is the compiler. **`wgt` is a separate Rust executable**, inspired by
Cargo's project workflow and external subcommands. This implementation is on
main; it is not part of the published weightc 0.2.0 Python wheel or a published
wgt binary release.

## Install

```sh
cargo install --path crates/wgt --locked
wgt --help
wgt --list
wgt doctor
```

The installed executable does not embed, import, or invoke Python for package
management, artifact verification or graph rendering. HTTPS uses Rust's reqwest
and rustls. Remote tag resolution uses `git`; uploading releases uses `gh`.

`wgt sync` prepares a project-scoped Python/PyTorch environment using uv and the
compiler bundled in the native binary. `build`, `run`, `check` and `test` then use
that environment. `WEIGHTC=/absolute/path/to/weightc` is an explicit external
compiler override. See [managed environments](environment.md).

## Project workflow

For a project with separate model entries, local block libraries, datasets and
behavior tests, see [complete_project](../examples/complete_project/README.md)
and the [project layout convention](project-layout.md).

```sh
wgt new demo                   # creates demo/wgt.toml, wgt.sum, main.wgt and tests/main.json
cd demo
wgt sync                       # prepare Python/PyTorch and lock dependencies
wgt check                      # parse/type-check every root .wgt, including libraries
wgt export                      # compile model files into weights/; executes training directives
wgt run --input '[[0,1]]'       # builds main.wgt, then infers
wgt test                       # runs tests/*.json behavior assertions
wgt graph weights/demo --open
```

`wgt init` initializes an existing directory. Neither creation command overwrites
existing project files; an existing `.gitignore` receives missing defaults while
retaining its original contents and explicit exceptions. Defaults exclude model
weights, checkpoints, caches and build outputs (see [storage](storage.md)).
The starter trains a small linear identity map on four explicit rows; its
behavior test checks two of those rows within an absolute tolerance of 0.001.
This is a toolchain example, not a held-out generalization benchmark.

`-C DIR` selects a directory; project commands search upward for `wgt.toml` or legacy `wgt.mod`.
Source paths for weights/check/run are relative to the discovered project root.
Use `wgt export model.wgt` or `wgt run --source model.wgt --input '[1,2]'`
to select a source. Building always calls the compiler; there is no incremental
training cache yet.

## Dependencies

```sh
wgt add github.com/WeightLang/WeightLang/packages/gates  # latest stable release, then pin
wgt add github.com/WeightLang/WeightLang/packages/gates --version v0.2.0
wgt tree
wgt metadata
wgt fetch
wgt fetch --locked
wgt export --offline --locked
wgt update github.com/WeightLang/WeightLang/packages/gates --version v0.2.0
wgt remove github.com/WeightLang/WeightLang/packages/gates
```

These commands retain the existing **wgt.mod / wgt.sum** formats and
`WEIGHTLANG_CACHE` location. Existing consumers and `weightc` keep working.
Add/update fetch successfully before changing project metadata, preserve
comments in the manifest, and retain immutable hash checks for known versions.
Remove does not rewrite source imports.

`fetch` verifies the tag's commit, release index, every asset's exact byte size
and SHA-256, and the tagged source against the artifact source. It validates
safetensors headers, tensor shapes, dtype, origins, and contiguous data ranges.
A module is staged and installed only after verification. Native clients
sharing the same cache serialize mutations with an advisory lock.
The lock hashes use trust on first use, not publisher signatures.

`--offline` never contacts the registry. All required files must already be
locked and cached. `--locked` prohibits dependency edits and lock changes.
Normal builds reuse verified cached packages; when dependencies are absent,
they fetch unless offline. Explicit fetch rechecks remote tags.

Current scope: **exact versions and direct GitHub dependencies**. This is
Cargo-like command ergonomics, not Cargo's full feature set: no semver solver,
transitive dependencies, workspaces, hosted registry service or feature unification.
`add MODULE` and `update MODULE` select the highest numeric stable published
version for that package; `--version latest` is equivalent. Drafts, prereleases
and other subdirectory packages are excluded. Discovery follows all GitHub release
pages (up to 1000, then fails); it does not use the repository-wide Latest release.
An exact `--version vX.Y.Z` skips discovery. `update` without MODULE still updates
the toolchain, not model dependencies.

The selected version is fetched and verified before writing an **exact** manifest
pin and `wgt.sum`; future sync/build commands do not chase latest. Missing release
assets or verification failures are errors, not permission to fall back silently.
Discovery requires network access and GitHub API availability/rate allowance;
`--offline` requires an explicit cached and locked version. `--locked` rejects
add/update dependency edits. No version ranges or dependency solver are added.

Bundled libraries need no `add`: in `wgt 0.3`, write `import std/nn` or
`import std/nn as n`. Existing `use std/nn` remains supported. These import
computation definitions, not pretrained weights; their version follows the compiler.
Use `wgt add weightlang/gates` for registered names. See [Git registry](registry-design.md)
for hash locks, registration PRs, and `publish --register-only` recovery.

## Behavior tests

Each `tests/*.json` file declares a source, input, expected numerical output and
optional absolute tolerance (default `1e-6`):

```json
{
  "source": "main.wgt",
  "input": [[0, 1], [1, 0]],
  "expected": [[0, 1], [1, 0]],
  "atol": 0.000001
}
```

Each model builds once per test invocation. Shape or value mismatches fail the
command with a nonzero exit status. These are numerical assertions, not inferred
training-quality claims.

## Packaging and publishing

From a committed, clean package directory whose module path matches its Git
location:

```sh
wgt export
wgt package --version v0.1.0 --artifacts build -o dist
wgt publish --index dist/wgt.index.json --license MIT --dry-run
# Push the exact commit and annotated package tag shown in the index, then:
wgt publish --index dist/wgt.index.json --license MIT
```

`package` consumes **prebuilt** artifacts and needs no Python. It checks source
hashes against local committed .wgt files, rejects untrained tensors by default,
and writes the same release assets/index as `weightc publish`. Output must be
empty. `--allow-untrained` is an explicit experimental escape hatch and does
not alter provenance. Packages with dependencies cannot be published yet.

`publish` checks the existing remote tag matches the index commit and invokes
`gh release create --verify-tag --latest=false`. It does not create/push tags,
overwrite releases, or claim the CLI's Latest slot. Dry-run uploads nothing.
By default it then verifies the remote release and submits a registration PR;
`--license` is required, `--no-register` opts out, and `--register-only` retries
submission after a partial failure. See [registry workflow](registry-design.md).
Ignore weights/ and dist/ in the package's Git configuration.

## Native architecture rendering

One command renders HTML and opens the default browser:

```sh
wgt graph weights/demo --open
```

Without `-o`, output goes to `<system temporary directory>/weightlang/model.<format>`
(on Linux, normally `/tmp/weightlang/model.html`). The same file is atomically
replaced on each invocation. Temporary filenames do not accumulate per model or
run. The command prints the resulting path. `--open` is supported for HTML and uses
the platform's default opener; if no graphical session is available, the command
reports failure and retains the generated file.

Use `--stdout` for pipes or agents, e.g.
`wgt graph weights/demo --format json --stdout`. It conflicts with `-o` and `--open`.


```sh
wgt graph weights/demo --format html -o model.html
wgt graph weights/demo --format svg -o model.svg
wgt graph weights/demo --format md -o model.md
wgt graph weights/demo --format json -o model.json
wgt graph weights/demo --format toon -o model.toon
wgt graph weights/demo --format mermaid --stdout
wgt graph weights/demo --format dot --stdout
```

Input is an artifact directory or its manifest.json. All artifact hashes and
tensor layouts are verified before rendering. Rendering does not execute
models, train them, or need Python, Graphviz or a CDN.

The HTML viewer opens with a module-level architecture overview. Use
`展开内部层` to inspect every underlying node and `返回模型总览` to collapse them.
Select a node for its purpose, input/output shapes (without the batch axis),
and parameter count; overview counts include the module's internal layers.
Solid connections show data flow, dashed connections show residual skips,
and dotted connections show recurrent feedback. `查看参数值` opens the existing
tensor inspector, including slicing and baseline comparisons. These are
parameter values, not execution activations or attention probabilities.

Use the header's `Language / 语言` selector to switch between Chinese and English.
The initial language follows the browser (Chinese for `zh-*`, English otherwise);
the viewer remembers an explicit selection in browser local storage when available.
Switching preserves the selected node, overview/detail mode, tensor, axes, slice
indices and comparison mode. Model paths, tensor names and operator arguments
remain unchanged. Native file-picker text follows the browser/OS language.

The default `Tensor view` draws vectors as cells, matrices as grids, and
higher-rank tensors as stacked matrices. At most three axes are displayed,
defaulting to the first three; the node inspector lets users hide one axis and
show another. Hidden axes are fixed at index 0, never flattened or averaged.
These glyphs describe tensor shapes,
not activations: they do not execute the model or invent values. An ellipsis
marks omitted elements; the full dimensions remain visible. `Compact view`
restores text-only nodes. The weight inspector still shows recorded parameters.

The theme selector supports light, dark and system appearance, remembers the
choice when browser storage is available, and updates heatmap colors as well.
The reusable [frontend library](../web/viewer/README.md) exposes `configure()`,
`nodeRenderer`, `tensorRenderer`, CSS tokens, and bounded shape-preview limits.
The CLI embeds these same frontend sources; `--style` accepts a local CSS theme.

- **HTML:** standalone diagram plus expandable tensor/provenance and JSON views.
- **SVG:** a directly viewable, self-contained vector diagram.
- **Markdown:** Mermaid flowchart and readable layer table.
- **JSON:** versioned `weightlang.graph` schema with nodes, edges and tensors.
- **TOON:** the same graph data using counted tabular arrays, targeting the
  [TOON 4.4 specification](https://github.com/toon-format/spec/blob/main/SPEC.md).
- **Mermaid / DOT:** source for other diagram tools.

Dotted paths and parent fields retain block nesting. Residuals have explicit
add nodes and skip edges. Widths omit the optional batch dimension. Shapes in
the tensor table use `2x8` notation. Scalar parameters are counted exactly once
at leaf tensors; block boundary nodes own zero. Origins are displayed, not
inferred from parameter values. Explicit `-o` output files must not already exist; default temporary outputs are replaced.

A rendered example from [examples/residual.wgt](../examples/residual.wgt):

![Residual architecture](images/residual.svg)

## Extensions

Put an executable named `wgt-hello` on PATH:

```sh
#!/bin/sh
printf 'Hello from %s\n' "$WGT_PROJECT_DIR"
```

`wgt hello arg1 arg2` executes it with `arg1 arg2` as separate arguments and
preserves its exit code. Built-ins take precedence. Global `-C` sets the working
directory. `WGT_PROJECT_DIR` is provided when a project is found;
`WGT_OFFLINE` and `WGT_LOCKED` are `1` or `0`. Plugins are ordinary trusted
executables, not sandboxed; they must implement those flags themselves.
`wgt --list` lists installed extensions. No plugin registry or Python entry-point
system is required.

## Local checks

```sh
cargo test --locked
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
```


Main also verifies artifact v3 (wgt 0.3 Transformer/CNN models). Graph format v2
adds `input_shape` and `output_shape` typed strings to each node, e.g.
`i64[4]`, `f32[4,8]` or `f32[1,28,28]`. Legacy numeric `input`/`output`
fields retain the last dimension; use the shape strings for full contracts.
Attention parameter counts include its QKV and output projections.

## Storage and replacement

See [storage conventions](storage.md). wgt export [-o ROOT] [--force] writes weights/<model>/ by default and refuses an existing model unless --force is supplied. run and test rebuild only disposable copies under .cache/weightlang/. Package defaults also use weights/.

## TOML configuration and one-command sync

New `wgt new/init` projects use `wgt.toml` and an empty `wgt.sum`. Legacy
`wgt.mod` projects remain supported. Do not keep both manifest formats in one
project: all clients reject that ambiguity. Python `weightc` also reads TOML
for dependency resolution (Python 3.10 uses the conditional `tomli` dependency).

```toml
[project]
module = "local/demo"
edition = "0.3"
description = "My model project"

[model]
source = "models/main.wgt"
output = "weights"
description = "Training entry point"

[revisions]
max_versions = 5  # versioned weightc legacy-builds; 0 keeps all snapshots

[runtime]
device = "cuda:0"

[runtime.placements]
"body" = "cuda:0"
"body.fc1.weight" = "cpu"
"head" = "cpu"

[dependencies]
"github.com/WeightLang/WeightLang/packages/gates" = "v0.2.0"
```

The model name and architecture remain in the `.wgt` source. TOML records the
entry point, output root, description and execution defaults. Source/output
paths are relative to the project root. With no configured source, build retains
legacy root-source discovery; run defaults to `main.wgt`.

```sh
wgt sync                  # prepare runtime, sync wgt.sum and wgt.lock
wgt sync --offline        # require cached runtime and model dependencies; no network
wgt sync --locked         # require existing lock; prohibit changes
wgt export                 # use configured source/output/devices
wgt run --input '[[0,1]]'
wgt export --device cpu --place body=cpu --place head=cpu
```

CLI source/output/device values override TOML defaults. Individual `--place`
entries override matching project entries; other project placements remain.
An explicit `--device-map` replaces the project placement map, with CLI
`--place` overrides on top. `--device cpu` alone does not erase specific GPU
placement rules; use `--device-map` with an empty JSON object to clear them.
`wgt test` applies runtime defaults to each fixture's source.

`wgt.sum` **is the dependency lock file**, shared by Rust and Python clients.
It records exact versions, resolved Git commits, index hashes, and artifact
manifest/source/weight hashes. Commit it with the manifest. Python runtime packages have a separate `wgt.lock`; see [managed environments](environment.md). `sync` retains known hashes, removes stale dependency entries
unless locked, and reuses verified caches without contacting the registry.
Use `fetch` for an explicit remote recheck, `add/update` for explicit dependency
changes. `sync` also prepares Python/PyTorch unless `--no-env` is supplied. It
does not train models or choose newer model-package versions. Model dependencies
remain direct and exact-version only. `--upgrade-env` explicitly updates Python
package resolution.

See [devices](devices.md) for parameter residence, execution devices, gradient
transfers, and their performance limitations.

## Project toolchain version

Use `.wgt-version` to pin the native CLI independently of the language edition.
`wgt version VERSION` writes it; `wgt sync` installs a missing pinned executable.
See [toolchain versions](toolchains.md) for offline use and local imports.

Native tool versions: `wgt update` selects the latest stable release;
`wgt update --version X.Y.Z` switches to an exact version;
`wgt download X.Y.Z` downloads without switching. See [toolchains](toolchains.md).

## Source diagnostics and bounded debugging

`wgt check SOURCE --error-format json` forwards structured compiler errors.
`wgt trace SOURCE [--weights ARTIFACT] [--sample FILE] [--watch PATH] [--time 0,3,7] [--json]`
inspects one input. `wgt debug SOURCE [--weights ARTIFACT] --steps 10` performs CPU training diagnostics
without writing model artifacts. `wgt explain E0301` explains a diagnostic code.
See [debugging.md](debugging.md) for flags, report contracts and limitations.

## One-command upgrades

```sh
wgt update --all     # upgrade all direct model packages and lock exact results
wgt self update      # download, verify and select latest native wgt for this platform
```

`wgt update` remains a self-update shorthand; `wgt update owner/name` updates one
package. Bulk upgrades preserve each registered package's registry identity and
write project files only after all downloads pass verification. `--all` does not
change the Python environment or toolchain pin. See [update semantics](toolchains.md).
