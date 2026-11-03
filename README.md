# WeightLang

[简体中文](README.zh-CN.md)

Native WGT programs for model construction, training and inference. `weightc`,
`wgt` and `wgt-native` are Rust executables. The current source tree has retired
Python implementation files, tests, callbacks, packaging and installers.
These changes are local source capabilities, not a claim about published releases.

## Install and update

Rust/Cargo is required to build from source:

```sh
cargo install --locked --force --path crates/wgt --bin weightc --bin wgt
cargo install --locked --force --path crates/wgt-runtime --bin wgt-native
```

The [shell installer](install.sh) and [PowerShell installer](install.ps1) install
and update `wgt` and `weightc` together. Set `WEIGHTLANG_SOURCE` to a checkout;
outside a checkout they build GitHub main, or the tag in `WEIGHTLANG_VERSION`.
They do not install Python, uv or Torch. Existing model directories are preserved.

```sh
wgt update --version X.Y.Z
# Or select the latest complete native release:
wgt update
```

Updates require a `wgt-vX.Y.Z` release containing both platform binaries and
`SHA256SUMS`. Both hashes and versions are checked before selecting the new
version. `weightc` and `wgt` follow the same project pin or default selection.
Old Python-only releases and incomplete native releases are not update candidates.
See [migration and release packaging](docs/weightc-migration.md).

## Run

```sh
cargo build --locked --workspace --bins
./target/debug/weightc check examples/native_training/build.wgt
./target/debug/wgt run examples/native_training/supervised.wgt
./target/debug/weightc legacy-build examples/native_training/build.wgt -o weights/native-demo
./target/debug/weightc compile examples/native_training/compiled.wgt -o .cache/weightlang/demo
./.cache/weightlang/demo
```

The only source language is `wgt 0.5`: model declarations and functions share one compilation unit. See the [language reference](docs/reference.zh-CN.md). `check` parses without
executing; compilation generates a native executable; scripts explicitly control
training and saving. `legacy-build` (`build` alias) requires `main` to return a
model and writes a complete model directory. It refuses existing destinations.

Use `wgt new demo` to create a native script project, `wgt sync` to synchronize
package dependencies, and `wgt run main.wgt` to run it. No Python environment
is prepared. Native artifact inspection, graph rendering and package fetching
remain separate from script execution.

## Scope and documentation

The native runtime supports the subset described in [native training](docs/native-training.md),
[native inference](docs/native-backend.md), [quantization](docs/quantization.md)
and [adaptive execution](docs/self-modifying-runtime.md). Python removal does
not mean every historical operator, trainer or CLI command has a native replacement.
Old CLI-dependent `wgt check/build/package/install/train/export/test/trace/debug`
commands report that they are retired. Use the native script commands above.

`std/` retains WGT library sources; `language/primitive_specs.json` holds operator
signatures. Retained library sources and artifact fixtures are migration material,
not proof of execution support. See the [WGT 0.5 reference](docs/reference.zh-CN.md),
[package guide](docs/packages.md), [command guide](docs/wgt.md), and
[editor guide](docs/editors.md).

Complete models and retained experiment results belong in `weights/`.
Large build/download caches must be disk-backed; small scratch files use
`/tmp/weightlang/`. See [storage](docs/storage.md). Model artifacts and optimizer
checkpoints are distinct; initialization alone is not training.

## Verification and releases

```sh
cargo test --locked --workspace
cargo build --locked --release -p weightlang-wgt --bins
sh scripts/package_wgt.sh --binary target/release/wgt --output dist/native-release
# Windows: scripts/package_wgt.ps1 -Binary target/release/wgt.exe -Output dist/native-release
```

Packaging verifies matching versions and creates both platform assets plus
checksums in a new directory. Publishing is a separate operation.

[MIT license](LICENSE).
