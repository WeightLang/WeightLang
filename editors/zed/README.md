# WeightLang for Zed

Syntax highlighting, bracket pairing, comment toggling and a declaration outline
for WGT 0.5 models and functions. Colors follow your Zed theme. No Python, PyTorch,
model execution, Rust extension code or language server is needed for highlighting.

Version 0.2.0 supports WGT 0.5. The manifest pins the published grammar commit.
Zed registry publication and GUI acceptance are separate from this source release.

## Install locally

1. Clone this repository and keep this checkout on disk.
2. In Zed's command palette, run **`zed: install dev extension`** (or open
   Extensions and choose **Install Dev Extension**).
3. Select **`editors/zed`**, not the repository root.
4. Open a `.wgt` file; the language selector should show **WeightLang**.

Zed downloads and builds the grammar from the commit pinned in `extension.toml`.
The parser is in the same repository under `editors/tree-sitter-wgt`; the manifest's
`path` selects that directory. First installation needs network access and Zed's
WASI toolchain download. Subsequent highlighting works offline.

This package is implemented but has not been submitted to the Zed extension
registry. Local installation is a development extension, not a registry release.
No installer modifies your settings or overwrites another extension.

## Features and limitations

- Highlights declarations, builtin operations, types, parameters, numbers,
  strings and comments. A `#` inside a quoted path remains part of the string.
- Lists functions, model, block, layer, input/output, constant and dimension declarations
  in the outline. Declarations remain flat rather than nested by indentation.
- Pairs braces, parentheses, square brackets and quotes; indents nested function bodies.
- Does not provide completion, diagnostics, definition navigation, formatting or
  indentation-aware block folding.
- Excludes `wgt init 1` algorithm scripts and `backend torch-dynamic` function bodies.

Native parser and query behavior are tested automatically. A WASM compilation
check is separate from actual Zed GUI acceptance; the latter is not automated here.

## Development

Change the parser in [tree-sitter-wgt](../tree-sitter-wgt/README.md), regenerate
queries with `node scripts/generate_editor_support.cjs`, and run
`node scripts/test_editor_support.cjs` from the repository root.
After publishing a grammar change, update the manifest's `rev` to its full commit
hash. Do not use a moving branch for released grammar dependencies.

For local parser iteration, Zed also accepts a `file:///absolute/path/to/WeightLang`
grammar repository with a committed revision and the same subdirectory `path`.
Keep such machine-specific changes out of commits. Rebuild via the extension's
**Rebuild** action in Zed.

See the official [language extension documentation](https://zed.dev/docs/extensions/languages)
and [development instructions](https://zed.dev/docs/extensions/developing-extensions).
