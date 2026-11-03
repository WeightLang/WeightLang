# Tree-sitter WGT

An error-tolerant editor parser for declarative WeightLang (`.wgt`). The generated
C parser uses ABI 14. Zed consumes the WASM build; other Tree-sitter editors can
reuse the C parser and `queries/highlights.scm` with their own registration.
This directory is the grammar root, not the repository root.

## Build and test

Install the Tree-sitter CLI (tested with 0.26.9) and a C compiler. From this directory:

```sh
tree-sitter generate --abi 14
tree-sitter test
tree-sitter build
# Optional WASM build; the CLI may download wasi-sdk into its disk cache.
tree-sitter build --wasm -o wgt.wasm
```

From the repository root:

```sh
python scripts/generate_tree_sitter_queries.py
python scripts/generate_tree_sitter_queries.py --check
python -m pytest tests/test_tree_sitter_support.py -q
```

`grammar.js` defines structure. The shared `editors/vocabulary.json` supplies
builtin names and types; regenerate it with `scripts/generate_editor_support.py`
when the language changes. The query generator updates this directory and Zed
from the same source. Commit generated C/JSON/header files after grammar changes.

The tests check structural corpus fixtures, repository models, actual query
captures, namespaced calls, CRLF/EOF, incremental edits and recovery after an
unfinished declaration. Corpus fixtures exercise editor tokenization, including
legacy declarations; they are not runtime conformance examples.

## Boundaries

The editor parser deliberately accepts incomplete/legacy expressions without
validating versions, shapes, operator signatures or execution. The authoritative
model parser remains `src/weightlang/syntax.py`.

Declarations are flat: indentation is not represented as block ownership.
The outline lists declarations; it does not claim nested block folding. Indentation
queries cover `()` and `[]` continuations only. `wgt init 1` algorithm scripts and
the experimental `backend torch-dynamic` dialect are excluded from this release.
They need their own grammar rules before they can be claimed as supported.

For Zed installation, see [the extension README](../zed/README.md). Helix and
Neovim need their own filetype/grammar registration; including the parser here
does not automatically register it upstream.
