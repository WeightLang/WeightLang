# Changelog

## Unreleased

- Start the native `wgt lsp` server for tensor diagnostics, shape and parameter
  hints, completion, navigation, document symbols and semantic highlighting.
- Add executable-path settings, a restart command, lifecycle tests and workspace
  trust handling; Restricted Mode retains lexical highlighting.
- Bundle the standard LSP client and its dependency license notices into the
  VSIX, with no package installation required after extension installation.
- Highlight named shapes, tensor postfixes, tuple bindings, model names and
  `float`/`int`/`bool` learn types; correct native declarations and range operators.

## 0.2.0

- Update highlighting to WGT 0.5 functions, control flow, native training and forward builtins.
- Handle JSON string escapes, boolean/null literals, logical operators and brace pairing.
- Replace retired Python generators with an offline Node generator and native editor checks.

## 0.1.0

- Recognize `.wgt` files as WeightLang.
- Highlight declarative WGT keywords, built-in operators, tensor types, named
  parameters, numbers, strings, comments and graph expressions.
- Add comment toggling and bracket/quote pairing.
- Include the WeightLang logo and a tokenizer-rendered syntax preview.
- Run entirely offline with no executable extension code.
