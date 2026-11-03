# Extensible syntax proposal and first implementation

Status: implemented locally as an additive extension to unreleased `wgt 0.3`.
Released `wgt 0.1` and `wgt 0.2` keep their existing features.

The first step makes reusable blocks more expressive without changing the
runtime graph or tensor naming. Expressions produce numeric arguments; block
calls still expand into named `seq` nodes. No Python evaluation or arbitrary
code loading is involved.

```wgt
wgt 0.3
const WIDTH = 32
model encoder
input x: f32[WIDTH]
output y: f32[WIDTH]
block ff(dim, hidden=dim * 4):
  layer up: linear(in=dim, out=hidden)
  layer act: gelu
  layer down: linear(in=hidden, out=dim)
layer body: residual(ff(
  dim=WIDTH,  # hidden defaults to 128
))
init seed=42
```

## Implemented grammar

- File-level `const NAME = expression`, including in libraries. Constants are
  immutable and evaluate in declaration order; forward constant references fail.
- Numeric expressions in layer/block arguments, block parameter defaults and
  input/output dimensions. Operators: unary `+ -`, binary `+ - * / // %`, and
  parentheses. Multiplication/division/remainder bind more tightly than addition
  and subtraction; binary operators associate left to right.
- `/` returns a float; `//` performs floor division. Dimensions and block
  parameters still require positive integers: `8 / 2` is not an integer argument.
  Division by zero, overflow and non-finite numbers produce source errors.
- Block defaults follow required parameters. Defaults evaluate in signature
  order using constants in the defining library and earlier bound parameters.
  Explicit arguments override defaults; parameters shadow constants. Call
  arguments evaluate in the caller's scope. Constants are private to their file;
  `use` exposes blocks, not a constant namespace.
- Bracketed statements can span physical lines, retain comments and use the
  existing trailing comma support in calls/signatures. Data vectors may also span
  lines. Diagnostics identify the statement start, or an unmatched closing
  bracket's physical line. Unbalanced brackets fail before expansion.

Constants used in port dimensions must be declared before the port; block bodies
and top-level layer arguments resolve after file parsing and see that file's
completed constant scope. Training options, initialization options and inline
numeric data retain their literal syntax. Strings remain the existing quoted
syntax without escapes. There are no executable expressions, functions, boolean
conditions or dynamically shaped tensors in this iteration.

## Extension boundaries and next steps

`expressions.py` owns the numeric expression tree, binding-power table,
interpreter and physical-to-logical line handling. `syntax.py` owns declarations,
scopes, block expansion and static contracts. The resulting `Layer` tree is the
same representation consumed by compilation and execution today.

Next candidates, not implemented:

1. Named `repeat` expansion, with explicit limits and stable tensor paths.
2. Branch/merge graphs, with shape rules and matching runtime/artifact support.
3. Operator registration that supplies argument validation, shape inference,
   execution and artifact compatibility together. A parser-only registration
   mechanism would accept programs the runtime cannot execute.

Validation covers old syntax, expression precedence, invalid inputs, lexical
library scopes, default overrides, source locations, and forward/backward
execution of an expanded residual block.
