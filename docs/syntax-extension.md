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
  dimension/count parameters still require positive integers: `8 / 2` is not an integer dimension.
  Block parameters themselves allow finite numbers, including eps/probability floats and zero flags.
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

Implemented follow-up: [architecture composition](dataflow.md) adds named
`repeat`, shared `recur`, sequence `scan`, pipelines and branch/merge wiring,
including runtime, artifact and Rust inspection support.

Remaining candidate, not implemented:

1. Operator registration that supplies argument validation, shape inference,
   execution and artifact compatibility together. A parser-only registration
   mechanism would accept programs the runtime cannot execute.

Validation covers old syntax, expression precedence, invalid inputs, lexical
library scopes, default overrides, source locations, and forward/backward
execution of an expanded residual block.


Implemented dimension follow-up: [named dimensions and transforms](dimensions.md)
adds named sample axes, inferred `linear` input widths, explicit `rearrange` and
`reduce`, plus per-layer shape display. Source examples and the latest-only
WGT development skill are maintained with the implementation.

Implemented source sugar (unreleased 0.3): binary tensor `a + b` / `a * b`,
`linear(out)` with one positional numeric expression, and named-axis
`mean(axis=NAME)`, `sum(axis=NAME)`, `transpose(AXIS, ...)`. Tensor arithmetic
requires two names in a standalone layer; numeric argument expressions retain
their existing grammar. Axis sugar resolves during shape checking and serializes
only canonical `reduce`/`rearrange` operators. Anonymous residual pipelines are
also supported. See [signatures and limitations](reference.zh-CN.md) and
[the runnable example](../examples/syntax_sugar.wgt).


## 可选动态 Torch 模型前端（0.3，本地未发布）

`backend torch-dynamic` 提供 `module/setup/fn` 的 Python 兼容可执行模型源码，
支持动态张量、多输入／输出和显式跨调用状态；它与静态 layer/block 图分开，
`check --shapes` 只报告运行时契约，不提供静态形状证明。构造参数沿用对应 Torch 类，
GRU 为 PyTorch reset-after。制品使用独立格式，加载须显式信任源码；Rust、静态导出、
revision/自动复用不支持该格式。完整语法、参数、默认值、保存与验证边界见
[动态张量程序](tensor-programs.md)。


## Initialization algorithms

`init file="init.wgt"` in 0.3 selects a `wgt init 1` program. Its `algorithm`
functions and `tensor "PATTERN":` bodies support variables, tensor expressions,
conditionals and loops, independently of forward-graph expressions.
`init_algorithm.py` validates an AST and interprets only declared algorithms
and tensor primitives; it does not execute generated Python or allow host imports.
`initialization.py` binds target contracts and verifies the complete generated
state before import/reuse/freezing. See [initialization](initialization.md) for
complete syntax, primitives, limits and examples. The optional `init python=`
backend is a separate, explicitly declared host callback.


本地 0.3 已新增 `dim NAME min=N max=M` 和 `require LEFT OP RIGHT`，用于同一具名模型
接受有限范围内的运行时维度。每个符号需直接绑定某个输入轴，编译器检查所有合法组合，
运行时先验证约束。制品 v5 保留表达式，详见 [有界维度及限制](dimensions.md#bounded-symbolic-dimensions-03-localunreleased)。
这不代表通用动态形状、零长度 cache 或 KV/MoE 图端口已完成。

`recur(N, body, until=NAME)` adds an optional direct-body control dependency.
The parser retains NAME as a string; shape validation requires f32[1]. Runtime
validates probability values and samples after each update. Reachability lint,
loop reports, artifact validation, Python export and decompilation preserve
this dependency. See [dataflow](dataflow.md#probabilistic-early-exit-local-03-unreleased).

## 条件与自适应计算增量（0.3，本地未发布）

`route(branch0,branch1,...) from selector,x` 支持按样本真正选择执行路径；
`recur(N,body,until=stop,compact=1)` 压缩活动样本；
`ponder(N,body,until=stop)` 返回可微停止分布、每步状态和期望计算量，训练留在 Python。
`resize2d`、`pad2d`、`avgpool2d` 支持多尺度空间组合。
完整签名、默认值、约束、限制和可运行例子见 [adaptive-compute.md](adaptive-compute.md)。
