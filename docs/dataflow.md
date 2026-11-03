# Architecture composition (wgt 0.3, unreleased)

A layer names both an operation and its output tensor. Without `from`, it takes
its predecessor's output. The last layer is the result of a model or block.
Existing sequential sources and their weight names remain unchanged.

## Pipelines and branches

```wgt
wgt 0.3
model gated
input x: f32[16]
output y: f32[16]
layer gate: linear(in=16, out=64) -> silu from input
layer value: linear(in=16, out=64) from input
layer product: mul(gate, value)
layer down: linear(in=64, out=16)
init seed=42
```

`->` composes operations left to right. `from name` supplies the input to the
**whole** pipeline. Pipeline weights have stable paths such as `gate.step0.weight`.
Use separate named layers when individual internal stages need public names.
Pipelines can appear inside `repeat`, `recur` and `residual`.

`input` always means the current model/block's original input. At model scope,
the input port name (such as `x`) is also accepted until a layer shadows it.
References are local to a block and must name an earlier layer, `input`, or
`state` inside a scan cell. References cannot reach into a different block.
Duplicate layer names and forward/self references fail before weight allocation.
The name `input` is reserved; `state` is reserved inside scan cells.

| Operation | Meaning |
| --- | --- |
| `add(a, b, ...)` | Elementwise sum |
| `mul(a, b, ...)` | Elementwise product; use with sigmoid/SiLU branches for gates |
| `concat(a, b, ...)` | Concatenate on the last feature dimension |
| `mix(g, a, b)` | `g * a + (1 - g) * b`; apply sigmoid yourself if g should be in [0,1] |
| `where(c, a, b)` | Select a where c > 0, otherwise b, element by element |

Merge values are f32; where additionally accepts a bool condition. Add/mul/mix/where require identical complete sample
shapes: there is no implicit broadcasting. Concat requires matching shapes except
for the last dimension. Batch dimensions are never concatenated. Merge operations
occupy named layers rather than inline pipeline stages.

`where` is an **eager selection**: both branches have already executed. It does
not save compute, implement lazy `if/else`, or give a gradient to its condition.
Selected values remain differentiable. Soft gates (`mul`/`mix`) train normally.

## Depth versus shared iteration

```wgt
layer stack: repeat(6, residual(ff(dim=64)))
layer refinement: recur(4, residual(ff(dim=64)))
```

Assume `ff` is a previously defined shape-preserving block.

- `repeat(N, body)` expands N independent copies, named `step0` through `stepN-1`.
  Each copy owns its parameters. This is suitable for Transformer depth.
- `recur(N, body)` runs the same body N times with one set of parameters. The body
  must preserve the complete shape and dtype. All iterations participate in
  backpropagation. Weight paths live directly under the loop name.

Counts may use constants and block parameter expressions. They must be integers
in [1,1024]. Expansion is bounded to 10,000 resolved nodes and 64 nested block
calls. These are compiler expansion limits, not a promise that any resulting
model fits in memory. Neither form is a runtime `while` loop.

## Sequence state

```wgt
wgt 0.3
model memory
input x: f32[8,4]
output y: f32[8,16]
block cell(features, hidden):
  layer joined: concat(input, state)
  layer gate: linear(in=features + hidden, out=hidden) -> sigmoid
  layer candidate: linear(in=features + hidden, out=hidden) -> tanh from joined
  layer next: mix(gate, candidate, state)
layer memory: scan(hidden=16, cell(features=4, hidden=16))
init seed=42
```

`scan` takes f32[L,D]. At time t, its block receives `input = x[t]` and
`state = h[t-1]`. The block must output f32[hidden]; that output becomes the next
state. Scan returns all states as f32[L,hidden]. `last` selects the final sequence
item, e.g. `layer head: last -> linear(in=16, out=3)`.

Weights are shared across time. Initial state is zero and resets on every model
call; gradients pass through the entire sequence. Batched samples have independent
states. Scan has one vector state; explicit initial state is described below.
Tuple states, implicit persistent state, padding masks and dynamic lengths remain unsupported.
The cell's references are scoped to the cell; nested blocks have their own input.

## Standard architecture blocks

```wgt
wgt 0.3
use std/transformer as t
model language_model
input tokens: i64[16]
output logits: f32[16,128]
layer embed: t.tokens(vocab=128, dim=32, max_len=16)
layer decoder: repeat(4, t.gated_decoder(dim=32, heads=4))
layer head: layernorm(dim=32) -> linear(in=32, out=128)
init seed=42
```

`std/transformer.swiglu(dim, hidden=dim*4)` implements a SiLU-gated feed-forward
block. `gated_decoder(dim, heads, hidden=dim*4)` combines causal pre-norm attention,
SwiGLU and residual connections. It inherits the existing attention limitations:
learned positions, no KV cache or padding mask.

`std/recurrent.gru(features, hidden)` wraps a shared GRU cell in a scan. Its reset
gate applies **before** the candidate hidden projection; this is not PyTorch's
reset-after GRU variant and its checkpoints are not interchangeable with it.
The reusable `gru_cell(features, hidden)` block is intended for scan bodies.

Runnable sources: [SwiGLU](../examples/swiglu.wgt),
[gated Transformer](../examples/gated_transformer.wgt),
[custom recurrent cell](../examples/recurrent.wgt), [GRU](../examples/gru.wgt).
They initialize architectures; they are not trained models or quality benchmarks.

## Artifacts and tools

These additions use the unreleased v3 artifact schema with optional `inputs`
references and `recur`/`scan` structural nodes. Repeat and pipelines lower to
ordinary named sequences. Python loading and Rust artifact inspection independently
validate references, shapes and weight layouts. `wgt graph` renders branch edges
and recurrent feedback, and counts shared parameters once. Old tools that do not
recognize these operators need updating before reading these artifacts.

Imports/freezing use the same stable dotted tensor paths. Import compatibility
includes wiring as well as operations and dimensions. Explicit placement works
across branches; merge values transfer to the merge's execution device.

## Tensor arithmetic shorthand (0.3, unreleased)

`layer result: a + b` and `layer result: a * b` lower exactly to `add(a, b)` and
`mul(a, b)`. Only two tensor names are accepted, with the same scope, f32, shape
and axis checks as explicit merges. No broadcasting, scalar operands, chained
expressions, extra `from`, or inline pipeline merges. Use named intermediate
layers for complex expressions. Anonymous residual pipelines need at least two
steps and retain `step0`, `step1`, etc. weight paths; named block paths remain
unchanged. Switching a named block to a pipeline can change weight paths.


## 可选动态 Torch 模型前端（0.3，本地未发布）

`backend torch-dynamic` 提供 `module/setup/fn` 的 Python 兼容可执行模型源码，
支持动态张量、多输入／输出和显式跨调用状态；它与静态 layer/block 图分开，
`check --shapes` 只报告运行时契约，不提供静态形状证明。构造参数沿用对应 Torch 类，
GRU 为 PyTorch reset-after。制品使用独立格式，加载须显式信任源码；Rust、静态导出、
revision/自动复用不支持该格式。完整语法、参数、默认值、保存与验证边界见
[动态张量程序](tensor-programs.md)。

## Named tensor boundaries and explicit scan state (0.3, local/unreleased)

Static models can declare multiple `input name: dtype[shape]` ports and multiple
`output name: dtype[shape] from tensor` ports. Every output of a multi-output model
must specify `from`; references select a top-level input or layer result. A single
output without `from` still selects the final layer. Multiple inputs, multiple
outputs, or any explicit output reference opt into a dictionary return, keyed by
output names. Legacy single-input/output sources continue to return one tensor.

Call `model(samples=x, old=h)` or supply tensors in declaration order. Missing,
extra or duplicated names fail before execution. Samples have fixed positive
shapes and exact dtypes (`f32`, `i64`, `bool`); all inputs must have the same optional
leading batch dimension and device. No implicit shape broadcasting or dtype cast
occurs. The first input is also named `input` inside the graph. Port names cannot
be duplicated or shadowed by top-level layers. Block scopes remain local.

`where(condition, a, b)` additionally accepts a same-shaped bool condition; its
values and output remain f32. Both branches execute. `identity` and `rearrange`
can carry bool/index tensors without changing dtype; arithmetic is still f32.

`scan(hidden=H, cell(...)) from sequence, initial` accepts f32[L,D] and f32[H],
with identical batch dimensions. Omit the second tensor to use zeros. Scan still
returns f32[L,H]; use a `last` layer and a named output to return the final state.
Python controls whether state is retained or detached, so gradients can continue
across chunks. The same cell weights are used at every step. `recur` composes as
an ordinary graph node; it does not acquire a hidden cache.

Runnable example: [streaming_ports.wgt](../examples/streaming_ports.wgt).

```python
import torch
import weightlang
model = weightlang.load("examples/streaming_ports.wgt")
x = torch.randn(2, 6, 2)
h = torch.zeros(2, 4)
first = model(samples=x[:, :3], old=h)
second = model(samples=x[:, 3:], old=first["new"])
loss = second["states"].square().mean()
loss.backward()  # includes both chunks, unless Python detaches the state
```

Named boundaries use artifact version **4**, with ordered `interface.inputs` and
`interface.outputs`; each output records `from`. Python and Rust validate all
port shapes, types and references. Saving Python weight updates preserves version
4 and the boundary. Independent Python export preserves dictionary returns and
state gradients. `Artifact.predict` accepts a dict of tensors for these models;
use `artifact.module` for training. `check --shapes` lists all ports.

This first increment supports flat records of fixed-shape tensors and one-vector
scan state. It does **not** expose RoPE KV cache or MoE auxiliary results as graph
ports, introduce symbolic/zero-length/scalar shapes, optional ports, multiple scan
states or general block arguments/returns. Source `data/train`, `inherit` and
derivations are rejected for named interfaces; Python owns training. The existing
special `run_sequence` interface remains separate. This is not completion of the
full composition roadmap.


### Bounded variable lengths

The next increment adds declared integer dimensions and constraints to these same
named interfaces. It retains one-vector scan state but permits different legal
chunk lengths without changing source or parameters. See [bounded dimensions](dimensions.md#bounded-symbolic-dimensions-03-localunreleased).
Version 5 extends v4 with dimension/constraint/expression metadata; fixed-shape
interfaces remain v4. This does not yet expose the specialized KV/MoE results.


## 张量原子、标量与空端口（0.3 本地增量）

当前支持标量 `f32[]/i64[]/bool[]`、非负静态维度及 `dim P min=0 max=N`；
新增 matmul、broadcast、cat/stack/slice、transpose_axes、reduce_sum/mean/max、
gather/scatter_add、eq/lt 和 masked_softmax。归约可以输出标量，空 sum 为零；
全 mask 的 softmax 行输出和梯度为零。0.3 允许无参数计算模型。
Conv1d 支持 groups/dilation/causal 左填充。
完整签名、参数默认值、形状/dtype、空集合边界、限制和例子见 [primitives](primitives.md)。
这些规则替代前文历史增量中“所有端口正整数、禁止标量”的限制；
KV/MoE 的显式结果端口、具名 block 返回和多状态 scan 见下节；完整精度策略仍未完成。

## Probabilistic early exit (local 0.3, unreleased)

`recur(N, body, until=stop)` shares body parameters for at most N iterations
(1..1024). Omitting `until` retains fixed-depth behavior. `stop` names a direct
body layer with sample contract f32[1]; the state is f32 and the last body layer
must preserve its complete shape and axes. See the complete signature and
[Chinese reference](reference.zh-CN.md#概率门控提前退出03本地实现尚未发布) and
[runnable example](../examples/early_exit.wgt).

After each update, independently sample a uniform value for each sample and stop
when it is below the probability. Nonfinite or out-of-range probabilities fail.
Stopped samples retain their result; the loop breaks when all samples stop.
By default the full batch is computed until then; `compact=1` instead computes only active samples. At least one
iteration runs. State resets per call; both train and eval sample using Torch RNG.
The discrete decision has no gradient. Output loss alone does not train a gate
used only for stopping; training objectives/estimators remain in Python.
Artifacts, standalone Python export and decompilation retain the gate reference;
Rust validates the contract and weight layout, not tensor execution.

## 具名结果、状态、配置与参数身份（0.3 本地增量）

完整签名、默认值、限制、可运行例子与 Python API 见
[组合契约](composition.md)。本次包含：

- block 末尾 `return output=x field=y`，多实参通过 input/arg1/arg2 访问；结果通过层名.字段引用。
- `scan_state(cell()) from sequence,state0,state1,...`；返回 output 和全部同形同类型状态，支持 T=0 和跨块梯度。
- `rope_attention(...,ports=1) from x,mask,key,value` 返回 output/key/value；
  `sparse_moe(...,ports=1)` 返回 output/aux。ports 默认为 0。
- `parameter identity = layer.weight`、`share other.weight = identity`，以及按身份迁移和兼容性报告。
- `config WIDTH: int default=4 min=2 max=16`（也支持 float/bool），Python config 覆盖和 package_model 固定源码。
- `adapter task: lora target=projection rank=2 alpha=4 freeze=1`，合并/解除合并、独立适配器和完整模型保存。
- 独立训练断点保存优化器、参数身份、RNG 和进度；与完整模型制品严格区分。

以上替代此前增量中“cache/MoE/multi-state 尚未接入”的状态描述。仍不支持任意嵌套 Python
返回对象、一般可选输入或任意动态形状；精度四部分契约与旧算子的完整统一 IR 尚未完成。

## 条件与自适应计算增量（0.3，本地未发布）

`route(branch0,branch1,...) from selector,x` 支持按样本真正选择执行路径；
`recur(N,body,until=stop,compact=1)` 压缩活动样本；
`ponder(N,body,until=stop)` 返回可微停止分布、每步状态和期望计算量，训练留在 Python。
`resize2d`、`pad2d`、`avgpool2d` 支持多尺度空间组合。
完整签名、默认值、约束、限制和可运行例子见 [adaptive-compute.md](adaptive-compute.md)。

原生适配器还支持 `dora/pissa/qlora`；完整方法语义、NF4 存储边界及训练例子见 [适配器](adapters.md)。


## 可编程参数更新（0.3 本地实现，尚未发布）

WGT 使用既有具名张量输入/输出与算子表达更新计算，不新增专用优化器语法。
Python 显式调用更新程序并决定应用结果；加载和普通前向不隐式训练。
Torch 模块接口为 `read_parameters(module, names=None)`、
`call_with_parameters(module, parameters, *args, **kwargs)`、
`write_parameters(module, parameters)`；参数绑定保留梯度，写回不依赖自动求导。
原生 `NativeSession` 支持 `parameter_info()`、
`read_parameter(name, *, start=0, length=None)` 和 `write_parameters(writes)`，
可更新常驻权重及其共享连续视图，下一次 CPU/GPU 前向观察到新值。
完整签名、形状/dtype、冻结、别名、版本、错误与存储限制见
[参数更新](parameter-updates.md)；运行示例见
[更新程序](../examples/parameter_updates/updater.wgt) 和
[原生控制器脚本](../examples/parameter_updates/native.py)。
示例控制器为显式给定系数，未训练；原生后端没有因此获得通用多输入计算图。
