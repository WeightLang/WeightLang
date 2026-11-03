# WeightLang language

> 当前命令已拆分：`build` 生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例使用 `weightc legacy-build`；详见[分发包、元数据与迁移规范](bundles.md)。


One model per file, explicit initialization and tensor origins. Released 0.2
uses one-dimensional f32 feature vectors. Main additionally accepts **wgt 0.3**
(unreleased) for fixed-size sequences, i64 tokens and CHW images. Using new
features in a 0.1 source fails with a line-numbered "requires 'wgt 0.2'" error.
Existing 0.1 examples remain unchanged.

## Model syntax

One statement per line; comments begin with `#`. Names are identifiers.
A model needs a version header, `model name`, `input x: f32[N]`,
`output y: f32[M]`, at least one parameter-bearing layer and an explicit `init` statement.
In 0.3, `init file="init.wgt" [seed=N]` and `init python="generate.py[:function]" [seed=N]`
provide complete custom initialization; see [initialization](initialization.md).
Layers run in source order; wgt 0.3 can name earlier tensor outputs with `from`.
Shapes are checked before training. See [architecture composition](dataflow.md)
for pipelines, branches, gates, repeated blocks and shared recurrent state.

| Operator | Meaning | Version |
|---|---|---|
| `linear(in=N, out=M)` | Fully connected with bias; positive integer dimensions | 0.1 |
| `relu`, `tanh`, `sigmoid` | Elementwise activation | 0.1 |
| `gelu`, `silu`, `identity` | Activation / identity | 0.2 |
| `softmax` | Normalize on the last dimension | 0.2 |
| `layernorm(dim=N)` | Normalize the feature dimension; learned weight and bias | 0.2 |
| `dropout(p=P)` | Training-only dropout, 0 ≤ P < 1; identity at inference | 0.2 |
| `residual(block(...))` | x + block(x); input and output widths must match | 0.2 |

## Blocks and libraries

For project/library/block/model terminology and directory conventions, see
[project layout](project-layout.md) and the [complete project](../examples/complete_project/README.md).

```wgt
wgt 0.2
use std/nn
use std/heads as h
model classifier
input x: f32[2]
output logits: f32[3]
block features(in, out):
  layer body: nn.mlp(in=in, hidden=8, out=out)
layer body: features(in=2, out=4)
layer head: h.classifier(in=4, classes=3)
init seed=42
```

This example has untrained weights. See `examples/std_mlp.wgt` for training.
Block arguments are named finite numbers and must match the signature (including defaults).
Dimension/count arguments still require positive integers at their consuming operators;
floats and zero allow eps, probabilities and bias flags.
Blocks contain only indented layer statements, may call other blocks, and cannot
be empty, recursive or named after built-in operators. Missing parameters fail
at their source line. A call expands into a nested sequence with stable names
such as `body.body.fc1.weight`.

`import std/nn` (wgt 0.3) or `use std/nn` selects a bundled library without
`wgt add`, network access or model-package locks. `import std/nn as n` chooses
an alias; model and library files both support this form. Libraries contain
computation definitions, not pretrained weights, and ship with the compiler; its default alias is `nn`.
`use "./layers.wgt" as layers` selects a local library relative to the source.
Library files allow the version header, `use`, `block`, and (in 0.3) `const`
and bundled `import std/...` statements. Weight imports remain model-only.
Cycles are errors. Remote source libraries are not supported.
`weightc std` lists the available block signatures.

`layer skip: residual(nn.mlp(in=4, hidden=8, out=4))` adds the input to
the block output. In 0.3, a pipeline such as `residual(linear(8) -> gelu -> linear(4))`
is also accepted for an input width of 4. Its children use stable `stepN` names.
See [source sugar signatures](reference.zh-CN.md) for tensor arithmetic, named-axis
operations and positional linear. A bare operator such as `residual(linear(...))` is rejected.

## Data

Inline data uses indented `[inputs] -> [targets]` rows:

```wgt
data train:
  [-2, 0] -> [0]
  [2, 0] -> [1]
  [0, 2] -> [2]
```

In 0.2, optional `license="…"` and `source="…"` annotate inline data.
External data uses:

```text
data train: file="train.csv" sha256=<64 lowercase hex digits> skip=1 license="CC-BY-4.0" source="<URL>"
```

Paths are relative to the source. Hashes cover the original file bytes, including
any header. `skip=N` skips initial lines (default 0). Remaining nonempty lines
must contain comma-separated numbers: the first input-width columns are
features, all remaining columns are targets. Quoted numeric CSV fields are supported.
A file without a hash, a hash mismatch or malformed row fails before training.

In 0.3, `format=` defaults to the file extension and supports CSV, TSV, JSON,
JSONL, NPZ and standalone WGT datasets. Use `dataset=` to select a dataset in a
WGT file (default: the current dataset name). All file sources remain hash-pinned.
`data samples: python="hooks.py:load"` or `entrypoint="name"` defers a Python
loader until training. `train ... python="hooks.py:train"` or `entrypoint="name"`
selects a source-declared trainer. See [data formats and callback contracts](datasets.md).

## Training

```wgt
train data=train loss=cross_entropy optimizer=adamw lr=0.01 epochs=200 batch=16 validate=test weight_decay=0.01
```

`data`, `loss`, `optimizer`, `lr` and `epochs` are required.
0.1 supports mse/bce and sgd/adam. 0.2 adds cross_entropy, adamw and these options:

- `batch=N`: seeded mini-batch shuffling; absent or ≥ dataset size means full batch.
- `validate=name`: evaluate a separate named dataset after training. Naming the
  training dataset is rejected; the author must ensure the rows are held out.
- `weight_decay=P`: nonnegative optimizer weight decay, default zero.

BCE requires a final sigmoid and binary targets. Cross entropy requires one
integer class index per row in [0, output width), with raw logits; a final
softmax or sigmoid is rejected. MSE targets match the output width.
Metrics distinguish training loss/accuracy from validation loss/accuracy.
A seed reproduces training on the same CPU and PyTorch version, including dropout.

## Imports and origins

Initialization runs first, imports second, training last:

```wgt
import body.fc1 from "../weights/source" layer=body.fc1 freeze
import body from "../weights/source" layer=body sha256=<hash>
```

Import statements copy all tensors under the exact dotted prefix. Source and
target expanded structures (operations, arguments, child names) must match.
`freeze` excludes only that prefix from training and requires a train statement.
Whole-layer imports also work in 0.1; dotted paths require 0.2.
Package references use `wgt.mod` and `wgt.sum`; see [packages.md](packages.md).

Each tensor records `random-init`, `trained`, `imported` or
`imported+trained`. Imported tensors retain `upstream_origin`. Importing an
untrained tensor does not make it trained. LayerNorm parameters use the same
origin system even though its default initialization uses constants.
Artifacts without training are useful for structure/loader tests, not predictions.

## Limits

No embedding, i64 ports, convolution, attention, general graph wiring,
multidimensional feature shapes or learning-rate schedules in the released language.
Source builds support CPU/CUDA execution and parameter placement through
CLI/Python configuration; see [devices](devices.md).


## wgt 0.3: sequences and images (main, unreleased)

Write `wgt 0.3` to opt in. Existing 0.1/0.2 files retain their vector contracts
and write artifact v2; 0.3 writes artifact v3. Using multidimensional ports,
i64 or the operators below in older files reports a version error with a line number.

Shapes describe **one sample**, excluding the optional leading batch dimension:
`i64[L]` tokens, `f32[L,D]` sequences, `f32[C,H,W]` images.
All sizes are fixed positive integers. Linear and LayerNorm act on the last
dimension; residual blocks preserve the entire shape and dtype.

| Operator | Sample input → output |
|---|---|
| `embedding(vocab=V, dim=D)` | i64[L] → f32[L,D]; indices in [0,V) |
| `position_embedding(dim=D, max_len=M)` | f32[L,D] → same; learned positions, L ≤ M |
| `self_attention(dim=D, heads=H, causal=0)` | f32[L,D] → same; D divisible by H; causal 0 or 1 |
| `meanpool` | f32[L,D] → f32[D] |
| `conv2d(in=C, out=O, kernel=K, stride=1, padding=0)` | f32[C,H,W] → f32[O,H′,W′] |
| `maxpool2d(kernel=K, stride=S)` | f32[C,H,W] → f32[C,H′,W′], no padding |
| `globalavgpool2d` | f32[C,H,W] → f32[C] |
| `flatten` | f32[...] → f32[product of sample dimensions] |

Convolution/pooling use square kernels, dilation 1, and
`floor((size + 2*padding - kernel)/stride) + 1`; empty results are rejected.
Attention uses learned Q/K/V and output projections with bias and scaled
dot-product attention. `causal=1` excludes future positions. No padding mask,
cross-attention, KV cache, RoPE, tokenizer or pretrained checkpoint import is
provided. CNN support currently covers these basic building blocks.

Inline/CSV training rows remain flat, in row-major order. Inputs reshape to
the declared sample shape. With cross_entropy, the last output dimension is
the class count: f32[L,V] expects L integer targets per row and reports token
accuracy. Other losses expect all output scalars. Read-only evaluation JSON
uses nested arrays with shape [N, *sample_shape], and CE targets omit the class axis.

Runnable synthetic examples: [Transformer](../examples/transformer.wgt) and
[CNN](../examples/cnn.wgt). Both demonstrate training and deterministic rebuilds
on their tiny training sets; neither supplies pretrained weights or a quality benchmark.

## Selective training and weight iterations (main, unreleased)

`wgt 0.3` additionally supports standalone `freeze`/`unfreeze` rules and
hash-pinned whole-model `inherit`. See [weight iterations](weight-iterations.md)
for exact selectors, precedence, runnable workflows and limitations. The separate
[revision store v1](specs/revision-v1.md) deduplicates unchanged weight chunks;
checkout produces an unchanged standard artifact for existing consumers.

## Architecture composition (main, unreleased)

`wgt 0.3` supports `->` pipelines, `from` tensor references, `add`/`mul`/`concat`,
soft `mix` gates, eager `where` selection, independent `repeat`, shared `recur`,
and stateful sequence `scan`. See [dataflow.md](dataflow.md) for syntax,
parameter-sharing rules, examples and explicit limitations.


## Named dimensions and transforms (wgt 0.3, local source)

Ports may name sample axes, for example `input x: f32[time=8, feature=32]`.
`linear(out=64)` infers its input width from the actual incoming tensor; explicit
`in=` remains checked. Names propagate through layers and incompatible known
names are rejected even when axis sizes happen to match. A grouped/flattened
axis has no inferred semantic name.

- `rearrange("time feature -> feature time")` permutes sample axes.
- `rearrange("time (head width) -> head time width", head=4)` splits an axis;
  exactly one missing size per input group can be inferred by exact division.
- `reduce("time feature -> feature", op=mean)` averages the omitted axis;
  `op=sum` sums it. Retain at least one output axis.

Batch axes are implicit and never rearranged or reduced. No new axes, implicit
broadcasting, ellipsis or dynamic shapes are supported. Run
`weightc check examples/dimensions.wgt --shapes` to see all expanded layer contracts.
See [the annotated dimensions guide](dimensions.md) for complete examples.

## Construction-only builds

`weightc legacy-build SOURCE --no-train` keeps the `train` declaration in the source but
skips its execution and training-data reads/checks for this build. Initialization,
imports, freezing and automatic compatible HEAD weight migration still run.
See [build/migrate without training](architecture-reuse.md#buildmigrate-without-training)
for migration rules, provenance and standalone output behavior. Without the flag,
`build` continues to execute declared training.


## Audited Transformer primitives (0.3, local)

`linear` accepts optional `bias=0|1` (default 1); `layernorm` accepts `eps=1e-5`,
`bias=1`, `affine=1`; `gelu(approximate=1)` selects tanh approximation (default 0).
`embedding` accepts optional `padding_idx` in the vocabulary; the padding row has
zero gradient, but does not create an attention mask. `self_attention` accepts
`dropout=0` (training only). New `rmsnorm`, `scale` and `rope_attention` operators
and `std/hf_decoder` blocks are described with full signatures in the
[Chinese reference](reference.zh-CN.md) and [migration record](transformers-migration.md).
RoPE attention currently has fixed zero-based positions and no external mask or cache.

`rope_attention` 的 `qk_norm=2` 在拆头前对 Q/K 完整投影向量做 RMSNorm；
模式 1 仍按每头归一化，0 禁用。标准库 `hf_decoder.olmo2_block` 使用模式 2，
以及注意力与 FFN 输出端 RMSNorm 后的残差，完整签名见中文参考。


## 可选动态 Torch 模型前端（0.3，本地未发布）

`backend torch-dynamic` 提供 `module/setup/fn` 的 Python 兼容可执行模型源码，
支持动态张量、多输入／输出和显式跨调用状态；它与静态 layer/block 图分开，
`check --shapes` 只报告运行时契约，不提供静态形状证明。构造参数沿用对应 Torch 类，
GRU 为 PyTorch reset-after。制品使用独立格式，加载须显式信任源码；Rust、静态导出、
revision/自动复用不支持该格式。完整语法、参数、默认值、保存与验证边界见
[动态张量程序](tensor-programs.md)。


具名多输入/输出、bool 条件和显式 scan 初态已加入本地 0.3 静态图。
返回字典与固定形状检查、制品 v4 及限制见 [端口契约](dataflow.md#named-tensor-boundaries-and-explicit-scan-state-03-localunreleased)。
符号维度、KV cache 图端口和多状态循环仍未完成。


本地 0.3 已新增 `dim NAME min=N max=M` 和 `require LEFT OP RIGHT`，用于同一具名模型
接受有限范围内的运行时维度。每个符号需直接绑定某个输入轴，编译器检查所有合法组合，
运行时先验证约束。制品 v5 保留表达式，详见 [有界维度及限制](dimensions.md#bounded-symbolic-dimensions-03-localunreleased)。
这不代表通用动态形状、零长度 cache 或 KV/MoE 图端口已完成。


## 张量原子、标量与空端口（0.3 本地增量）

当前支持标量 `f32[]/i64[]/bool[]`、非负静态维度及 `dim P min=0 max=N`；
新增 matmul、broadcast、cat/stack/slice、transpose_axes、reduce_sum/mean/max、
gather/scatter_add、eq/lt 和 masked_softmax。归约可以输出标量，空 sum 为零；
全 mask 的 softmax 行输出和梯度为零。0.3 允许无参数计算模型。
Conv1d 支持 groups/dilation/causal 左填充。
完整签名、参数默认值、形状/dtype、空集合边界、限制和例子见 [primitives](primitives.md)。
这些规则替代前文历史增量中“所有端口正整数、禁止标量”的限制；
KV/MoE 的显式结果端口、具名 block 返回和多状态 scan 见下节；完整精度策略仍未完成。

The local 0.3 `recur(N, body, until=stop)` extension supports bounded stochastic
early exit with shared parameters. See [dataflow](dataflow.md#probabilistic-early-exit-local-03-unreleased)
for batching, probability contracts and nondifferentiable stopping semantics.

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

## Native tensor math and frontend unification (local 0.3)

The declarative graph now includes square/rsqrt/exp/log/sin/cos/softplus,
shift(value=V), and cumsum(axis=A), plus std/tensor_math.l2norm(eps=1e-6)
and rms(eps=1e-6). See [complete contracts and example](primitives.md).
These operations use f32 and do not implement a general precision policy.
The [unification plan](migrations/unified-language.zh-CN.md) retains dynamic models
in WGT; the legacy torch-dynamic frontend is still required until migration passes
numerical, gradient and artifact checks. Unification is not yet complete.

原生适配器还支持 `dora/pissa/qlora`；完整方法语义、NF4 存储边界及训练例子见 [适配器](adapters.md)。

## Custom and mixed weight domains (0.3, local)

`wtype ternary = [-1, 0, 1]` declares a finite f32 codebook.
`weight head.weight: ternary` applies it to a parameter; optional
`axis=0 start=0 stop=2` applies it only to a half-open slice. Disjoint slices along
one axis can mix different codebooks and untouched floating-point values.
Forward uses nearest-value projection (lower value on ties); backward uses an
identity STE on floating-point master parameters. Python owns training.
See [complete contracts and runnable example](weight-types.md), including named
outputs, shared parameters, artifact v7, and current limitations. This is not
packed low-bit storage or an integer compute backend.


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
