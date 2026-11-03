# 固定维度与命名轴（wgt 0.3）

本文描述当前源码可执行的 `wgt 0.3` 语法，尚未发布。
完整小模型见 [dimensions.wgt](../examples/dimensions.wgt)：144 个可训练参数，
只声明初始化，不含数据和训练。可训练结构不等于训练好的模型。

## 端口、常量与最后一维

```wgt
wgt 0.3
const TIME = 8
const FEATURES = 32
model projection
# time 是序列位置，feature 是每个位置的特征；不包括运行时 batch。
input x: f32[time=TIME, feature=FEATURES]
output y: f32[time=TIME, feature=64]
# 一套权重作用于所有时间步，[8,32] -> [8,64]，不会为每步创建独立权重。
layer project: linear(out=64)
# 仅初始化，没有训练数据，也没有执行训练。
init seed=42
```

端口描述单样本，运行时一批输入可为 `[batch,8,32]`。维度是固定正整数，
支持已声明常量与数值表达式；`/` 得到浮点数，整数维度计算使用 `//`。
允许位置维度、命名维度和混用，例如 `f32[8, feature=32]`。
命名轴不得重名；名称是轴的语义标记，不是动态长度变量，也不触发自动转置。

`linear(out=M)` 的 `out` 必填，省略的 `in` 从输入最后一维推导；
`linear(in=N, out=M)` 仍可显式检查 N。它只改变最后一维的大小，保留前面的维度，
同一层的权重沿前面的所有位置共享。已有的可选轴名会保留，包括最后一维的名称；
未命名轴不会因 linear 自动获得名字。不要把转置后的最后一维误当成原来的 feature。

## 重排、拆分与合并

`rearrange("输入模式 -> 输出模式", axis=SIZE, ...)` 没有可训练参数，
通过 reshape 和转置保持全部元素，不执行归约。模式不包含 batch。

| 输入与调用 | 输出 |
| --- | --- |
| `[8,32]`，`rearrange("time feature -> feature time")` | `[32,8]` |
| `[8,32]`，`rearrange("time (head width) -> head time width", head=4)` | `[4,8,8]` |
| `[4,8,8]`，`rearrange("head time width -> time (head width)")` | `[8,32]` |

空格分开轴，括号将若干轴组合成一维。每个输入组合至多有一个未知大小：
第二行提供 head=4，因此 width=32//4=8；32 必须能整除 4。
额外大小参数必须为输入模式里出现的轴提供正整数；已知大小必须与实际输入匹配。
每侧轴名只出现一次，rearrange 两侧必须包含相同的轴。
拆成 head 不会自动执行注意力，也不代表每个 head 拥有独立参数。

简单轴名传播并参与一致性检查。输入已知为 `[time=8,feature=32]` 时，
`"feature time -> time feature"` 的输入名字不匹配，不能借此悄悄重命名。
输入组合 `(head width)` 的外层名字未知，因此可拆解一个名为 feature 的维度。
输出组合 `(head width)` 的名字也未知，不自动恢复为 feature，shape 显示为 `[time=8,32]`。
后续简单模式可以为未知维度提供名称。未知名称不表示未知大小；数值形状仍然严格检查。
输出端口和分支合并也检查双方已知的简单轴名；未知名不构成名称冲突。

## 归约

`reduce("输入模式 -> 输出模式", op=mean, axis=SIZE, ...)` 接收 f32。
`op` 必填，仅支持 `mean` 或 `sum`，没有默认值。输出模式省略的轴被归约；
保留轴可重排或组合，轴大小推导规则同 rearrange。
允许不省略任何轴，例如 `reduce("time feature -> feature time", op=mean)`
只重排为 `[32,8]`，不求均值；op 仍须显式给出。

- `reduce("time feature -> feature", op=mean)`：`[8,32] -> [32]`，对每个特征平均 8 个时间步。
- `reduce("time feature -> feature", op=sum)`：形状相同，数值为各时间步之和。

这些操作可传递梯度，但本身没有权重。不允许归约成标量，必须保留至少一个样本轴。

## 门控、数据与检查

`add`、`mul`、`mix`、`where` 要求完整形状相同，不进行隐式广播；
门控分支应显式产生与值分支同形状的张量。`mix(g,a,b)` 表示 `g*a+(1-g)*b`，
需要 [0,1] 软门时自行使用 sigmoid。`where` 的两支都会计算，不是惰性分支。
重复调用的参数关系见 [数据流](dataflow.md)：repeat 是独立副本，recur 和 scan 共享权重。

训练行仍是扁平数值，最后一维变化最快。输入 `f32[8,32]` 每行需要 256 个输入值；
MSE 目标匹配全部输出元素，序列交叉熵则每个位置一个类别索引。
轴名称不会自动读取列名或改变数据顺序。注明真实/合成数据来源和训练状态，
不要把初始化、前向通过或梯度存在写成模型已学会任务。

```sh
weightc check examples/dimensions.wgt --shapes
```

命令检查源文件并打印每层输入/输出契约，包含展开后的内部层；不会执行训练。
使用仓库虚拟环境时可运行：

```sh
PYTHONPATH=src .venv/bin/python -m weightlang.cli check examples/dimensions.wgt --shapes
```

目前不支持省略号、模式中的数字轴、嵌套轴组、插入新轴、隐式广播、动态形状或标量输出。
端口中的 `f32[...]` 只能在说明文字中表示任意形状，不能把省略号写进可执行源文件。
更多可执行语法见 [语言说明](language.md)、[表达式与默认参数](syntax-extension.md)。

## Named-axis shorthand (0.3, unreleased)

`mean(axis=time)` and `sum(axis=time)` reduce one named axis and retain the others
in order. `transpose(feature, time)` lists every input axis exactly once. All
input axes must be named; unknown/duplicate axes and scalar reductions fail.
Reductions require f32; transpose also accepts i64. These lower to existing
`reduce`/`rearrange` patterns, so artifacts and runtime operators are unchanged.
`linear(128)` is equivalent to `linear(out=128)`; its sole positional argument
may be a numeric constant/expression. Use named arguments for additional options.
See [the complete example](../examples/syntax_sugar.wgt).


具名多输入/输出、bool 条件和显式 scan 初态已加入本地 0.3 静态图。
返回字典与固定形状检查、制品 v4 及限制见 [端口契约](dataflow.md#named-tensor-boundaries-and-explicit-scan-state-03-localunreleased)。
符号维度、KV cache 图端口和多状态循环仍未完成。

## Bounded symbolic dimensions (0.3, local/unreleased)

`dim T min=1 max=16` declares a runtime integer dimension, before all input/output
ports. Both bounds are required inclusive integer literals; there is no default.
Names cannot conflict with constants or other dimensions. Values must satisfy
`1 <= min <= max <= 2147483647`. Each symbol must appear **directly** on at least
one input axis (for example `input x: f32[T,4]`) so the runtime can bind it without
solving equations. Other axes can use integer expressions such as `P + T`.
All repeated occurrences must agree across ports. No dtype conversion or implicit
broadcasting is introduced.

`require LEFT OP RIGHT` adds a constraint; it requires declared dimensions.
`OP` is `==`, `!=`, `<`, `<=`, `>` or `>=`. Expressions can contain integer
constants, previously declared `const` names, declared dimension names, parentheses,
and `+ - * // %`. `/` is rejected; divisors must be positive. Arithmetic uses
integers bounded to magnitude 2147483647 and expression depth at most 16.
There is no `eval`, Python call, tensor-value-dependent shape or arbitrary solver.

Examples of expressions (declare all symbols/constants first):

```wgt
const HEADS = 6
const KV_HEADS = 2
dim T min=1 max=16
dim P min=1 max=8
require HEADS % KV_HEADS == 0
require P + T <= 20
input tokens: i64[T]
input previous: f32[P,4]
input mask: bool[P+T]
```

This fragment declares lengths only; it does not implement attention or KV cache.
This fragment uses positive P. Use `dim P min=0 max=N` for an empty initial cache;
see [composition contracts](composition.md) for executable cache ports.

The compiler exhaustively checks every admissible integer assignment using the
existing static graph checker. It rejects an empty legal domain and an architecture
that is valid only at its first length. Dimensions may not change parameter layout;
operator/block configuration remains compile-time constants. Inferred linear input
width is fixed by the first assignment and checked against all subsequent ones.

This deliberately limited verifier allows at most 8 dimensions, 64 constraints,
4096 combinations in the **unfiltered** Cartesian product, and 200000
`combination × expanded layer node` checks. Narrow declarations when exceeding
these limits; constraints do not bypass the raw-domain budget. This is exhaustive
bounded verification, not a general symbolic proof or a performance claim.

Runnable initialized example: [variable_streaming.wgt](../examples/variable_streaming.wgt).

```python
import torch
import weightlang
model = weightlang.load("examples/variable_streaming.wgt")
x, initial = torch.randn(2, 6, 2), torch.zeros(2, 4)
whole = model(samples=x, old=initial)
first = model(samples=x[:, :2], old=initial)
second = model(samples=x[:, 2:], old=first["new"])
torch.testing.assert_close(torch.cat([first["states"], second["states"]], 1), whole["states"])
second["states"].square().mean().backward()
```

All symbolic models use named dictionary returns, even with a single input/output.
Runtime binds symbols from actual input shapes, checks bounds and constraints,
validates every port, then executes the graph. Invalid inputs fail before any
layer executes. A single sample is internally given a batch axis so operators
such as attention follow the same behavior as a batch of one.

Artifacts use **version 5**. `interface.dimensions` stores bounds, `constraints`
stores integer comparison trees, and every port stores `shape_expr`. Numeric
`shape` and internal layer records show the first legal assignment in alphabetical
symbol order. They are representative shapes, not the full runtime contract.
Python and Rust independently recheck all assignments on load and reject altered
representative records or metadata downgraded to an older format. Old formats
remain readable. Saving Python weight updates, standalone Python export (including
`INTERFACE`) and structural decompilation preserve the expressions and constraints.
`check --shapes` displays both symbolic ports and labeled representative layer shapes;
Rust graph output shows symbolic port labels.

Current restrictions: no zero-length/scalar ports, optional dimensions, symbolic
operator arguments, or `rearrange`/`reduce`/`flatten` (including transpose/mean/sum shorthand) in bounded models. The latter
need dynamic axis plans rather than their fixed-shape implementations. Existing
meanpool/last and shape-preserving layers remain available. Source data/train and
inherit/derivation restrictions of named interfaces still apply. KV cache, MoE
auxiliary outputs, arbitrary block multi-state returns, and larger-domain symbolic
proofs remain subsequent work. The example is untrained and these features are
not released.


## 张量原子、标量与空端口（0.3 本地增量）

当前支持标量 `f32[]/i64[]/bool[]`、非负静态维度及 `dim P min=0 max=N`；
新增 matmul、broadcast、cat/stack/slice、transpose_axes、reduce_sum/mean/max、
gather/scatter_add、eq/lt 和 masked_softmax。归约可以输出标量，空 sum 为零；
全 mask 的 softmax 行输出和梯度为零。0.3 允许无参数计算模型。
Conv1d 支持 groups/dilation/causal 左填充。
完整签名、参数默认值、形状/dtype、空集合边界、限制和例子见 [primitives](primitives.md)。
这些规则替代前文历史增量中“所有端口正整数、禁止标量”的限制；
KV/MoE 的显式结果端口、具名 block 返回和多状态 scan 见下节；完整精度策略仍未完成。

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

原生适配器还支持 `dora/pissa/qlora`；完整方法语义、NF4 存储边界及训练例子见 [适配器](adapters.md)。
