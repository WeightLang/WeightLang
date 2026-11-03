# 固定维度与命名轴（wgt 0.3）

本文描述当前源码可执行的 `wgt 0.3` 语法，尚未发布。
完整小模型见 [dimensions.wgt](../examples/dimensions.wgt)：72 个可训练参数，
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

## 数据与检查

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
