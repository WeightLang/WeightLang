# WeightLang 语言与标准库参考

> 当前命令已拆分：`build` 生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例使用 `weightc legacy-build`；详见[分发包、元数据与迁移规范](bundles.md)。


本文集中维护 `.wgt` 的关键字、内置算子和标准库 block（可复用网络模块）。
依据当前仓库源码；**0.3 功能尚未发布，不代表已发布的 0.2 工具支持它们**。
这里的“函数”指算子或 block 调用，不是可执行任意代码的通用函数。
标准库只提供网络结构，不自带训练好的权重。

## 目录

- [基本写法与类型](#基本写法与类型)
- [关键字与语句](#关键字与语句)
- [内置算子](#内置算子)
- [连接、分支与循环](#连接分支与循环)
- [标准库函数](#标准库函数)
- [数据与训练参数](#数据与训练参数)
- [完整示例](#完整示例)
- [维护与实现依据](#维护与实现依据)

## 基本写法与类型

通常一行一条语句，`#` 开始注释，缩进属于上方的 `block` 或 `data`。
0.3 允许括号内跨行和行尾注释。字符串用双引号，不支持转义字符。
名字使用字母或下划线开头，后续可含数字；点号用于库别名或展开后的层路径。

| 写法 | 含义与约束 |
| --- | --- |
| `f32[N]` | 单个样本的浮点向量，0.1 起支持 |
| `f32[L,D]` | 长度 L、每项 D 个特征的序列，0.3 |
| `f32[C,H,W]` | 通道数 C、高 H、宽 W 的图片，0.3；不是 HWC |
| `bool[D1,...]` | 固定形状布尔端口；where 条件可用，不隐式转换为 f32；0.3 |
| `i64[L]` | 整数 token 序列，通常交给 embedding，0.3 |
| `f32[D1,D2,D3]` | 0.3 的固定多维浮点形状示意；维度填正整数或常量，各算子有各自维数限制 |
| `name=value` | 算子和 block 参数通常按名字传递；0.3 的 linear(out) 允许单个位置参数 |
| `dim T min=1 max=16` | 有界运行时整数维度，0.3；完整约束与限制见文末 |
| `require T % 2 == 0` | 有界整数约束，0.3；执行层前验证 |
| `const WIDTH = 32` | 0.3 编译期数值常量，不是运行时张量 |
| `+ - * / // %` | 0.3 数值表达式；支持一元正负号和括号；乘除余数优先于加减 |

形状不含 batch（一次处理的样本数），每一维必须是正整数。
`/` 得到浮点数，`//` 是向下取整除法；`8 / 2` 不能作为整数维度。
表达式可用于形状、算子参数、block 参数和默认值，不能替代训练参数或数据行里的数值字面量。
常量按声明顺序求值，固定端口维度引用已声明常量；有界符号维度另见文末；层参数在文件解析完成后解析。
block 参数覆盖同名常量，默认值可引用前面的参数。没有任意代码求值；有界运行时维度另见文末。

## 关键字与语句

这里按语法用途列词，不表示这些名字在所有上下文中一律保留。

| 关键字 / 写法 | 解释 | 版本 |
| --- | --- | --- |
| `wgt 0.3` | 文件首条语句，声明语言版本；接受 0.1、0.2、0.3 | 0.1+ |
| `model name` | 声明该文件唯一的模型 | 0.1+ |
| `input x: f32[N]` | 声明输入名称、类型、单样本形状 | 0.1+ |
| `output y: f32[M]` | 声明输出契约；默认匹配最终层，0.3 可用 from 显式选取张量 | 0.1+ |
| `layer name: expression` | 创建一个层并命名其输出；默认接前一层 | 0.1+ |
| `init seed=42` | 必须显式初始化；seed 是 `[0, 2**64)` 整数 | 0.1+ |
| `init file="init.wgt" seed=42` | 按外部 WGT 规则生成完整权重；seed 可省略，默认 0 | 0.3 |
| `init python="generate.py:initialize" seed=42` | 构建时调用 Python 权重生成器；函数名默认 initialize，seed 默认 0 | 0.3 |
| `data name:` | 定义数据集，后接缩进的数据行或文件选项 | 0.1+；选项 0.2+ |
| `train data=... loss=... optimizer=... lr=... epochs=...` | 声明一次训练；每个模型最多一条 | 0.1+ |
| `import body from "../weights/base" layer=body` | 从制品导入兼容的层权重；省略 layer 时使用目标同名路径 | 0.1+；点路径 0.2+ |
| `import std/nn` / `import std/nn as n` | 导入内置计算库，默认别名 nn；无需 add、联网或包锁；随编译器版本分发，不包含预训练权重 | 0.3；模型和库文件均可用 |
| `use std/nn` | 加载源码库，默认别名是末段名称 nn | 0.2+ |
| `use std/heads as h` | `as` 指定库别名；通过 h.classifier 调用 | 0.2+ |
| `use "./layers.wgt" as local` | 相对当前源文件加载本地库；不支持远程源码库 | 0.2+ |
| `block name(in, out):` | 定义可复用层组合，内部是缩进的 layer；不可为空或递归调用 | 0.2+ |
| `const NAME = expression` | 定义文件私有数值常量；use 不导出常量命名空间 | 0.3 |
| `block ff(dim, hidden=dim * 4):` | block 默认参数，必填参数必须在默认参数前 | 0.3 |
| `layer b: linear(in=4, out=4) from input` | `from` 指定整条层表达式的输入；也用于 import / inherit | 0.3 的张量引用 |
| `inherit from "../weights/base" sha256=...` | 用固定权重哈希继承整套模型权重；仍需声明架构和 init，不能与 import 混用 | 0.3 |
| `freeze body` | 冻结指定层子树或精确参数；`freeze all` 匹配全部参数 | 0.3 独立语句 |
| `unfreeze head` | 恢复指定参数参与训练；all 同样有效 | 0.3 |

`import ... freeze` 从 0.1 起可将导入层排除出训练；`sha256=` 可固定导入权重哈希。
导入要求展开后的结构兼容，不能只凭张量大小相同就交换网络层。
初始化先执行，然后导入，再应用独立 freeze/unfreeze，最后训练。
独立冻结规则按源码顺序执行，最后匹配的规则生效；需要 train，且最终不能冻结全部可训练参数。
冻结不会关闭 dropout，也不会切断前面层需要的梯度。

模型需要版本、model、input、output、至少一个带参数层和 init；train 可省略。
未训练模型只是已初始化结构，不能当成训练完成的预测模型。
库文件只允许 `wgt`、`use`、`block`、`const`，循环导入会报错。

## 可编程初始化

`init` 每个模型只能出现一次，`file` / `python` 互斥，路径相对模型源码。
自定义初始化使用 meta 构建和空内存分配，跳过真实随机默认权重生成；
之后依次执行 reuse/derivation/import、冻结和设备放置，不进入 forward。
`load` 不输出制品、不调用训练器或数据加载器，但会执行显式指定的初始化脚本。

初始化规则文件使用独立头 `wgt init 1`，不是模型或 block 库：

```wgt
wgt init 1
# weight 的两个轴是输出、输入；DCT 基无需随机采样，尚未训练。
tensor "*.weight" dct gain=1
# 初始偏置为零，避免引入整体平移。
tensor "*.bias" zeros
```

完整语句为 `tensor "PATTERN" METHOD [key=value ...]`。
名称匹配使用大小写敏感 glob，最后匹配规则生效；每条规则必须命中，
每个状态张量必须覆盖，共享参数的规则和值必须一致。
支持 `zeros`、`ones`、`constant(value=0)`、`eye(gain=1)`、`dct(gain=1)`、
`orthogonal(gain=1)`、`xavier_uniform(gain=1)`、`xavier_normal(gain=1)`、
`normal(mean=0,std=1)`、`uniform(low=0,high=1)`；此处括号仅表示参数签名，
实际文件使用空格分隔的 `key=value`。参数须有限，std≥0、low≤high；
eye/dct 要求矩阵，orthogonal/xavier 要求秩≥2。DCT 对宽矩阵生成正交行，
高矩阵生成正交列，按 gain 缩放；它不是训练得到的知识。
WGT 同时支持 `algorithm name(arg, ...):` 自定义算法和 `tensor "PATTERN":` 算法体。
可使用 let/赋值/索引写入、if/elif/else、for/while、break/continue、return，
数值和张量表达式（含广播、切片、`@` 矩阵乘法）及算法调用；无需 Python 脚本。
上下文有 shape/rank/seed/pi/dim(axis)，可复用函数有独立局部作用域。
原语覆盖张量创建、随机采样、形状变换、归约、逐元素数学、qr/svd/solve。
完整签名、默认值和约束见[算法原语表](initialization.md#tensor-primitives)。
每个独立目标最多 100000 求值步骤、64 调用层；共享参数只执行一次。
check 仅解析，不证明动态分支或收敛；运行时验证最终形状/类型/有限性。
不支持 host import、属性调用、嵌套文件引用或模型推理调用。
完整的纯 WGT 算法例子见 [init.wgt](../examples/initialization/init.wgt)。

Python 签名为 `initialize(context: InitializationContext) -> dict[str, torch.Tensor]`。
context 含 `seed` 和不可变 `tensors` 元组，各项含 `name`、`shape`、`dtype`、
`alias_of`（自身或之前出现的共享参数名）。返回全部同名 CPU 稠密张量，
形状/类型须精确相等、数值有限、共享值相等；不能返回活模型或省略参数。
回调在 no_grad 和隔离的 Torch RNG 中运行，不隔离 Python/NumPy RNG，
也不是脚本沙箱；不应把数据加载或训练藏在初始化中。

check 读取规则但不执行 Python；规则覆盖、秩和共享值在构建时检查。
保存后加载制品只读权重，不再运行生成器；manifest.init 记录方法、seed、
源文件 SHA256，不保证脚本间接依赖或跨硬件复现。生成参数使用既有 edited 来源，
不声称已训练；`initial_state=` 不能与自定义 init 混用。
源码包支持包内 WGT 规则；Python 初始化模型须显式选择已生成制品打包。
完整接口、约束和 Python 示例见 [初始化专题](initialization.md)，
可运行例子见 [WGT 模型](../examples/initialization/model.wgt) 和
[Python 模型](../examples/initialization/python.wgt)。

## 内置算子

表中 N/M/D/C/O/K/S/L/V 均为正整数；以下稳定写法显式提供必填参数。
下表除 embedding 接收 i64 外，算子接收 f32；下文 rearrange 也支持 i64。激活函数保持形状。

| 调用 | 功能、参数与输出 | 版本 |
| --- | --- | --- |
| `linear(in=N, out=M)` | 带偏置全连接；最后一维 N → M，前面维度不变 | 0.1+ |
| `relu` | 负值变成 0 | 0.1+ |
| `tanh` | 双曲正切，将值压到 (-1,1) | 0.1+ |
| `sigmoid` | 将值压到 (0,1)，常用于二分类或门控 | 0.1+ |
| `gelu` | 平滑激活，标准 MLP 默认采用它 | 0.2+ |
| `silu` | x × sigmoid(x)，用于平滑门控 | 0.2+ |
| `softmax` | 在最后一维归一化为概率；cross_entropy 训练输出不能先调用它 | 0.2+ |
| `identity` | 原样传递输入 | 0.2+ |
| `layernorm(dim=D)` | 归一化最后一维；需宽度 D，带可训练缩放和偏置 | 0.2+ |
| `dropout(p=P)` | 训练时随机丢弃，0 ≤ P < 1；推理时原样传递 | 0.2+ |
| `embedding(vocab=V, dim=D)` | i64[L] → f32[L,D]，索引范围 [0,V)；V 是词表大小 | 0.3 |
| `position_embedding(dim=D, max_len=M)` | 给 f32[L,D] 加上可学习的位置向量；L ≤ M，形状不变 | 0.3 |
| `self_attention(dim=D, heads=H, causal=0)` | f32[L,D] 自注意力；D 必须整除 H；causal=1 禁止看未来位置，0 为双向 | 0.3 |
| `conv2d(in=C, out=O, kernel=K, stride=1, padding=0)` | f32[C,H,W] → f32[O,H′,W′]；K 为正方形核边长，stride 为步长，padding 为非负边缘填充 | 0.3 |
| `maxpool2d(kernel=K, stride=S)` | 在图片每个通道做最大池化，不填充，通道数不变 | 0.3 |
| `globalavgpool2d` | f32[C,H,W] → f32[C]，对每个通道的空间维求平均 | 0.3 |
| `meanpool` | f32[L,D] → f32[D]，沿序列求平均 | 0.3 |
| `last` | f32[L,D] → f32[D]，取最后一个序列位置 | 0.3 |
| `flatten` | 将单样本所有维度展平成一个向量，不合并 batch | 0.3 |

卷积和池化输出空间尺寸为 `floor((size + 2*padding - kernel)/stride) + 1`，
结果必须大于 0；目前使用正方形核，dilation=1，不提供 groups 参数。
注意力没有 cross-attention、padding mask、KV cache 或 RoPE。

### 命名轴与维度变换（0.3，尚未发布）

当前源码支持以下接口；完整解释见 [固定维度与命名轴](dimensions.md)，
可检查示例见 [dimensions.wgt](../examples/dimensions.wgt)。示例只初始化可训练参数，没有训练。

| 写法 | 参数、契约与用途 |
| --- | --- |
| `f32[time=8, feature=32]` | 固定正整数维度附带语义名称，可引用此前常量；名称不能重复 |
| `f32[8, feature=32]` | 允许位置维度和命名维度混用；未命名不等于动态大小 |
| `linear(out=M)` | out 必填正整数；in 省略时推导输入最后一维，也可显式指定并检查；只改变最后一维大小，保留已有轴名，跨其他位置共享权重 |
| `rearrange("输入模式 -> 输出模式", axis=SIZE, ...)` | 重排、拆分或合并轴；可选大小参数为输入模式中的轴提供正整数，无参数权重；可处理 f32 或 i64 |
| `reduce("输入模式 -> 输出模式", op=mean, axis=SIZE, ...)` | 输入必须为 f32；省略的轴被归约，op 必填且只能 mean 或 sum；大小参数规则同 rearrange |

模式中的空格分隔轴，括号组合轴；每侧轴名不可重复。
`rearrange("time feature -> feature time")` 将 `[8,32]` 转为 `[32,8]`。
`rearrange("time (head width) -> head time width", head=4)` 将 `[8,32]` 转为 `[4,8,8]`：
每个输入组至多推导一个未知大小，并要求整除。rearrange 必须保留全部轴，不能插入新轴。
`reduce("time feature -> feature", op=mean)` 对时间求均值，`[8,32] -> [32]`；sum 则求和。

简单轴名传播并检查已知名称冲突；输入和输出的组合轴外层名称均视为未知。
例如合并 `(head width)` 后只知道大小 32，不会自动命名为 feature；后续简单模式可以赋名。
未知名称不跳过大小检查，已知轴不能任意改名。linear 保留已有可选轴名，包括最后一维名称，
但不会给未命名轴自动命名。reduce 允许不删除任何轴：
`reduce("time feature -> feature time", op=mean)` 只重排，不求均值；op 仍必填。
端口、模式都不包含 batch。
不支持省略号、数字轴模式、嵌套分组、隐式广播、动态形状或归约成标量。
表中的省略参数仅为签名记法，不能把 `...` 写入程序。
`weightc check examples/dimensions.wgt --shapes` 打印逐层契约，不执行训练。

## 连接、分支与循环

| 写法 | 解释与约束 | 版本 |
| --- | --- | --- |
| `residual(block(...))` | 输入 + block 输出；0.3 也可包至少两步的管线；完整形状与类型必须相同；不能直接包单个 linear | 0.2+ |
| `linear(in=4, out=8) -> relu` | 从左到右连接；尾部 from 作用于整个管线 | 0.3 |
| `add(a, b, ...)` | 至少两个具名张量逐元素相加 | 0.3 |
| `mul(a, b, ...)` | 至少两个具名张量逐元素相乘 | 0.3 |
| `concat(a, b, ...)` | 至少两个张量沿最后一维拼接，其余维度一致 | 0.3 |
| `mix(g, a, b)` | g*a + (1-g)*b；不会自动把 g 限制到 [0,1] | 0.3 |
| `where(c, a, b)` | c>0 选 a，否则选 b；两支都会执行，条件不获得梯度 | 0.3 |
| `repeat(N, body)` | 展开 N 份独立参数，层名为 step0…stepN-1 | 0.3 |
| `recur(N, body, until=stop, compact=0)` | 共享参数，最多 N 次；可选 until 指向 body 内停止概率层；省略则固定 N 次。body 保持完整形状与类型 | 0.3 |
| `route(branch0, branch1, ...) from selector,x` | 标量 i64 路由，或两个分支时用 bool；只计算被选样本，分支保持 f32 状态形状与轴 | 0.3 |
| `ponder(N, body, until=stop)` | 固定 N 步共享计算，返回 output/states/probabilities/halt_weights/expected_steps；停止分布可微 | 0.3 |
| `scan(hidden=H, cell(...))` | f32[L,D] → f32[L,H]；按序列执行共享参数的 cell | 0.3 |

合并值接收 f32，where 条件另可接收 bool，不隐式广播；add/mul/mix/where 的所有输入形状必须完全相同。
它们必须占据具名层，不能直接作为管线内的步骤。
`input` 指当前模型或 block 的原始输入；模型输入端口名称也可引用。
引用只能指向同一作用域内更早的层，不能向前引用、自引用或穿透其他 block。
`input` 是保留层名；scan cell 内的 `state` 也是保留名。

scan cell 的 `input` 是当前时间步 f32[D]，`state` 是上一时刻 f32[H]，
cell 输出必须为 f32[H]。默认初始状态为零；`scan(...) from sequence, initial` 可显式传入
f32[H] 初态，batch 必须一致。全部时间步参与反向传播；没有隐式持久状态或可变长度支持。
repeat/recur 次数为 1…1024；最多展开 10,000 个节点、嵌套 64 层 block。
`seq` 是编译器展开后的内部节点，不是供用户调用的函数。

### 概率门控提前退出（0.3，本地实现，尚未发布）

完整签名 `recur(N, body, until=NAME, compact=0)`；N 为 1…1024 的静态整数，body 为算子、
管线或 block；until 可省略，无默认停止门，省略保持原固定次数语义。
NAME 必须是展开后循环体的直接具名层，不可跨作用域或引用 input。
使用 until 时循环状态必须为 f32，停止层必须输出单样本 `f32[1]`（不含 batch）。
状态的完整形状、dtype 与轴名仍保持不变；循环体最后一层是新状态。

每轮先执行 body，再读取停止概率 p，对各样本独立采样 u∈[0,1)，u<p 即停止。
p 必须有限且在 [0,1]，否则运行时报错，不自动 sigmoid 或截断。
至少执行一轮，达到 N 强制结束；p=1 第一轮退出，p=0 执行至上限。
训练与 eval 均采样；Python `torch.manual_seed` 控制随机序列，init seed 不会重置每次前向 RNG。
已停止样本保持其最后结果；全部停止立即退出。部分停止时仍计算完整 batch，
默认 compact=0 没有活动样本压缩；compact=1 只计算仍活动的样本，且要求 until。
compact 只允许 0/1，压缩改变随机数消费顺序；不保证两种模式同 seed 的路径相同。状态不跨调用保存。

退出判断不传梯度；被保留结果沿已执行更新反传。仅用于停止判断的门参数不会由普通
输出损失学到停止策略，需要 Python 侧额外监督或梯度估计；不内置 ACT、REINFORCE
或可微停止估计。门可单独训练或由其他可微计算共享参数。声明不代表训练完成。

```wgt
wgt 0.3
model early_exit
input x: f32[feature=8]
output y: f32[feature=8]
block think(width):
  # [8] -> [8] 的隐状态更新，各轮共享权重。
  layer next: linear(width) -> tanh from input
  # 每个样本一个停止概率：[8] -> [1]。
  layer stop: linear(1) -> sigmoid from next
  # 最终输出保留新状态，不输出概率。
  layer result: identity from next
# 最多 16 轮，随机退出不可微；本例只初始化，没有训练。
layer reasoning: recur(16, think(width=8), until=stop)
init seed=42
```

完整例子见 [examples/early_exit.wgt](../examples/early_exit.wgt)。支持 Python 前向、
制品保存重载、独立 Python 导出及结构反编译；Rust 校验概率层契约，不执行张量推理。

### 条件路由、可微停止分布与空间组合

`route`、`ponder`、`recur compact=1` 的完整签名、参数、类型/形状、batch、
共享权重、梯度和训练边界，以及可直接运行的完整例子，见
[条件计算与多尺度组合](adaptive-compute.md)。

- route 至少两个单输入单输出分支，selector 为 i64[]，范围 0…分支数−1；
  两分支可用 bool[]，false/true 对应分支 0/1。选择不传梯度，未选分支不执行。
- ponder 的 N 为 1…1024，until 必填；f32 状态保持完整形状和轴，停止层为 f32[1]。
  始终执行 N 步，最后一轮接收剩余概率。output 为最后状态，states 为 [N,*状态形状]，
  probabilities/halt_weights 为 [N,1]，expected_steps 为标量 f32[]；所有值可通过具名端口返回。
  不隐式训练、采样或添加损失，N=1 时停止分布恒为 1。
- resize2d(height,width,mode=0,align_corners=0)：必填目标正尺寸；mode=0 最近邻/1 双线性，
  align_corners=0/1，最近邻须为 0。输入非空 f32[C,H,W]，输出 f32[C,height,width]。
- pad2d(left,right,top,bottom,mode=0)：必填非负四边距；mode=0 零填充/1 反射/2 复制；
  反射每侧小于相应尺寸，输出空间尺寸加对应边距，不支持负数裁剪。
- avgpool2d(kernel,stride,padding=0,count_include_pad=1)：必填正窗口和步幅，padding 在
  0…floor(kernel/2)；count_include_pad=0/1，floor 输出且必须非空，无 ceil_mode。
  pad2d/avgpool2d 同样只接收非空 f32[C,H,W]。调用配置用命名参数，不支持元组尺寸。

这些能力均为 0.3 本地实现，未发布。空间算子无参数；结构示例只初始化。
现有多输入、多结果 block 和 scan_state 的状态契约仍见 [组合接口](composition.md)。

### 结构简写（0.3，本地实现，尚未发布）

这些写法在编译时展开为现有算子，不增加运行时算子，也不附带训练权重。

| 完整签名 | 用途、参数与约束 |
| --- | --- |
| `linear(out)` | 唯一位置参数 out 必填，正整数编译期表达式；等价于 `linear(out=out)`，从输入最后一维推导 in，保留其他轴；支持 f32。不能混用位置参数和命名参数，需要 in/bias 等选项时使用完整命名形式 |
| `left + right` | 等价于 `add(left, right)`；两个名字必填，引用同作用域中更早的张量或 input；f32、完整形状和已知轴名一致，不广播 |
| `left * right` | 等价于 `mul(left, right)`，逐元素乘法；输入约束同加法，不是矩阵乘法 |
| `mean(axis=NAME)` | axis 必填，无默认值；沿指定轴求均值，保持其余轴顺序；输入 f32，所有轴必须有名称，NAME 必须存在，不允许归约成标量 |
| `sum(axis=NAME)` | 参数与 mean 相同，沿指定轴求和 |
| `transpose(AXIS, ...)` | 按位置列出全部输入轴名，每个恰好一次；所有输入轴必须有名称，至少一个轴；支持 f32/i64，输出按所列顺序重排；不拆轴、不合轴，无默认参数。省略号是文档签名记法 |
| `residual(OP -> OP -> ...)` | 至少两个管线步骤，无默认参数；输入加管线输出，要求 f32、完整形状及已知轴名一致；管线参数路径为 `层名.step0`、`层名.step1` 等 |

加乘简写只接受两个名字，不接受数字常量、括号或链式表达式；复杂计算用多个命名层。
与 add/mul 一样，它们不能嵌入管线，也不能额外指定 from。
数值参数内的 `+`、`*` 仍是编译期算术，不引用张量。
mean/sum/transpose 为内置简写名称；自定义库中的同名 block 可通过库别名调用。

例如输入 `f32[time=3, feature=4]` 时，`mean(axis=time)` 展开为
`reduce("time feature -> feature", op=mean)`；`transpose(feature, time)` 展开为
`rearrange("time feature -> feature time")`。简写和对应显式写法的层名、参数路径一致；
把已有命名 block 改成匿名管线仍会改变内部路径，不保证旧权重可直接导入。
完整可运行例子见 [syntax_sugar.wgt](../examples/syntax_sugar.wgt)，只初始化、未训练。

## 标准库函数

先 `import std/库名`（0.3，也兼容 `use std/库名`），再 `库名.函数(参数名=值)`；也可通过 `as` 改别名。
以下列出随源码提供的 block 签名；block 参数允许有限数值，维度/次数必须为正整数；
概率、eps 和 0/1 开关按最终算子约束检查。新增 hf_decoder 签名见文末迁移章节。
in/out 是输入/输出特征宽度，hidden/h1/h2 是隐藏层宽度；图像库中 in/out 是通道数。
标准库跟随编译器版本，展开后的层拥有权重；库本身不含权重。

### std/nn（0.2+）

| 签名 | 功能 |
| --- | --- |
| `mlp(in, hidden, out)` | linear → gelu → linear |
| `mlp_relu(in, hidden, out)` | linear → relu → linear |
| `mlp_tanh(in, hidden, out)` | linear → tanh → linear |
| `mlp2(in, h1, h2, out)` | 两个隐藏层：linear → gelu → linear → gelu → linear |
| `dense_gelu(in, out)` | linear → gelu |
| `dense_relu(in, out)` | linear → relu |
| `dense_tanh(in, out)` | linear → tanh |

### std/heads（0.2+）

| 签名 | 功能 |
| --- | --- |
| `classifier(in, classes)` | 全连接输出 classes 个原始分类分数（logits），配 cross_entropy |
| `binary(in)` | 全连接输出 1 维后 sigmoid，配 bce |
| `regressor(in, out)` | 全连接输出 out 维实数，配 mse |

### std/blocks（0.2+）

| 签名 | 功能 |
| --- | --- |
| `prenorm_ff(dim, hidden)` | layernorm → mlp(dim, hidden, dim)，保持特征宽度 |
| `residual_mlp(dim, hidden)` | 输入 + prenorm_ff 输出，保持完整形状 |

### std/vision（0.3）

| 签名 | 功能 |
| --- | --- |
| `conv_relu(in, out, kernel)` | conv2d → relu；步长 1、padding 0 |
| `conv_pool(in, out, kernel, pool)` | conv_relu → maxpool2d；池化核与池化步长均为 pool |
| `classifier(channels, classes)` | globalavgpool2d → linear(channels, classes) |

kernel/pool 为正方形窗口边长。需要自定义 stride/padding 时直接用内置 conv2d；
不要给 vision.conv_relu 传它不支持的参数。
vision.classifier 接图片特征，heads.classifier 接最后一维特征，两者输入契约不同。

### std/transformer（0.3）

| 签名 | 功能 |
| --- | --- |
| `tokens(vocab, dim, max_len)` | token embedding + 可学习位置编码，i64[L] → f32[L,dim] |
| `attention(dim, heads)` | layernorm → 双向 self_attention |
| `causal_attention(dim, heads)` | layernorm → 因果 self_attention |
| `feed_forward(dim, hidden)` | layernorm → mlp(dim, hidden, dim) |
| `encoder(dim, heads, hidden)` | 带残差的双向 attention，再接带残差的 feed_forward |
| `decoder(dim, heads, hidden)` | 带残差的 causal_attention，再接带残差的 feed_forward；不是编码器交叉注意力 |
| `swiglu(dim, hidden=dim * 4)` | 两个线性投影，其中一支 SiLU；逐元素相乘后投影回 dim |
| `gated_decoder(dim, heads, hidden=dim * 4)` | 因果注意力残差 → 归一化 → SwiGLU → 残差相加 |

heads 是注意力头数，必须整除 dim；max_len 是最大位置长度，vocab 是词表大小。
模块一般保持 f32[L,dim]；tokens 负责把整数序列转为特征。

### std/recurrent（0.3）

| 签名 | 功能 |
| --- | --- |
| `gru_cell(features, hidden)` | 单步 GRU，features 为当前输入宽度，hidden 为状态宽度；用于 scan 内部 |
| `gru(features, hidden)` | 包装 scan，f32[L,features] → f32[L,hidden] |

gru_cell 需要 scan 提供 state，不能当普通顶层层独立使用。
这里的重置门在候选隐藏状态投影之前应用，与 PyTorch reset-after GRU 不同，权重不可直接互换。

## 数据与训练参数

行格式为 `[输入数值] -> [目标数值]`；多维输入按 C/H/W 等声明顺序展开，最后一维变化最快。
CSV/TSV 前若干列为输入，其余为目标；支持带引号的数字字段。
0.3 支持独立 `.wgt`、JSON/JSONL、NPZ 和 Python 数据加载，完整格式与回调契约见[数据加载](datasets.md)。

| 参数 | 解释 |
| --- | --- |
| `file="train.csv"` | 相对源文件的数据路径；必须同时提供 sha256 |
| `sha256=...` | 数据文件原始字节的 64 位小写十六进制 SHA256，含被跳过的表头 |
| `skip=0` | 跳过开头若干原始文本行，非负整数，仅 CSV/TSV/JSONL |
| `format=csv/tsv/json/jsonl/npz/wgt` | 0.3；默认按文件扩展名推断；数据全部载入内存 |
| `dataset=name` | 0.3；仅用于 `.wgt`，选择内部数据集；默认与当前 data 名相同 |
| `python="file.py:load"` / `entrypoint="name"` | 0.3 数据加载器；与 file 互斥；函数名默认 load；入口点组为 weightlang.data_loaders |
| `license="..."` / `source="..."` | 数据许可证与来源标注；内联数据也可用 |
| `data=name` | 训练数据集名，必填 |
| `loss=mse` | 均方误差；目标形状与输出一致 |
| `loss=bce` | 二元交叉熵；末层 sigmoid，目标为二元值 |
| `loss=cross_entropy` | 0.2+；最后一维为类别数，目标为合法整数类别索引，输出不能先 softmax/sigmoid |
| `optimizer=sgd` / `adam` / `adamw` | 默认 backprop 训练必填；adamw 从 0.2 起支持 |
| `lr=0.01` | 正学习率，默认 backprop 训练必填 |
| `epochs=100` | 正整数训练轮数，必填 |
| `batch=N` | 0.2+，正整数；省略或不小于数据量时使用全量批次 |
| `validate=name` | 0.2+，训练后评估另一数据集；不能与训练数据集同名，数据是否重叠由作者保证 |
| `weight_decay=0` | 0.2+，非负权重衰减，默认 0 |
| `trainer=backprop` | 0.3，默认反向传播；其他名字选择已注册训练器 |

自定义 trainer 必填 data/loss/epochs，可选 batch/validate/lr（lr 默认 0.01），
不接受 optimizer/weight_decay；名字能解析不代表训练器已注册，见[训练器接口](trainers.md)。
0.3 还可用 `python="file.py:train"` 或 `entrypoint="name"` 替代 trainer，函数名默认 train，
入口点组为 weightlang.trainers；其他参数沿用自定义训练器规则。脚本路径相对模型文件。
数据加载器 `load(DataContext)` 返回二维 inputs/targets 字典；训练器 `train(TrainingContext)`
原地更新参数，返回有限数值指标字典或 None。完整签名、上下文属性和限制见[数据加载](datasets.md)。
check 不执行 Python；实际训练前加载并校验数据，`--no-train` 跳过加载器和训练器。
可运行的分离数据例子见 [model.wgt](../examples/external_data/model.wgt) 与
[data_x.wgt](../examples/external_data/data_x.wgt)，脚本例子见
[python.wgt](../examples/external_data/python.wgt)。这些是当前本地实现的合成训练演示，尚未发布。
序列交叉熵 f32[L,V] 每行需要 L 个类别目标。seed 不保证跨硬件或跨 PyTorch 版本结果一致。

## 完整示例

### CNN：两类 4×4 图片

下面是可解析的完整训练源；只有两条演示数据，不代表泛化能力。

```wgt
wgt 0.3
use std/vision
model tiny_cnn
input image: f32[1,4,4]
output logits: f32[2]
layer features: vision.conv_relu(in=1, out=4, kernel=2)
layer head: vision.classifier(channels=4, classes=2)
data lines:
  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0] -> [0]
  [1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0] -> [1]
init seed=42
train data=lines loss=cross_entropy optimizer=adam lr=0.03 epochs=80
```

### 自定义 block、常量与库别名

```wgt
wgt 0.3
use std/nn as n
use std/heads as h
const WIDTH = 8
model custom
input x: f32[WIDTH]
output y: f32[3]
block feature(dim, hidden=dim * 2):
  layer body: n.mlp(in=dim, hidden=hidden, out=dim)
layer body: residual(feature(dim=WIDTH))
layer head: h.classifier(in=WIDTH, classes=3)
init seed=42
```

### GRU：序列分类结构

```wgt
wgt 0.3
use std/recurrent as r
model sequence_classifier
input x: f32[8,4]
output y: f32[3]
layer memory: r.gru(features=4, hidden=16)
layer head: last -> linear(in=16, out=3)
init seed=42
```

后两个例子只初始化，不训练。更多示例见 [CNN](../examples/cnn.wgt)、
[Transformer](../examples/transformer.wgt)和[分支与循环](dataflow.md)。
`weightc std` 可查看当前安装版本实际提供的标准库签名。

## 维护与实现依据

本文是面向使用者的统一查询入口；内部 Python API 和 CLI 子命令分别见
[Python API](python.md)和[wgt 工具](wgt.md)，不混作语言关键字。

- 语句、算子、参数、默认值与版本检查：[syntax.py](../src/weightlang/syntax.py)。
- 数值表达式：[expressions.py](../src/weightlang/expressions.py)。
- 标准库签名和组合：[std/](../src/weightlang/std/)。
- 运行时行为：[modern.py](../src/weightlang/modern.py)、[dataflow.py](../src/weightlang/dataflow.py)。
- 专题规则：[语言说明](language.md)、[标准库与权重包](stdlib.md)、[表达式](syntax-extension.md)、[冻结与继承](weight-iterations.md)。

新增、删除或修改上述公开能力时，同一次变更更新本文对应条目和示例；
维护流程由根目录 [AGENTS.md](../AGENTS.md) 约束。它要求开发助手主动同步，
不是后台自动生成器；文档未验证的能力不得标成已实现或已发布。

## 导出 Python 源文件

`weightc export-python SOURCE -o NEW.py`：SOURCE 为 `.wgt` 时只导出结构，不运行训练或权重导入；
SOURCE 为已构建模型目录时，导出独立 PyTorch 类和同名 `.safetensors` 权重。
生成代码无需安装 WeightLang。使用 `Model()` 创建新初始化结构；模型目录导出额外提供
`load_model(weights_path=None, device="cpu")`，核对权重哈希并恢复冻结参数。
输入必须显式带 batch 维；不会导出训练算法、优化器状态或设备分片。
Python API 为 `export_python(source, output)`，拒绝覆盖现有文件。
完整用法及支持范围见[Python 导出](python-export.md)。

## 只构建或迁移，不训练

`weightc legacy-build model.wgt --no-train`：保留源文件中的 train 声明，但此次只初始化、
导入/继承或迁移权重，并保存模型；不读取训练/验证 CSV、不加载训练器、不更新参数。
结构、形状和冻结选择器仍检查，训练数据与损失相关的检查跳过；可以冻结全部参数。
默认从权重版本库 HEAD 自动复用同名且兼容的参数，新参数保持初始化。
扩大 linear/embedding 等受支持层时，用 `--reuse overlap` 明确允许复制重叠部分；
重命名用 `--reuse-map 新名=旧名`。架构迁移不保证预测效果保持不变。
`--no-store` 没有自动 HEAD，需要 `--reuse-artifact 路径` 指定基线；仍不允许隐式覆盖。
Python 编译接口对应 `no_train=True`。输出记录 `training: null` 和 `build.no_train: true`，
区分本次未训练、上游训练权重和新增随机参数。详见[架构迁移](architecture-reuse.md#buildmigrate-without-training)。

## 原生权重调试（0.3，本地实现，尚未发布）

无需增加 `.wgt` 关键字；调试器复用源码的结构、数据、训练与冻结声明。
完整可运行示例见 [debug_weights.wgt](../examples/debug_weights.wgt)。

```sh
weightc legacy-build --no-store examples/debug_weights.wgt
weightc trace examples/debug_weights.wgt --weights weights/debug_weights --watch fit
weightc debug examples/debug_weights.wgt --weights weights/debug_weights --steps 3 --json
```

命令签名（`wgt` 驱动转发同样的参数）：

```text
weightc trace SOURCE [--weights ARTIFACT] [--sample JSON_FILE]
  [--watch LAYER] [--time INDICES] [--json] [--max-parameters N] [--error-format human|json]
weightc debug SOURCE [--weights ARTIFACT] [--steps N]
  [--watch LAYER] [--time INDICES] [--json] [--max-parameters N] [--error-format human|json]
```

- `SOURCE` 为模型源码。`--weights` 为完整模型目录，相对当前工作目录；省略时使用源码种子和显式权重导入。
  指定时替代初始化、import/inherit/derive 的权重加载，不隐式读取 HEAD。
  解析源码所需的库和派生架构基线仍须可用。
  严格要求展开后的结构及输入输出 shape/dtype 一致，校验模型文件哈希；不做部分迁移。
- `--sample` 接受 JSON 张量或 `{"inputs": 张量}`，形状为声明的单样本形状或带 batch 的形状；
  i64 输入必须是非负整数。省略时使用训练集（或首个数据集）第一行，无数据则用标明用途的全零输入。
  trace 不验证训练任务；提供 sample 时不读取训练 CSV。仍检查结构和冻结选择器。
- `--steps` 默认为 10，范围 1..1000，表示优化器更新次数。debug 要求 train 和有效数据，
  使用源码损失、优化器、学习率、权重衰减及固定首个小批次（最多 64 行）。不恢复优化器历史状态。
- `--watch` 为层路径，包含子层，同时过滤激活和参数报告；省略时显示全部。
  `--time` 为从零开始的最内层循环索引，如 `0,3,7`；省略时不过滤，对共享权重统计无影响。
- `--max-parameters` 默认 10000000，在权重分配前检查参数数量；不是总内存上限。
  debug 额外保留所选参数的初始副本来计算变化量，激活和优化器也需要内存。
- `--json` 默认关闭，开启后返回结构化报告；错误格式默认为 human，可选 json。

报告包含实际权重路径、SHA-256、未有训练来源的张量名单和逐参数统计；debug 另报最后一次
反向传播的梯度状态（冻结、缺失、全零、非零）、最终参数统计、变化元素数和总更新范数。
相对更新为 `||最终权重-初始权重|| / ||初始权重||`，初始范数为零时返回 null。
冻结策略先恢复模型记录，再应用源码 import 冻结默认值和有序 freeze/unfreeze 规则。
所有初始参数都检查 NaN/无穷大，即使被 watch 隐藏或不参与输出。

目前仅 CPU，debug 仅内置 backprop；不自动保存权重、优化器状态或版本。
这些数值用于定位故障，不代表模型质量、训练完成或泛化表现。更多见[调试说明](debugging.md)。


## 只反编译结构

本地命令 `weightc decompile SOURCE -o OUTPUT [--references]` 默认只输出结构。
`SOURCE` 可为原生模型目录、支持的 Python 源码、外部权重文件/目录、
`hf://owner/model[@revision]` 或已有的 `wgt tensors 1/2` 清单；`OUTPUT` 必须不存在。
`--references` 默认关闭；显式开启才保留外部张量位置或原生模型的权重导入。
两种模式均不将具体权重数值写入文本。

- 原生模型和支持的 Python 子集生成 `wgt 0.3`，默认没有 `import` 和训练声明，
  构建时重新初始化，不能当作恢复了训练权重。原生结构提取只读 manifest，不验证权重内容。
- 外部权重默认生成描述格式 `wgt architecture 1`，不是可执行的 `wgt 0.3`。
  JSON 正文保留 `config`、`modules`、`tensors` 和 `executable: false`。
  `modules` 每项包含带 `{i}` 的 `module` 路径、字符串索引列表 `indices`、
  成员字典 `members`；每个成员仅包含 `shape` 和 `dtype`。
  相同维度和类型的编号模块合并描述，索引仍各自独立，不表示参数共享。
  未分组的张量位于 `tensors`；标量形状为 `[]`，空维度为 `0`，类型沿用来源格式。
  不包含权重值、字节偏移、来源文件位置、校验码或权重导入。
- `weightc check` 检查描述格式；不支持 build/run。仅凭张量形状和配置无法恢复
  完整计算图、参数共享关系及运行语义，Qwen 的结构描述也不代表已经能够执行 Qwen。

```sh
weightc decompile examples/decompile/model.py -o /tmp/weightlang/model-architecture.wgt
weightc check /tmp/weightlang/model-architecture.wgt --shapes
# 已有清单可离线精简，不需要重新下载或读取权重值；输出路径需不存在。
weightc decompile weights/qwen3_5_4b/source.wgt -o /tmp/weightlang/qwen-architecture.wgt
```

这是当前源码的本地实现，未声明已发布；格式与限制见 [反编译说明](decompile.md)。


### 从 Python 模型定义生成可执行 WGT

完整命令：
`weightc decompile SOURCE.py -o OUTPUT.wgt [--class-name NAME] [--init-args JSON] [--input-shape D1,D2,...] [--input-dtype f32|i64]`。
`--class-name` 省略时选唯一顶层模型实例，或没有实例时的唯一类；多个类需显式选择。
`--init-args` 是构造参数 JSON 对象，默认使用源码的参数/默认值；覆盖实例参数时必须选择类。
`--input-shape` 不含 batch，省略时从第一个参数层 Linear 推导单个向量宽度；
图片、序列和 token 应显式提供形状。`--input-dtype` 默认 `f32`，token 使用 `i64`。
输出形状由实际 WGT 契约推导。所有选项仅用于 Python 输入；已有输出不覆盖。

当前源码静态解析 PyTorch 的 `nn.Module.__init__/forward`、同文件嵌套类和 Sequential。
支持 Linear、Conv2d、MaxPool2d、末维 LayerNorm、Dropout、Embedding、保留 batch 的
Flatten，以及 ReLU/Tanh/Sigmoid/GELU/SiLU/Identity 和末维 Softmax。
forward 支持一个输入/输出、局部赋值、嵌套调用、残差加法、门控乘法、末维拼接、
相应函数式激活和 `flatten(1)`。加乘不广播；不把不同归一化或激活参数静默换成默认值。
完整构造签名、参数默认值与映射限制见 [Python 转换说明](decompile.md#python-definitions-to-executable-wgt)。

ModuleList 支持显式列表或固定 range 推导式；可遍历容器或按固定 range 索引。
循环展开后各模块保留独立参数，不把同一个共享层复制成独立层；重复调用参数层会报错。
固定 range 最多 1024 次，嵌套模块最多 32 层，生成图最多 10,000 个节点。
任意 Python、动态分支、多输入输出、自定义继承、外部辅助模块、attention/RNN/BatchNorm
及不可表达选项尚不能自动迁移，错误带源码位置且不生成不完整文件。

```sh
weightc decompile examples/decompile/stacked_mlp.py --class-name StackedMLP \
  --init-args '{"width":8,"depth":3}' -o /tmp/weightlang/stacked-architecture.wgt
weightc check /tmp/weightlang/stacked-architecture.wgt --shapes
```

查看 [原始 Python](../examples/decompile/stacked_mlp.py) 与
[可执行 WGT 示例](../examples/decompile/stacked_mlp.wgt)。生成的是本地 `wgt 0.3` 架构，
不执行输入 Python、不加载权重、不恢复训练状态；构建时重新初始化，没有训练声明。
转换测试使用相同参数对比前向与梯度，不能据此声称迁移了训练好的权重或已发布该能力。


## Transformers 迁移所需的基础算子（0.3，本地实现）

以下扩展只用于当前 `wgt 0.3`。旧参数省略时保持旧行为；保存制品和重新加载时
使用相同的参数检查，Rust 契约检查同步识别权重布局。所有结构都是未训练的，未声明已发布。

| 完整签名 | 语义、默认值和约束 |
| --- | --- |
| `linear(in=N, out=M, bias=1)` | `f32[...,N] -> f32[...,M]`；bias=0 无偏置参数，1 保留；0.3 可省略 in，按最后一维推导 |
| `layernorm(dim=D, eps=0.00001, bias=1, affine=1)` | 最后一维必须为 D；eps 有限且 >0；affine=0 没有缩放/偏置参数，affine=1 时 bias=0 仅去偏置；完整形状不变 |
| `gelu(approximate=0)` | 0 为精确 GELU；1 为 tanh 近似；输入输出任意同形 f32 |
| `rmsnorm(dim=D, eps=0.000001, offset=0)` | 最后维归一化：`x / sqrt(mean(x²)+eps)`；offset=0 后乘 weight，offset 非零时在 f32 乘 `weight+offset` 再还原类型；weight 初始化为 `1-offset`，因此 offset=1 初始化为 0；eps >0，offset 有限，形状不变 |
| `scale(factor=F)` | 任意 f32 张量乘有限常数 F，形状不变，无参数；可表达 embedding 的 sqrt(dim) 缩放 |
| `embedding(vocab=V, dim=D, padding_idx=P)` | P 可省略，默认无 padding 索引；指定时须为 0..V-1；该行初始化为 0、反向不更新；`i64[L] -> f32[L,D]`；这不自动生成 attention mask |
| `self_attention(dim=D, heads=H, causal=0, dropout=0)` | `f32[L,D] -> f32[L,D]`；D 能被 H 整除；causal 为 0/1；新增 dropout∈[0,1)，仅训练时作用于注意力概率 |
| `rope_attention(dim=D, heads=H, kv_heads=K, head_dim=A, theta=10000, eps=0.000001, qk_norm=0, qkv_bias=0, out_bias=0, window=0, dropout=0, clip=0)` | causal GQA，`f32[L,D] -> f32[L,D]`；H 能被 K 整除，A 为正偶数，H×A 不必等于 D；Q 宽 H×A，K/V 宽 K×A，输出投影回 D |

`rope_attention` 使用半维旋转的完整 RoPE，位置固定为 `0..L-1`，theta 是有限正数。
`eps` 是 Q/K 头 RMSNorm 的有限正稳定项；`qk_norm=1` 对每个头最后 A 维归一化，0 不归一化。
`qk_norm=2` 在拆头前对完整 Q（H×A）和 K（K×A）投影分别归一化，
对应权重形状为 [H×A]/[K×A]，与模式 1 的 [A]/[A] 不兼容；用于 OLMo2/3。
`qkv_bias/out_bias` 各为 0/1，分别控制三个输入投影和输出投影偏置。
`window=0` 为完整 causal；正整数 W 仅允许 `0 <= query_position-key_position < W`，包含自己。
`dropout` 默认为 0、范围 [0,1)，仅训练时作用于注意力概率。
`clip=0` 不裁剪；有限正数 C 将投影后的 Q/K/V 限制在 [-C,C]，再进行头归一化及 RoPE。
不支持外部 mask、padding mask、任意 position_ids、缓存、缩放/部分维度 RoPE 或 cross-attention。
目前实际对照验证为 CPU f32；不能外推为低精度/GPU 或完整生成 API 已验证。

### block 的数值参数

block 实参和默认值允许有限整数或浮点数（包括 0 和负数），例如 eps、dropout、bias。
尺寸和循环次数仍由使用它们的算子要求正整数，不因为 block 参数放宽而允许小数维度。
默认值按参数顺序求值，可引用之前的参数和常量；没有运行时动态参数。

```wgt
wgt 0.3
model normalized_projection
input x: f32[time=4, feature=8]
output y: f32[time=4, feature=8]
# eps 是浮点稳定项，bias=0 表示没有偏置权重；各时间位置共享投影参数。
block project(dim, eps=0.000001, bias=0):
  layer norm: rmsnorm(dim=dim, eps=eps)
  layer fc: linear(in=dim, out=dim, bias=bias)
layer body: project(dim=8)
# 只有初始化，未训练；归一化与投影保持 [time, feature] 形状。
init seed=0
```

### std/hf_decoder

所有 block 输入输出 `f32[time,dim]`，内部维度固定，残差要求完整同形。
相同 block 的多次实例化各自独立参数；库不携带权重，作用是提供可执行架构组合。
D/H/K/A/F 分别对应 dim/heads/kv_heads/head_dim/hidden（下列签名使用完整名字）。

| 完整 block 签名 | 目的 |
| --- | --- |
| `gpt2_block(dim, heads, hidden, eps=0.00001, attn_dropout=0.1, resid_dropout=0.1)` | pre-LayerNorm、causal 自注意力、残差、tanh-GELU FFN、残差；attn_dropout 作用于概率，resid_dropout 作用于两个残差分支 |
| `rope_block(dim, heads, kv_heads, head_dim, hidden, eps=0.000001, theta=10000, qk_norm=0, qkv_bias=0, out_bias=0, mlp_bias=0, window=0, attn_dropout=0)` | RMSNorm→RoPE/GQA→残差→RMSNorm→SiLU 门控 FFN→残差；MLP 的 gate/up/down 均按 mlp_bias 控制偏置 |
| `gemma_block(dim, heads, kv_heads, head_dim, hidden, eps=0.000001, theta=10000, attention_bias=0, attn_dropout=0)` | Gemma 1 的 offset=1 RMSNorm、RoPE/GQA 和 tanh-GELU 门控 FFN；embedding 缩放位于 block 外；attention_bias 控制全部注意力投影 |
| `olmo_block(dim, heads, kv_heads, head_dim, hidden, theta=10000, attention_bias=0, attn_dropout=0, clip=0)` | OLMo 1 的无 affine LayerNorm（eps=1e-5）、RoPE/GQA、SiLU 门控 FFN；clip 在投影后裁剪 Q/K/V |
| `olmo2_block(dim, heads, kv_heads, head_dim, hidden, eps=0.000001, theta=10000, attention_bias=0, attn_dropout=0, window=0)` | 完整投影 Q/K RMSNorm、注意力输出 RMSNorm 后残差、SiLU 门控 FFN 输出 RMSNorm 后残差；所有注意力投影 bias 由 attention_bias 控制；输入输出 f32[L,dim]，默认 window=0 为全序列 causal；window>0 为滑动窗口，只允许关注当前位置及之前 window-1 个位置，无缓存 |
| `starcoder2_block(dim, heads, kv_heads, head_dim, hidden, eps=0.00001, theta=10000, bias=1, window=0, attn_dropout=0, resid_dropout=0)` | LayerNorm、RoPE/GQA、非门控 tanh-GELU FFN；bias 控制线性层，LayerNorm 的 affine/bias 保留 |

dim/heads/kv_heads/head_dim/hidden 必须满足对应算子的正整数、整除及偶数约束；
eps/theta 为有限正数，概率∈[0,1)，所有 bias 开关为 0/1，qk_norm 为 0/1/2，window 为非负整数。
有两个投影分支的门控使用独立参数，乘法不广播。示例和命令见
[Transformers 迁移验收](transformers-migration.md) 与 [小尺寸模型](../examples/transformers/README.md)。
`weightc transformers --family FAMILY --config FILE.json --sequence N -o NEW.wgt` 生成上面说明的
基础隐藏状态接口；`weightc transformers --inventory -o NEW.json` 逐文件扫描安装包并记录转换失败，
两者均拒绝覆盖已有输出。完整 442 个模型文件尚未迁移，不能把这些 block 当作全库完成。

## Python 主入口与执行边界

当前源码新增 `weightlang.load(source: str | Path, *, device="cpu", placements=None) -> torch.nn.Module`，
尚未发布。source 为可执行 WGT 文件或完整模型目录；返回 eval 模式模块。
文件加载在内存中初始化并执行显式权重导入/派生、冻结，不写制品、不读取训练数据、
不执行训练回调；依赖解析仍可能使用包缓存。完整目录加载校验哈希并恢复实际权重。
device 默认 CPU，placements 默认 None，其层/参数映射规则见 [设备](devices.md)。
普通架构初始化不代表训练；训练前调用 train()，优化器在加载后创建。

```python
import torch
from weightlang import load
model = load("examples/python_first/model.wgt")
model.train()
x = torch.ones(2, 4)
model(x).sum().backward()
```

Python 管理数据、loss、训练循环和推理编排；WGT 管理模型内部张量计算。
模块当前只接收单个带 batch 维的张量，不自动补 batch 或转换列表，不关闭梯度。
多输入、动态 mask、KV cache 不因新增标准库而自动获得支持，必须另行实现接口与运行时。
CLI 的 train/回调保留为显式编译任务功能。架构、完整模型制品、训练断点必须区分；
`load` 不恢复优化器状态。保存与完整例子见 [Python 接口](python.md)。

### Python 前向 hook

当前源码提供 `forward_hooks(model: nn.Module, hooks: Mapping[str, Callable])` 上下文管理器，
尚未发布。两个参数必填，无默认监听层；键为 `model.named_modules()` 中的精确路径，
空字符串表示整个模型，不支持通配符。进入时校验并安装，退出或异常时仅移除自己安装的 hook。
回调签名为 `(module, inputs, output)`；inputs 是位置参数 tuple，合并算子的首项可能是张量 list。
返回 None 只观察，返回其他值会替换输出；原地修改同样会改变行为，不提供只读隔离。
不关闭 autograd；共享循环内的层每次调用都触发，循环外层仅观察整个循环结果。
监听的是模块边界，并非每一个内部张量操作。原生 PyTorch hook 仍可直接使用。

```python
import torch
from weightlang import load, forward_hooks
model = load("examples/python_first/model.wgt")
def show(module, inputs, output):
    print(output.detach().flatten()[:6].cpu().tolist())
with forward_hooks(model, {"head": show}):
    model(torch.ones(2, 4)).sum().backward()
```

hook 不写入 WGT 或权重；保存完整制品须先退出上下文。若回调替换输出是模型必要行为，
移除它会改变模型，不能把移除后的制品称为等价模型。hook 属于实例，并发调用同一模型也会触发。
保留调试快照用 detach().clone()，避免保留训练计算图。完整说明见 [Python 接口](python.md)。


## 可选动态 Torch 模型前端（0.3，本地未发布）

`backend torch-dynamic` 提供 `module/setup/fn` 的 Python 兼容可执行模型源码，
支持动态张量、多输入／输出和显式跨调用状态；它与静态 layer/block 图分开，
`check --shapes` 只报告运行时契约，不提供静态形状证明。构造参数沿用对应 Torch 类，
GRU 为 PyTorch reset-after。制品使用独立格式，加载须显式信任源码；Rust、静态导出、
revision/自动复用不支持该格式。完整语法、参数、默认值、保存与验证边界见
[动态张量程序](tensor-programs.md)。

OLMo3 结构适配器复用 `hf_decoder.olmo2_block`。缺省 `layer_types` 时，每四层的
第四层使用完整注意力，其余使用滑动窗口（缺省 4096）；显式列表逐层保留。
OLMo3 的缺省 rms_norm_eps 为 1e-5，转换器显式传给 block；
输入输出仍为 i64[time] → f32[time,feature]，仅验证无 padding、无缓存的基础隐藏状态。
可执行小配置见 [OLMo3 示例](../examples/transformers/olmo3.wgt)，只是初始化结构，未训练、未发布。

### 内置标准库直接导入

完整签名：`import std/<库名> [as <别名>]`，仅支持 wgt 0.3；省略别名时取库名。
与 `use std/<库名>` 共用命名空间及展开逻辑，重复别名报错；未知库报错。
不改变 `import <目标层> from "<制品路径>" [layer=...] [sha256=...] [freeze]` 的权重导入语义，
也不增加远程源码库支持。标准库 block 的参数、默认值及形状限制仍以本参考对应条目为准。

```wgt
wgt 0.3
import std/nn as n
model builtin
# feature 为单样本特征轴，运行时另加 batch；4 -> 8 -> 4。
input x: f32[feature=4]
output y: f32[feature=4]
# 两次投影参数独立，随模型初始化；标准库不携带已训练权重。
layer body: n.mlp(in=4, hidden=8, out=4)
init seed=42
```

## 显式序列运行与 MiniMind 迁移扩展（本地 0.3，未发布）

`linear(in=..., out=..., bias=0, tie_embedding=N)`：默认 N=0 不共享；N>0
引用当前 block 作用域内此前第 N 个 embedding（从 1 起）。输入宽度必须等于 dim，
输出宽度等于 vocab，禁止 bias。共享同一个 Parameter，优化器只更新一次；制品为
两个名称记录同值张量，重载拒绝冲突值并恢复共享。绑定参数禁止分到不同设备。

`rmsnorm(dim, eps=1e-6, offset=0, cast_after=0)` 新增 cast_after 开关；0 保留
先将归一化结果转回输入 dtype 再乘权重，1 在 float32 归一化结果乘权重后转换 dtype。
MiniMind 使用 1。`rope_attention` 新增以下可选参数，已有 GQA/窗口等参数保持有效：

- `max_len=32768`：允许的最大 past+current 长度；`flash=1`：允许无 padding 全序列 SDPA。
- `yarn_factor=1`（1 不缩放）、`yarn_original=2048`、`yarn_fast=32`、`yarn_slow=1`、
  `attention_factor=1`：MiniMind 风格 YaRN。当 max_len>yarn_original 才应用频率缩放。
  所有尺度/beta 为有限正数，长度为正整数；theta 在启用 YaRN 时必须大于 1。
- `norm_cast_after=0`：Q/K RMSNorm 的 cast_after；`rope_fp32=0`：1 使用 float32
  cos/sin 运算后转回输入 dtype，0 保留输入 dtype 的 cos/sin。

`sparse_moe(dim, hidden, experts, top_k, normalize=1, aux_coef=0.0005, activation=0)`
输入输出均为 f32[time,dim]；top_k<=experts，维度/专家数/k 为正整数；aux_coef 非负。
activation=0/1/2 对应 SiLU/ReLU/精确 GELU。每个专家有 gate_proj/up_proj/down_proj
无偏置参数；路由为无偏置 gate 投影的 softmax。normalize=1 且 k>1 时选中权重归一化；
k=1 使用 `top1(detach(input))-detach(top1)+1`，保持 MiniMind 直通梯度。
未被选中的专家训练时获得零梯度；训练辅助损失为原 MiniMind top-k 负载均衡公式。

`weightlang.run_sequence(module, input_ids, *, attention_mask=None, past_key_values=None,
use_cache=False, capture=())` 是显式 Python 序列接口，不是新增 WGT 多输入端口语法。
input_ids 是非空 i64[B,T]；mask 是二值 [B,past+T]；cache 按深度优先注意力顺序传入，
每层为同形 [B,past,KV,D] K/V，层间长度一致。返回 SequenceOutput(output,
past_key_values,aux_loss,intermediates)。capture 是需保留的层路径，避免默认保留所有中间张量。
缓存不会存到 module 上；仅支持可沿时间长度变化的 seq/分支解码图，不支持 recur/scan、
自定义 position_ids、复杂 mask 或分层设备放置。静态端口仍是固定形状；普通 Artifact.predict
仍按固定端口验证。独立 Python 导出支持无 cache 的单张量 forward；扩展序列调用使用本运行时。
这些结构及算子本身不含训练权重。

保存已冻结的 Python 快照可用 `compile_program(..., no_train=True, initial_state=state_dict)`。
它初始化新制品，不修改已有冻结基线；严格要求完整名称、形状、有限 float32 和共享值一致。
禁止与训练、import/inherit/派生/reuse 同时使用，来源明确标记 edited/caller-supplied，
不会把外部参数自动认证成 trained。原有 Artifact.save 的冻结检查保持有效。


## 具名多端口与状态返回（0.3，本地未发布）

- `input name: dtype[D1,...]` 可重复声明，名称全局唯一；维度仍为固定正整数，类型为 f32/i64/bool。
- `output name: dtype[D1,...] from tensor` 返回顶层输入或层结果。多输出必须各写 from；单输出省略时默认最终层。
- 多输入、多输出或显式 output from 的模型返回 `dict[str, Tensor]`；旧单输入输出接口仍返回 Tensor。
- `model(*tensors, **named_tensors)` 按声明顺序绑定位置参数；拒绝重复、缺失、额外名称、错误类型/形状及不一致 batch/device。
- `scan(hidden=H, cell(...)) from sequence, initial` 的 initial 是同 batch 的 f32[H]，省略则为零。输出仍为完整状态序列，用 `last` 得到最终状态。
- 状态不进入 state_dict；Python 保留或 detach 状态，权重跨时间共享。`recur` 仍保持原有形状与共享规则。
- 制品 v4 保留有序 inputs/outputs 和输出 from，加载时重新推导每个端口；Python 导出与保存更新保留该契约。
- 有界符号维度、标量/空端口、block 多端口及 KV/MoE 结果端口见后续章节；可选输入和任意嵌套返回仍不支持。
- 此接口不接受源码 data/train、inherit/derivation；训练在 Python 中完成。不代表模型已训练或功能已发布。

完整可执行例子、默认值与限制见 [具名端口专题](dataflow.md#named-tensor-boundaries-and-explicit-scan-state-03-localunreleased)
及 [streaming_ports.wgt](../examples/streaming_ports.wgt)。

## 有界符号维度与整数约束（0.3，本地未发布）

| 声明 | 参数、默认值与约束 |
| --- | --- |
| `dim T min=1 max=16` | min/max 都必填，正整数字面量，含端点；必须位于全部端口前；不能与 const 重名 |
| `require P + T <= MAX_LEN` | 比较符为 `== != < <= > >=`；引用已声明符号/整数常量；至少已声明一个 dim |
| `input x: f32[T,4]` | 每个符号至少直接出现在一个输入轴上；据真实输入绑定，同名符号必须一致 |
| `input mask: bool[P+T]` | 其他轴可以使用 `+ - * // %` 及括号；除数正整数，无浮点 `/` |

整数绝对值不超过 2147483647，表达式深度不超过 16；端口所有合法形状仍须为正整数。
至多 8 个符号、64 个 require、未经过约束过滤的 4096 种取值组合，及 200000 个
“取值组合 × 展开层节点”。编译器逐一检查所有合法组合，不只检查一个示例长度；
无合法解、不同长度导致权重形状变化或违反输出契约均报错。层和 block 参数仍是编译期常量。

符号模型始终返回具名字典。运行时先绑定符号、验证范围/约束/各端口，再执行任何层。
模型制品 v5 保存 dimensions/constraints/shape_expr；普通 shape 和层记录是首个合法赋值的
代表形状。Python/Rust 加载重新验证完整有限范围，保存更新、独立导出和反编译保留原契约。

暂不支持符号算子参数、有界模型中的 rearrange/reduce/flatten（含 transpose/mean/sum 简写）。
零长度 cache、标量、KV/MoE 图端口和多状态 block 已由后续增量实现。完整签名、限制、运行例子和表示格式见
[维度专题](dimensions.md#bounded-symbolic-dimensions-03-localunreleased)；
[variable_streaming.wgt](../examples/variable_streaming.wgt) 仅初始化，未训练。


## 张量原子、标量与空端口（0.3 本地增量）

当前支持标量 `f32[]/i64[]/bool[]`、非负静态维度及 `dim P min=0 max=N`；
新增 matmul、broadcast、cat/stack/slice、transpose_axes、reduce_sum/mean/max、
gather/scatter_add、eq/lt 和 masked_softmax。归约可以输出标量，空 sum 为零；
全 mask 的 softmax 行输出和梯度为零。0.3 允许无参数计算模型。
Conv1d 支持 groups/dilation/causal 左填充。
原生数学原语：`square/rsqrt/exp/log/sin/cos/softplus` 无参数，`shift(value=V)`
要求有限数值 V，`cumsum(axis=A)` 要求样本轴整数；均接受 f32、保持形状，
不隐式转换精度。逐元素算子支持标量与空轴，cumsum 支持空轴但拒绝标量。
softplus 固定 beta=1、threshold=20；log/rsqrt 的非正输入遵循 Torch inf/NaN，
不自动 clamp。`std/tensor_math.l2norm(eps=1e-6)`、`rms(eps=1e-6)` 沿末轴
无参数归一化；eps 是有限数值，正值才能保证零输入的有限归一化因子。
完整示例见 [tensor_math.wgt](../examples/tensor_math.wgt)。这不代表完整精度策略已实现。
完整签名、参数默认值、形状/dtype、空集合边界、限制和例子见 [primitives](primitives.md)。
这些规则替代前文历史增量中“所有端口正整数、禁止标量”的限制；
KV/MoE 的显式结果端口、具名 block 返回和多状态 scan 见下节；完整精度策略仍未完成。

## Lean 线性公式桥接（本地实验，未发布）

`weightlang.lean.LinearFormula(in_features, out_features, bias=True)` 从同一个
受限线性公式生成既有 WGT linear 与 Lean 定义；不新增 `.wgt` 语法。
正整数宽度、f32 运行时、`[out,in]` 权重和行共享偏置；`module(device="cpu")`
在内存中初始化，`verify(lean="lean", timeout=60)` 显式检查 Lean 4.19.0 定理。
证明覆盖行拼接性质，不覆盖浮点内核、梯度或训练效果。
完整签名、默认值、公式、信任边界与可运行示例见 [Lean 桥接](lean.md)。

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

Lean 桥接现另有 `FormulaGraph(in_features, nodes)` 与
`FormulaNode(name, op, inputs, out_features=None, bias=True)`：不可变单向量输入图，
支持 linear、同宽 add/mul、分支复用和残差，最后节点为输出。构图时推导宽度并拒绝
前向/循环引用与隐式广播；整图生成 Lean `graph_rows` 定理。没有新增 WGT 语法。
节点签名、权重身份、证明范围与运行示例见 [可组合公式图](lean.md#可组合的带类型公式图)。

## 原生 DoRA、PiSSA 与 QLoRA（0.3，本地实现）

`adapter NAME: METHOD target=PATH rank=R alpha=A [freeze=1]` 支持
`lora/dora/pissa/qlora`。R 为 1..min(in,out)，A 正有限；目标为普通、不共享
weight/bias 的 linear。后三种要求 freeze=1，只冻结目标层基础参数。输入输出沿用
linear 的 f32 形状契约，低秩 A=[R,in]、B=[out,R]；Python 管理损失与训练循环。
DoRA 学习幅度与低秩方向；PiSSA 精确 SVD 初始化并冻结残差；QLoRA 为 64 元素
块 NF4、uint8 打包，f32 计算。QLoRA 完整制品仍 F32，无 double quantization、
paged optimizer 或 CUDA 专用内核；不能宣称完整论文训练栈或大模型性能。
普通基础模型须直接导入目标 linear 后初始化适配器；新方法不支持自定义 init。
完整默认值、公式、限制、保存/加载/合并/断点接口及有效例子见 [适配器](adapters.md)。
可执行模型：[DoRA](../examples/adapters/dora.wgt)、[PiSSA](../examples/adapters/pissa.wgt)、
[QLoRA](../examples/adapters/qlora.wgt)，均仅初始化；[训练脚本](../examples/adapters/train.py)
显式运行合成训练，不代表预训练或实际任务效果。

## 自定义权重类型（0.3，本地实现）

`wtype NAME = [VALUE, ...]` 定义非空有限 f32 码本（无默认值，名称和
f32 取值不可重复）；`weight PARAMETER_PATH: TYPE` 约束整块参数。
`weight PARAMETER_PATH: TYPE axis=A start=S stop=E` 只约束指定轴的半开切片，
三个整数选项必须同时提供，轴非负且切片不得越界。同一参数可声明沿同一轴的
多个不重叠切片；未指定部分保持浮点。共享参数的别名继承同一约束。

前向最近邻投影，等距取较小值；反向恒等 STE，优化器和制品保存 f32 主参数。
这是任意有限实数集合及分段混合，不是任意 Python 变换或低位存储。
具名输出、Python 训练边界、v7 制品、冻结、共享、适配器限制及完整示例见
[自定义和混合权重类型](weight-types.md)与
[可运行模型](../examples/custom_weight_types.wgt)。没有标准库签名或端口 dtype 改动。
