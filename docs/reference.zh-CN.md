# WeightLang 语言与标准库参考

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
| `i64[L]` | 整数 token 序列，通常交给 embedding，0.3 |
| `f32[...]` | 0.3 的固定多维浮点形状；各算子有各自维数限制 |
| `name=value` | 算子和 block 参数按名字传递；不是位置参数 |
| `const WIDTH = 32` | 0.3 编译期数值常量，不是运行时张量 |
| `+ - * / // %` | 0.3 数值表达式；支持一元正负号和括号；乘除余数优先于加减 |

形状不含 batch（一次处理的样本数），每一维必须是正整数。
`/` 得到浮点数，`//` 是向下取整除法；`8 / 2` 不能作为整数维度。
表达式可用于形状、算子参数、block 参数和默认值，不能替代训练参数或数据行里的数值字面量。
常量按声明顺序求值，端口维度只能引用已声明常量；层参数在文件解析完成后解析。
block 参数覆盖同名常量，默认值可引用前面的参数。没有运行时变量、动态形状或任意代码求值。

## 关键字与语句

这里按语法用途列词，不表示这些名字在所有上下文中一律保留。

| 关键字 / 写法 | 解释 | 版本 |
| --- | --- | --- |
| `wgt 0.3` | 文件首条语句，声明语言版本；接受 0.1、0.2、0.3 | 0.1+ |
| `model name` | 声明该文件唯一的模型 | 0.1+ |
| `input x: f32[N]` | 声明输入名称、类型、单样本形状 | 0.1+ |
| `output y: f32[M]` | 声明输出契约；最终层的结果必须匹配，不是 return 语句 | 0.1+ |
| `layer name: expression` | 创建一个层并命名其输出；默认接前一层 | 0.1+ |
| `init seed=42` | 必须显式初始化；seed 是非负整数 | 0.1+ |
| `data name:` | 定义数据集，后接缩进的数据行或文件选项 | 0.1+；选项 0.2+ |
| `train data=... loss=... optimizer=... lr=... epochs=...` | 声明一次训练；每个模型最多一条 | 0.1+ |
| `import body from "../weights/base" layer=body` | 从制品导入兼容的层权重；省略 layer 时使用目标同名路径 | 0.1+；点路径 0.2+ |
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

## 内置算子

表中 N/M/D/C/O/K/S/L/V 均为正整数；以下稳定写法显式提供必填参数。
除 embedding 接收 i64 外，下列算子接收 f32。激活函数保持形状。

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

### 正在开发的形状语法

本次核对期间，工作区解析器正在加入以下接口；这里只记录当前语法，
尚未完成端到端运行验证，不应据此认为已发布或运行时已全面支持。

| 写法 | 当前解析器中的用途 |
| --- | --- |
| `f32[time=8, feature=4]` | 为固定维度附加名称；名称不能重复，仍不是动态维度 |
| `linear(out=16)` | 省略 in，由输入契约推导最后一维；上表保留显式 in 的通用写法 |
| `rearrange("h w -> w h")` | 通过轴模式重排张量；额外轴大小通过具名正整数参数传入 |
| `reduce("h w -> h", op=mean)` | 按模式归约维度；op 必须显式为 mean 或 sum |

这些接口完成后，需要按维护规则补全版本约束、轴模式限制和可运行示例。

## 连接、分支与循环

| 写法 | 解释与约束 | 版本 |
| --- | --- | --- |
| `residual(block(...))` | 输入 + block 输出；完整形状与类型必须相同；不能直接包单个 linear | 0.2+ |
| `linear(in=4, out=8) -> relu` | 从左到右连接；尾部 from 作用于整个管线 | 0.3 |
| `add(a, b, ...)` | 至少两个具名张量逐元素相加 | 0.3 |
| `mul(a, b, ...)` | 至少两个具名张量逐元素相乘 | 0.3 |
| `concat(a, b, ...)` | 至少两个张量沿最后一维拼接，其余维度一致 | 0.3 |
| `mix(g, a, b)` | g*a + (1-g)*b；不会自动把 g 限制到 [0,1] | 0.3 |
| `where(c, a, b)` | c>0 选 a，否则选 b；两支都会执行，条件不获得梯度 | 0.3 |
| `repeat(N, body)` | 展开 N 份独立参数，层名为 step0…stepN-1 | 0.3 |
| `recur(N, body)` | 用同一组参数执行 N 次；body 必须保持完整形状与类型 | 0.3 |
| `scan(hidden=H, cell(...))` | f32[L,D] → f32[L,H]；按序列执行共享参数的 cell | 0.3 |

合并算子接收 f32，不隐式广播；add/mul/mix/where 的所有输入形状必须完全相同。
它们必须占据具名层，不能直接作为管线内的步骤。
`input` 指当前模型或 block 的原始输入；模型输入端口名称也可引用。
引用只能指向同一作用域内更早的层，不能向前引用、自引用或穿透其他 block。
`input` 是保留层名；scan cell 内的 `state` 也是保留名。

scan cell 的 `input` 是当前时间步 f32[D]，`state` 是上一时刻 f32[H]，
cell 输出必须为 f32[H]。初始状态为零，每次模型调用重置；全部时间步参与反向传播。
没有外部初始状态、持久流式状态或可变长度支持。
repeat/recur 次数为 1…1024；最多展开 10,000 个节点、嵌套 64 层 block。
`seq` 是编译器展开后的内部节点，不是供用户调用的函数。

## 标准库函数

先 `use std/库名`，再 `库名.函数(参数名=值)`；也可通过 `as` 改别名。
以下列出全部随源码提供的 block 签名；所有 block 参数均为正整数。
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
CSV 前若干列为输入，其余为目标；不支持带引号的 CSV 字段。

| 参数 | 解释 |
| --- | --- |
| `file="train.csv"` | 相对源文件的数据路径；必须同时提供 sha256 |
| `sha256=...` | 数据文件原始字节的 64 位小写十六进制 SHA256，含被跳过的表头 |
| `skip=0` | 跳过开头若干行，非负整数，仅用于文件数据 |
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
