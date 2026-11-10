# 显式初始化

WGT 没有默认随机初始化。模型源码声明参数的形状、约束和可选初始化配方；宿主明确选择执行配方，或者从完整权重构造实例。没有配方的参数可以通过检查和编译，但实例化时必须得到外部绑定。本文描述当前仓库的 WGT 0.5 实现，尚未发布。

## 定义与实例分开

`ModelDefinition::parse`、`ModelDefinition::from_file`、`wgt check` 与 `wgt compile` 只建立和检查定义，不抽取随机数、不执行初始化 helper，也不连接模型服务。配方会保留在定义或导出的 IR 中。检查包括名称、参数数量、形状、纯函数限制和可静态判断的范围；依赖张量值的错误可以到物化时才出现。例如 `sqrt(full(-1))` 形状合法，但实际初始化会因非有限结果失败。实现与回归见 [`initializer.rs`](../crates/wgt-language/src/initializer.rs)、[`model.rs`](../crates/wgt-language/src/model.rs) 和 [`initializers.rs`](../crates/wgt-language/tests/initializers.rs)。

| 宿主入口 | 参数如何得到值 | 状态如何得到值 |
| --- | --- | --- |
| Rust `Client::initialize`；Python `load`/`loads` 读取源码、`from_ir` | 执行全部声明的参数配方；任何未绑定项报错 | 执行声明的状态配方 |
| Rust `Client::load_weights` / `Client::load_snapshot`；Python `restore` | 要求完整、精确的参数绑定；跳过全部参数配方 | 新实例仍执行状态配方 |
| Rust `Model::load_weights` | 原子绑定现有实例；frozen 必须保持原值 | 保留现有状态；成功重置优化器状态 |
| Python `load` 读取 SDK `.pt` / `.pth` | 严格恢复保存的 `state_dict`，不先运行参数配方 | 恢复保存的状态，不先运行状态配方 |

权重恢复不在缺项时回退到源码初始化。参数权重、操作状态和优化器历史是不同数据；参数快照不包含后两者。`state` 也可以暂不绑定以供检查，但创建实例或从参数快照恢复时，未初始化的状态仍会明确报错。完整权重生命周期见[权重快照与迁移](weights.md)。

## 每个参数的配方

普通 `parameter` 与 `frozen` 可写有限字面量列表、初始化表达式，或者省略整个初值子句。显式 `= external` 与省略初值一样，表示必须由外部提供完整绑定；它们不会变成零。`learn` 声明也可省略初值以等待外部存储绑定，原有显式初值和域约束继续有效。

下面的模型将二维特征映射到二维特征。矩阵初值是单位阵加有界噪声，偏置明确为零；随机种子是 helper 的显式参数。

```wgt
wgt 0.5
model ExplicitProjection
input x: f32[2]
output y: f32[2]
parameter weight: f32[2,2] = near_identity(7, 0.05)
parameter bias: f32[2] = zeros()

# The declared target supplies [output_features, input_features].
fn near_identity(seed, amplitude) {
    let noise = uniform(seed, -1, 1);
    return eye() + noise * amplitude;
}

fn forward(x) {
    return linear(x, weight) + bias;
}
```

初始化表达式使用已有函数 parser，不是新的宿主脚本语言。自定义 helper 只允许不可变、不可重复的 `let` 和最后一个 `return`；参数数量须匹配，禁止递归、参数/状态捕获、I/O、训练、异步及任意宿主调用。无参初始化 helper 可以被配方引用，不因此成为 typed 操作入口。helper 展开深度最多 32，配方表达式深度最多 64，并受节点与张量大小预算约束。

### 初始化生成函数

以下函数仅属于初始化上下文。它们不增加 forward/loss 可用的张量原语，也不会预导入任何 `std` 算法。每个生成函数都支持在首位显式传入形状列表；省略这个形状参数时使用当前目标参数的声明形状。随机种子和值域参数均无默认值。

| 完整调用形式 | 结果与约束 |
| --- | --- |
| `zeros()` / `zeros(shape)` | 全零张量 |
| `ones()` / `ones(shape)` | 全一张量 |
| `full(value)` / `full(shape, value)` | 以有限标量填充 |
| `eye()` / `eye(shape)` | rank-2 单位矩阵；非方阵按主对角线填一 |
| `uniform(seed, low, high)` / `uniform(shape, seed, low, high)` | 显式种子的均匀样本，要求有限 `low < high` |
| `xavier_uniform(seed)` / `xavier_uniform(shape, seed)` | rank-2 矩阵，范围为 `±sqrt(6/(rows+cols))` |

形状为 1..8 个正整数轴，每轴 1..16384，参数及一般模型总参数元素数上限为 16,777,216，顺序 layer 模型源码另限 4,000,000；实际宿主与快照仍有独立字节预算。数值 seed 必须是可精确表示的非负整数，最大为 `2^53-1`；更大的 u64 seed 用十进制字符串，例如 `uniform("18446744073709551615", -1, 1)`。字符串只是种子的精确表示，不是路径。字符串 seed 当前须直接写在 `uniform` / `xavier_uniform` 调用中，不能经 helper 形参传递；可精确表示的数值 seed 可正常传递。

配方支持有限标量、布尔值、平坦列表、`+ - * /`、矩阵乘 `@` / `matmul`、平方 `** 2` / `square`，以及 `sqrt`、`exp`、`log`、`tanh`、`sigmoid`、`relu`。标量与张量可在初始化算术中组合；这个规则不改变 forward 的广播合同。几何操作包括 `reshape(value, shape)`、rank-2 `transpose(value)`、一维 `concat(left, right)` 和非空半开切片 `value[start:end]`。最终配方形状须精确匹配目标；显式列表可用 `reshape` 组成矩阵。

### 标准库配方

`std/initialization` 提供三个由上述生成函数组成的纯 helper：`identity(gain)`、`symmetric(seed, amplitude)`、`constant(value)`。使用前必须显式导入。完整签名与范围见[标准库](standard-library.md#初始化模块)。

```wgt
wgt 0.5
import init from "std/initialization"
model IdentityProjection
input x: f32[2]
output y: f32[2]
# One square matrix; no hidden bias or random choice.
parameter weight: f32[2,2] = init.identity(1)
fn forward(x) {
    return linear(x, weight);
}
```

## 顺序模型的初始化指令

顺序 `layer` 模型可用一条 `init weights=EXPRESSION bias=EXPRESSION` 为模型的权重和偏置提供配方；当前语法要求这两个子句都出现，不带偏置的模型也遵循这一写法。某一类参数需外部绑定时明确写 `external`。`init` 不可重复；省略整条指令表示所有隐式生成的参数都等待外部绑定。它不会补入 Xavier 或零偏置。当前顺序 init 支持上述内置生成函数与纯表达式组合；自定义或导入 helper 用于显式 `parameter` / `frozen` 声明，顺序 init 暂不解析这些 helper。

```wgt
wgt 0.5
model SequentialProjection
input x: f32[2]
output y: f32[2]
layer projection: linear(out=2)
# Explicit algorithms and seed; creates projection.weight and projection.bias.
init weights=xavier_uniform(7) bias=zeros()
```

顺序模型的一条初始化指令共享模型级随机流：相同 seed 的 `uniform` 配方按参数声明顺序续接 SplitMix64 样本流。上面的 Xavier 与零偏置组合保留历史模型级 seed 初始化的数值。独立的 `parameter` 配方各自求值，因此相同 seed、形状和表达式得到相同值；若希望不同参数独立取值，应明确使用不同 seed。参数改名、增删和顺序变动不会获得通用的身份稳定随机承诺。

现有兼容读取器仍接受旧模型级 seed 声明，以便读取历史源码快照；新代码和文档使用显式算法形式。`wgt format` 只规整现有语法，不负责升级历史语法。

## 自定义宿主算法与增量初始化

无法用纯初始化表达式描述的算法放在 Python/Rust 宿主。宿主通过 `Plan` 显式给每个目标参数提供完整张量、固定快照引用或初始化回调。`preview` 展示来源与缺项，不执行回调；`build` 先确认所有目标都有来源，然后才执行选择的配方或回调。新架构只需对新增参数选配方，旧参数显式复制或变换，不会重复初始化旧参数。

Rust 用 `Plan::initialize_declared(name)` 选择单个声明配方，或用 `Plan::initialize(name, callback)` 注册算法。Python 用 `Plan.initialize(names)` 选择声明配方，或用 `Plan.set(name, callback)` 注册算法；省略 `names` 是显式请求选择全部尚未绑定参数。回调收到目标形状，须返回同形有限 F32 张量。API 和两版本加权平均、扩宽迁移示例见[权重快照与迁移](weights.md)。

## 参数值域与可重复性

可训练 `prob` 参数的源码初值是概率，必须严格位于 `(0,1)`，物化为 logit 存储；`frozen prob` 直接存储 `[0,1]` 中的概率并允许端点。`learn bool` / `learn int` 的权重存储保留连续 shadow，读取和候选更新各有自己的离散/投影合同；恢复不得先取整。普通 `f32` 不会因初始化而自动获得值域约束。参见[可学习架构参数](learnable-architecture-parameters.md)。

显式 seed 使同一实现中的相同配方可重复执行，但不等于任意设备、数学库或未来版本都逐位相同。需要复用精确数值时，保存并固定快照的完整 revision；需要审查生成过程时，同时保存配方、源码和宿主算法版本。快照 revision 校验实际 payload，不证明算法、架构语义或 provenance 具有可信签名。
