# 模型内部标准库

WGT 声明模型结构、内部计算与显式局部更新，Rust/Python 程序负责实例化、数据、训练调度和应用控制。这里的标准库包含模型内部的纯张量函数与初始化配方，不提供网络、数据读取、异步任务或工具调度模块。当前为本地 WGT 0.5 实现，尚未发布。

## 语言核心与标准库的边界

语言核心提供组合语法，以及形状检查、原生执行和求导所需的基础张量原语。能够由这些原语组合表达的损失、注意力、网络块和数值算法放在 `std/` 或用户模块中，使用时显式 `import`；不因常用就增加预导入算法函数。标准库函数也不能隐式创建参数实例、读取数据或启动训练。

## 初始化模块

[`std/initialization.wgt`](../std/initialization.wgt) 提供参数初始化配方，使用 `import init from "std/initialization"` 显式导入。检查和编译只展开纯表达式，不执行生成器；宿主初始化实例或显式选择 `Plan` 配方时才物化值。完整权重恢复跳过参数配方，详见[显式初始化](initialization.md)与[权重快照](weights.md)。

| 完整签名 | 目标形状与结果 |
| --- | --- |
| `identity(gain)` | rank-2 矩阵；`eye() * gain`，gain 是显式有限标量 |
| `symmetric(seed, amplitude)` | 当前目标的固定形状；`uniform(seed,-1,1) * amplitude`，seed 显式，amplitude 有限 |
| `constant(value)` | 当前目标的固定形状；`full(value)`，value 有限 |

这些 helper 使用初始化上下文的 `eye`、`uniform` 和 `full`，不能因此在 forward/loss 调用。形状由目标声明提供；需要中间形状时使用生成器的显式 shape 首参。所有算法参数必填。标准库不补入默认 seed，不读取文件或捕获已有模型参数；跨快照复制和迁移由宿主 `Plan` 负责。

## 纯张量模块

```wgt
wgt 0.5
import activation from "std/activation"
import losses from "std/loss"
model FeatureGate
input x: f32[2]
output y: f32[2]
parameter gain: f32[2] = [1, 1]
# One feature axis. The model owns gain; calls share it explicitly.
fn forward(x) {
    let projected = x * gain;
    return activation.swiglu(x, projected);
}
# Reduce features of a sample. Rust controls batch training separately.
fn loss(prediction, target) {
    return losses.squared_error(prediction, target);
}
```

模块源为 [`std/activation.wgt`](../std/activation.wgt) 与 [`std/loss.wgt`](../std/loss.wgt)。以稳定逻辑名导入，无须访问仓库路径；不接受内部 `/operators` 名称或 `.wgt` 后缀别名。

所有参数必填，无默认值。输入是非空同形一维 `f32[N]`，N 表示特征或类别，不含隐式 batch 轴，没有标量广播。函数不分配参数、不初始化权重、不持有状态、不执行训练；重复调用的参数共享由模型显式决定。

| 模块 | 完整签名 | 输出与计算 |
| --- | --- | --- |
| `std/activation` | `swish(x)` | 同形 `f32[N]`，`x*sigmoid(x)` |
| 同上 | `swiglu(gate,value)` | 同形 `f32[N]`，`swish(gate)*value`，两路投影由模型提供 |
| 同上 | `squared_relu(x)` | 同形 `f32[N]`，`square(relu(x))` |
| `std/loss` | `squared_error(prediction,target)` | `f32[1]`，特征平方误差均值 |
| 同上 | `half_squared_error(prediction,target)` | `f32[1]`，上述均值乘 0.5 |
| 同上 | `categorical_logits(logits,target)` | `f32[1]`，`-sum(target*log_softmax(logits))` |

类别目标应为非负且总和为 1 的独热或软标签；纯函数不动态验证分布，其他有限目标按同一加权公式计算。输入必须是 logits。稳定 sigmoid 与 log_softmax 沿用原语数值边界；极大有限数的平方、乘法或减法仍可能溢出报错。特征归约不替代样本平均。CPU 梯度来自展开后的原语，不承诺 GPU 或融合性能。

## 注意力与数值迭代模块

新增源码 [`std/attention.wgt`](../std/attention.wgt) 与 [`std/numerics.wgt`](../std/numerics.wgt) 复用现有张量原语。`model_std_algorithms` 五项本地测试已通过，覆盖独立 f64 前向参考、输入与参数有限差分、原生 C 执行、共享增益梯度及显式导入要求；这不代表完整 Transformer 或数值求解器已经验收。调用必须分别显式导入 `import attention from "std/attention"` 或 `import numerics from "std/numerics"`。

| 模块 | 完整签名 | 形状与计算 |
| --- | --- | --- |
| `std/attention` | `dot_product_attention(q,k,v)` | `q:[Tq,D]`、`k:[Tk,D]`、`v:[Tk,Dv]`，返回 `[Tq,Dv]`；沿键轴归一化 `q × transpose(k)`，再乘 `v` |
| `std/numerics` | `relax(estimate,observation,gain)` | 三个输入为同形非空 f32 张量，返回 `estimate + gain * (observation-estimate)`，形状不变 |

所有参数必填。注意力函数不自动除以 `sqrt(D)`；调用方按需预缩放 `q`，例如 `D=4` 时使用 `scale(q,0.5)`。它不提供 mask、batch、多头拆分或缓存。数值更新函数不广播步长，也不限制 `gain` 到 `[0,1]`；标量情况用显式 `[1]` 张量。两者都不创建参数或状态，调用方决定参数共享关系，梯度由展开后的基础原语计算。

## 脉冲与局部学习模块

使用 `import snn from "std/snn"`、`import plasticity from "std/plasticity"`、`import ff from "std/ff"` 显式导入。所有参数必填，无默认值或隐式 batch；函数不分配参数、持有 state 或提交写入。元组返回值必须在模型内解构，外部端口仍是向量或 void。

| 模块 | 完整签名 | 形状、语义与输出 |
| --- | --- | --- |
| `std/snn` | `lif(voltage,current,leak,threshold)` | voltage/current/threshold 同形，leak 为 `[1]`；`u=scale(voltage,leak)+current`，返回 `(u-ge(u,threshold)*threshold, ge(u,threshold))` |
| 同上 | `integrate_fire(voltage,current,threshold)` | 三参同形；无泄漏的相同阈值/减阈值复位，返回 `(next_voltage,spike)` |
| 同上 | `trace(previous,spike,decay)` | previous/spike 同形，decay 为 `[1]`；先衰减旧迹再加当前脉冲 |
| `std/plasticity` | `hebbian(weight,pre,post,rate)` | weight `[M,N]`，pre `[N,1]`，post `[M,1]`，rate `[1]`；返回 `weight+rate*(post@pre.T)` |
| 同上 | `oja(weight,pre,post,rate)` | 单输出单元：weight/pre 同形，post/rate 为 `[1]`；返回 `weight+rate*post*(pre-post*weight)` |
| 同上 | `pair_stdp(weight,pre,post,pre_trace,post_trace,pre_decay,post_decay,plus,minus)` | weight `[M,N]`，pre/pre_trace `[N,1]`，post/post_trace `[M,1]`，其余 `[1]`；返回 `(next_weight,next_pre_trace,next_post_trace)` |
| `std/ff` | `goodness(activity)` | 平方和，输出 `[1]`；在长度归一化前计算 |
| 同上 | `normalize(activity,epsilon)` | epsilon `[1]`；除以 `max(L2长度,epsilon)`，保持 activity shape，零向量保持为零 |
| 同上 | `dense_relu(weight,bias,input)` | weight `[M,N]`，bias `[M,1]`，input `[N,1]`；返回 `[M,1]` ReLU 活动 |
| 同上 | `update_relu(weight,bias,positive,negative,rate,threshold)` | weight `[M,N]`，bias `[M,1]`，positive/negative `[N,1]`，rate/threshold `[1]`；返回同形 `(next_weight,next_bias)` |

SNN 的 threshold 应为正值，时间步由宿主约定，leak/decay 的物理含义由调用方选择。每一步每个元素最多一个脉冲；没有隐含不应期、延迟或事件队列。`ge` 不可自动微分，因此 SNN 库用于无 AD 的操作路径。脉冲输入应是 0/1；库按公式计算，未对所有生物模型前置条件做动态验证。

STDP 使用先衰减的旧迹计算 `plus*(post@old_pre.T)-minus*(old_post@pre.T)`，再向迹加入当前脉冲；同一步且零历史的前后脉冲不会产生配对更新。权重不自动裁剪。它是明确的离散 pair-STDP 计算规则，不声称完整生物动力学。

FF 更新最大化一对正负样本的平均局部二元对数似然，goodness 为 ReLU 活动平方和；正负分量均使用同一旧 weight/bias。平方的导数系数 2 与 pair mean 的 1/2 抵消。它无需自动微分，但数学上仍是局部梯度；不直接适用于硬脉冲层。epsilon 必须为正，rate 应为正且有限；这些配置由宿主或模型声明保证。数值溢出按基础原语合同报错。

完整可运行定义与调度接口见[无反传学习](local-learning.md)。

## 自定义计算模块与依赖闭包

自定义文件使用 `operators NAME`，只允许显式参数、不可重复的不可变 `let` 和最终 `return`。例如 `blocks.wgt`：

```wgt
wgt 0.5
import activation from "std/activation"
operators Blocks
# Inputs are already projected, equal-length feature vectors; no hidden weights.
fn gated_residual(input, gate, value) {
    return input + activation.swiglu(gate, value);
}
```

模型文件用 `import blocks from "./blocks.wgt"`，在 forward/loss 调用 `blocks.gated_residual(...)`。模块可嵌套导入纯计算模块；宿主 `module NAME`、模型入口、参数捕获、递归、async、I/O 和训练语句均拒绝。调用内局部名称隔离，导出函数改名后链接，重复别名报错。当前没有包注册表解析或自动下载。

`ModelDefinition::parse` 与内嵌 `model!` 只解析内嵌标准逻辑模块，不隐式读取当前目录。`ModelDefinition::from_file` 与 `include_model!` 另允许 `./` 文件导入，相对声明导入的文件目录解析；路径不得含 `..` 或绝对路径，符号链接解析后不得越过根模型目录。循环依赖、超过 32 层或 256 个模块、总源码或最终展开源码超过 1 MiB 均报错。同一物理依赖只追踪一次，各导入别名仍保持独立命名隔离。

文件解析将全部依赖链接为自包含合法 WGT 源码，保存在 `ModelDefinition::source()`。该源码可在没有源目录的服务端再次解析，不信任客户端附带的形状元数据。模型声明、形状、参数初始化文本保留；纯计算表达式通过共享 AST 重新输出，不对 Rust token 文本做替换。解析与宏展开只验证初始化器，不物化参数存储；实例化仍是显式服务边界。

`include_model!` 在编译期读取完整依赖闭包，嵌入闭合源码，并为根文件及每个文件依赖生成 `include_str!` 追踪。运行生成产物不寻找开发机模型文件。宏默认使用 `::wgt::ModelDefinition`，重命名 facade 时用 `crate = alias;`。原始字符串保留 WGT 换行、注释和标点；宏在运行表达式时仍重新解析闭合源码，不是预编译 IR，也不承诺零解析成本。修改模块文件会触发 Rust 重编译和重新验证。

## 范围

旧宿主 native prelude、OpenResponses harness、带 main 的数学演示和旧图占位不属于模型标准库。原宿主注册元数据若仍被退役代码的构建引用，只是内部迁移设施，不是新模型 API。标准库不提供完整 Transformer、卷积、RoPE、环境、优化器或网络框架占位；新增架构计算必须有真实形状、参数所有权、状态和梯度合同及行为验证。
