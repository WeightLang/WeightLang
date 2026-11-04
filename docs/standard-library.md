# 模型内部标准库

WGT 声明模型结构与可微计算，Rust 程序负责实例化、数据、训练调度和应用控制。这里的标准库仅包含可嵌入模型的纯张量函数，不提供网络、数据读取、异步任务或工具调度模块。当前为本地 WGT 0.5 实现，尚未发布。

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
