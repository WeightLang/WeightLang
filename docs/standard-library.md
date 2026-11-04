# 标准库（WGT 0.5，本地实现，未发布）

标准库包含原生预导入声明、可微张量计算库和普通脚本模块。目录中不存放模型入口、训练演示或尚不能执行的图结构占位。完整模型应在独立目录使用 `model.wgt`；导入标准库不会初始化或训练模型。

## 原生预导入声明

[`std/prelude.wgt`](../std/prelude.wgt) 是公开原生函数与张量 intrinsic 的声明来源，记录完整签名、参数默认值、适用上下文、用途及实现绑定；没有 WGT 函数体。`language/prelude.json` 是从这些声明和声明类型元数据生成的目录，构建会检查一致性。声明格式和生成规则见[预导入声明](prelude.md)。

普通脚本的 `dot`、`norm`、`softmax`、统计、排序、标量数学等原生函数直接可用，无需按分类导入。各函数的参数、数值域与执行上下文以[主参考](reference.zh-CN.md)和预导入声明为准；不能因为普通脚本支持某个函数，就在张量 forward/loss 中调用它。

`save weights NET to PATH [overwrite];` 和 `reload weights NET from PATH using OPTIMIZER;` 是语言语句，不是普通函数。权重文件不提供模型结构，重载要求已有兼容模型并显式指定优化器；详见[文件与持久化](file-io.md)。

## 可微张量库

显式导入稳定的逻辑模块名：

```wgt
wgt 0.5
import activation from "std/activation"
import losses from "std/loss"
model FeatureGate
input x: f32[2]
output y: f32[2]
parameter gain: f32[2] = [1, 1]
# x has one feature axis; gain is shared across every sample and invocation.
# The library receives projected features and allocates no parameters.
fn forward(x) {
    let projected = x * gain, gated = activation.swiglu(x, projected);
    return gated;
}
fn loss(prediction, target) {
    return losses.squared_error(prediction, target);
}
# This constructs initialized weights only; no training runs implicitly.
fn main() {
    let net = FeatureGate("sgd");
    return net([-1, 2]);
}
```

所有参数必填，没有默认值。输入为非空、定长的一维 `f32[N]`；`N` 是特征或类别轴，没有隐含 batch 轴。多个输入必须同形，不提供标量广播。归约输出为 `f32[1]`，与训练器对多个样本求平均分属不同轴。

| 逻辑模块 | 完整签名 | 输出与用途 |
| --- | --- | --- |
| `std/activation` | `swish(x)` | `f32[N]`，`x * sigmoid(x)`，sigmoid 使用稳定原语 |
| 同上 | `swiglu(gate, value)` | `f32[N]`，`swish(gate) * value`，调用方提供两路已投影特征 |
| 同上 | `squared_relu(x)` | `f32[N]`，`square(relu(x))` |
| `std/loss` | `squared_error(prediction, target)` | `f32[1]`，全部 N 个特征的平方误差均值 |
| 同上 | `half_squared_error(prediction, target)` | `f32[1]`，上述均值乘 0.5 |
| 同上 | `categorical_logits(logits, target)` | `f32[1]`，`-sum(target * log_softmax(logits))`，自然对数 |

`categorical_logits` 的目标应为非负且总和为 1 的独热或软标签；纯张量函数不检查目标分布，其他有限目标仍按加权公式计算。输入是 logits，不能用概率代替。减最大值的 `log_softmax` 避免先 softmax 再 log 导致的下溢；极大有限数的减法、乘法与平方仍可能溢出并报错，不静默裁剪。

这些函数仅在张量 `forward/loss` 内展开为现有原语，支持 CPU 一阶梯度和生成原生代码。普通脚本不能直接调用限定张量函数。模块不创建参数或状态、不捕获模型权重、不执行训练；参数共享和训练时机由调用方决定。初始化结构、已训练权重与优化器检查点是不同对象。没有据此提供 GPU、融合内核或性能保证。

实现源为 [`activation/operators.wgt`](../std/activation/operators.wgt) 和 [`loss/operators.wgt`](../std/loss/operators.wgt)。`/operators` 是内部组织路径，不是可导入的逻辑模块名。库中的批量 `let` 仍逐项从左到右求值和绑定，后项可引用前项；不新增张量批次或事务。

## 普通脚本模块

[`std/openresponses/harness.wgt`](../std/openresponses/harness.wgt) 声明 `module OpenResponsesHarness`，通过 `import harness from "std/openresponses/harness"` 导入。它复用普通脚本和 async 能力，不能在张量 forward/loss 中调用。

| 完整签名 | 返回与约束 |
| --- | --- |
| `harness.next(session, profile)` | async 函数，返回请求 future；在 async 函数中 `await` 得到协议 step。session 是当前 worker 的 opaque handle，profile 是已授权的名称 |
| `harness.submit(session, results)` | 返回 null；results 为完整的 `{call_id, output}` 列表。原生层整体拒绝重复、未知或缺失 ID |
| `harness.finished(step)` | 布尔值，`step.status != "tools"`；只判断无需继续提交工具结果，不代表响应成功完成 |

所有参数必填，无默认值。调用方先显式配置 `responses_scope` 并创建 session；scope 预算、worker 归属、权限收缩、网络与终止状态仍由原生 OpenResponses 实现管理。库不初始化模型、不持有全局可变会话、不自动选择或执行工具。`incomplete` 或 `failed` 等终态也会使 `finished` 返回 true，调用方仍须检查 `step.status` 及具体响应。使用方负责显式工具映射和取消会话。协议及完整循环见 [OpenResponses](openresponses.md)。

## 导入与能力边界

标准模块源码随运行时内嵌，导入不依赖当前目录或联网下载。未知 `std/` 名称明确报错；`..`、`.wgt` 后缀和内部 `/operators` 路径不是标准逻辑名。普通模块与纯张量算子分别校验，重复别名报错。开发阶段可显式导入 `./module.wgt`，路径必须留在源码目录内，不能覆盖标准逻辑名。

普通模块只允许一条 `module NAME` 声明和函数，不允许再导入其他模块或声明模型入口；模块内函数调用以导入别名改写，避免不同模块辅助函数混淆。纯张量库只允许 `operators NAME`、不可重复的不可变 `let` 和最终 `return`；不允许捕获参数、递归、循环、async 或副作用。完整限制见[自定义算子](custom-operators.md)。生成的原生代码包含实际依赖计算，不需要运行时读取开发机标准库源码。

旧的 `nn`、`transformer`、`vision`、`recurrent`、RoPE、HF decoder、attention、delta-rule 等图占位已删除，不是受支持的导入别名。当前标准库没有提供多轴 attention、卷积、参数化层工厂、循环状态模块或完整模型家族；JSON 制品中的形状契约、解析成功与原生执行支持应分别判断。后续扩充必须先有明确输入、形状、状态、训练和可执行行为契约，不能用同名空模块代替实现。

## 本地验证

声明一致性使用 `node scripts/generate_prelude.cjs --check`、`node scripts/check_prelude.cjs` 和 `node scripts/test_prelude.cjs`。张量库测试使用内联 WGT 验证真实导入、数值边界、形状拒绝、训练梯度与代码生成；普通 harness 通过本地 HTTP mock 验证回合和共享预算，不请求真实服务。数值原生函数测试使用内联函数调用，不依赖被删除的标准库演示。具体执行结果随变更报告记录；这些验证不表示已发布或部署。
