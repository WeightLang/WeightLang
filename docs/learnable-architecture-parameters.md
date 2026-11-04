# 可学习的模型架构参数（本地实现）

> 当前本地定点验证已通过：`learn_architecture` 5 项覆盖解释执行、真实 C 编译、checkpoint 与负向 round；`compiled_forward` 和 `adaptive` 定点测试通过。Rust facade 真实 daemon 集成测试 1 项覆盖三种模型入口、五种参数类型训练/推理，以及独立实例保存、加载和继续训练，并验证 `model!` 与 `std/loss` 半平方损失从 5 更新到 5.5。以上仅覆盖列出的局部范围，不代表全量验收；实现尚未提交或发布。

Rust 主程序负责数据准备、训练循环和应用控制；WGT 模型定义描述结构、参数、张量计算及可选 loss。顶层 `learn` 声明可训练模型架构参数。它不同于旧通用 WGT 宿主脚本中的整批 `learn agent { ... }`：后者采集记录并调用 learner 函数，本专题的 `learn` 为模型定义中的标量可学习参数声明。

## 语法

以下五种声明形式已实现，并在上述定点范围内验证：

```wgt
learn gain: float = 5.0;
learn bounded_gain: float in 2.0..=8.0 = 5.0;
learn positive_gain: float > 0 = 1.0;
learn enabled: bool = true;
learn depth: int in 1..=12 = 6;
```

- `learn name: float = initial` 声明无显式投影边界的连续标量参数，接受任意有限 f32 初值。
- `learn name: float in lower..=upper = initial` 要求有限端点且 `lower < upper`，初值必须在闭区间内。每次 SGD/Adam 更新后将 shadow 投影回闭区间。
- `learn name: float > lower = initial` 要求有限 `lower >= 0` 且低于最大有限 f32，初值必须严格大于 lower。每次 SGD/Adam 更新后按 `max(candidate, next_up(lower))` 投影。`> 0` 时 `next_up(0)` 是最小正 subnormal f32；这不保证该值适合下游数值计算。
- `learn name: bool = initial` 的初值只能是 `true` 或 `false`；shadow 限制在 `[0,1]`，前向 `shadow >= 0.5` 为 true。
- `learn name: int in lower..=upper = initial` 要求整数端点、`abs(endpoint) <= 16777216`、`lower < upper`，且初值是范围内整数；不支持无界整数声明。

声明必须位于模型顶层，每行一条并以分号结束。`learn` 声明不改变模型输入或输出的 shape，也不引入 choice/categorical 参数；当前不支持 choice。

## 连续参数的投影

连续 `float` 参数的优化状态是 shadow 值。无显式范围时不作边界投影；有界声明在优化器更新之后投影更新后的 shadow：

- 闭区间：`project(x) = min(max(x, lower), upper)`，将更新后的值限制在包含两端的区间。
- 严格下界：`project(candidate) = max(candidate, next_up(lower))`，`next_up` 表示大于下界的下一个可表示 f32；对于 `lower = 0`，这是最小正 subnormal f32。

这不是仅在前向读取时临时投影：SGD/Adam 完成更新后，持久 shadow 状态即按对应范围投影，前向读取该参数状态。连续 float 按普通可微路径传播梯度；离散化读取的 int/bool 才使用恒等 STE，且边界处不清零该代理梯度。

## 整数与布尔参数的 STE

整数和布尔参数分别持有连续 shadow 存储，并在前向使用硬离散值：整数使用 Rust `round`（半整数远离零），布尔使用 `shadow >= 0.5` 转换。反向传播使用 identity straight-through estimator（STE），即把离散化步骤的上游梯度原样传给连续 shadow。

STE 是训练代理梯度，不是 round 或阈值函数的数学真导数。硬前向值与连续可训练状态是两个不同概念：例如布尔前向只产生 true/false，但优化器仍更新连续 shadow。保存模型权重时保留 shadow，使离散参数能继续训练；参数元数据属于模型定义并须在定义解析时验证。

Rust `round` 的半整数按远离零处理，仅在读取 int 前向值时舍入，不会在优化器更新后额外 round shadow。bool shadow 保持在 `[0,1]`，且 `shadow >= 0.5` 前向为 true；字面量初值 `false` 映射为 shadow 0，`true` 映射为 shadow 1。float 路径使用通常的连续梯度，不称作 STE。

## 自定义损失示例

可学习参数是标量。普通平方误差 `(prediction - target) * (prediction - target)` 对预测值的梯度是 `2 * (prediction - target)`；使用半平方损失可消去系数 2，使梯度恰为差值。目标 10、初值 5、学习率 0.1 时，半平方损失的一步 SGD 更新应得到 5.5：

```wgt
wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
learn scale: float = 5.0;

fn forward(x) {
    return x * scale;
}

fn loss(prediction, target) {
    let errors = square(prediction - target);
    return [0.5] * errors.mean;
}
```

对输入 `1`、目标 `10`，半平方损失关于 `scale` 的梯度为 `scale - 10`，在初值 5 时为 -5。使用学习率 0.1 的一次 SGD 更新会得到 5.5。该模型定义及 `model!`/`std/loss` 集成路径已在 daemon 定点测试中验证；此说明不扩展为全量验收声明。

## 有界深度与门控

`depth_gate(depth, index)` 仅可在 `tensor_forward` 中调用：第一参数必须是顶层有界 `learn int` 参数名，不接受局部 tensor、列表或计算式；第二参数是带符号的整数数值字面量，且满足 `1..=upper`，其中深度下界非负、上界不超过 256。返回 `f32[1]` 硬门值：`depth >= index` 时为 1，否则为 0；反向对 depth 使用值为 1 的代理梯度。它不能用于无参数的 loss 或纯 operators helper，也不改变张量 shape。

下面以三个显式计算段展示按深度启用后续段；三个线性段共享同一个冻结权重 `w`。这只是静态展开，不自动重复模块、不改变循环次数或张量 shape，也不表示一般 while 循环可微：

```wgt
wgt 0.5
model Depth
input x: f32[1]
output y: f32[1]
frozen w: f32[1] = [2]
learn depth: int in 0..=3 = 1;

fn forward(x) {
    let first = route(depth_gate(depth, 1), x * w, x);
    let second = route(depth_gate(depth, 2), first * w, first);
    return route(depth_gate(depth, 3), second * w, second);
}
```

## 参数与 shape 边界

参数定义不增加模型输入维度或改变中间/输出 shape；当前没有 `choice` 参数。整数和布尔训练使用 shadow 与 STE，STE 不是离散函数的数学导数。learn 元数据属于模型定义；权重文件保存连续 shadow，读取参数按 `f32[1]` 硬投影/离散化后可用于 `gate`、`mix`，但不可直接作为 `linear` 原始 bias，也不可用 `write` 写入。

加载权重须绑定相同 learn 元数据的模型定义；prepare 校验已加载值是否在声明域内，越界即拒绝。普通 SGD/Adam 更新按声明规则投影；自定义 gradient commit 对越界候选值报错，不会静默投影。重复/无效名称由模型参数 schema 校验拒绝，服务 prepare 也会检查重复项。
