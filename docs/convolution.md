# `conv`：无批次空间卷积

`conv` 是 WGT forward 张量表达式中的直接数值 kernel 原子，用于对通道优先的 1D、2D 或 3D 输入执行分组空间卷积。它规定 layout、shape 和可微的卷积计算；CNN block、激活、池化等更高层组合仍由模型作者显式写在 WGT 中。此处不定义或暗示已有预导入的卷积 block、pooling 或其他 CNN preset。可复用算法若要封装，应放在 `std/` 并显式导入；不会因此扩展预导入内容。

实现通过直接枚举输出与 kernel 的空间位置计算，不构造 `im2col` 矩阵。以下行为描述对应当前源码实现；适用边界和验证状态见文末。

## 签名

```text
conv(x, kernel, bias, stride, padding, dilation, groups)
```

- `x`：f32 输入张量，shape 为 `[Cin, S1, ..., Sd]`。
- `kernel`：f32 卷积核，shape 为 `[Cout, Cin/groups, K1, ..., Kd]`。
- `bias`：f32 张量，shape 必须为 `[Cout]`。
- `stride`：长度为 `d` 的正整数列表，每个值在 `1..=16384`。
- `padding`：长度为 `d` 的非负整数列表，每个值在 `0..=16384`；各空间轴两侧使用相同的零填充。
- `dilation`：长度为 `d` 的正整数列表，每个值在 `1..=16384`。
- `groups`：正整数，在 `1..=16384`；必须整除 `Cin` 和 `Cout`，且 kernel 的输入通道维必须恰好等于 `Cin/groups`。

空间维数 `d` 为 1、2 或 3。`x` 与 `kernel` 的 rank 分别为 `d+1` 与 `d+2`。这些列表和 `groups` 必须是 WGT 源码中的整数常量；实现不接受运行时张量作为选项。

## Shape 与索引

输出 shape 为 `[Cout, O1, ..., Od]`，其中每个空间轴 `a` 的输出长度为：

```text
Ea = dilation[a] * (Ka - 1) + 1
Oa = floor((Sa + 2 * padding[a] - Ea) / stride[a]) + 1
```

`Ea` 是膨胀后的有效 kernel 长度。若填充后的输入短于有效 kernel，shape 非法并在 lowering 时被拒绝。输入、kernel、输出沿通道优先、行优先顺序展平存储；卷积原子保留显式多维 shape。当前原生 model forward 入口仍是一个向量，因此通常先用 `reshape` 将该向量恢复为通道优先的输入几何，最后将输出恢复为声明的向量长度。

对输出通道 `o`，其所属组为 `floor(o / (Cout/groups))`，只连接该组对应的 `Cin/groups` 个输入通道。对输出空间位置 `p`，kernel 位置 `q` 对应输入位置：

```text
u[a] = p[a] * stride[a] + q[a] * dilation[a] - padding[a]
```

若任一 `u[a]` 落在输入范围之外，该输入/kernel 配对不参与求和，相当于读取零填充。运算是互相关（cross-correlation）：kernel 不沿空间轴翻转。每个输出元素为：

```text
y[o, p] = bias[o]
        + Σ(c = 0 .. Cin/groups) Σ(q1 .. qd)
            valid(u) ? x[group(o) * (Cin/groups) + c, u] * kernel[o, c, q] : 0
```

Kernel 权重在所有输出空间位置共享。偏置对每个输出通道加一次。

## 预算与资源

一次 `conv` 指令的执行工作预算为：

```text
Cout * (O1 * ... * Od) * (Cin/groups) * (K1 * ... * Kd) + 1
```

这是按完整通道和 kernel 位置数计费的上界；即使某些位置因 padding 越界而跳过乘加，也不会减少该预算。它计入 adaptive execution 的 `max_steps`。整数乘加使用 checked arithmetic，预算计算溢出会报错。

运行时直接滑窗枚举有效输入/kernel 配对，没有 `im2col` 暂存矩阵。输出写入已有 register scratch，再拷贝到输出 register。session 的 managed memory budget 仍覆盖参数副本、register 和最大 register scratch 等 f32 存储；编译后的 workspace 若启用也受 session 总预算约束。`im2col` 未创建不代表内存预算可以忽略。

## 完整 WGT 示例

下面是一个可独立抽取的 `model.wgt` 内容。输入是一维向量，模型将其恢复为 `[2, 2, 3, 4]` 的 3D、通道优先几何，使用两组卷积，再显式展平作为输出。参数值是定义内的初始化值，不代表已训练权重。

```wgt
wgt 0.5
model TinyCNN3D
input x: f32[48]
output y: f32[16]
# 输入通道为 2，空间轴长度依次是 2、3、4；核的空间尺寸为 2×2×2。
# groups=2 时每组连接一个输入通道和一个输出通道，kernel 的 Cin/groups 维为 1。
# 下列参数只是模型初始化值，不是训练完成或从 service 加载的权重。
parameter kernel: f32[2,1,2,2,2] = [-0.25,-0.1,0.05,0.2,-0.2,-0.05,0.1,0.25,-0.15,0.0,0.15,-0.25,-0.1,0.05,0.2,-0.2]
parameter bias: f32[2] = [0.02,0.04]
fn forward(x) {
    # stride=[1,2,1]、padding=[0,1,1]、dilation=[1,1,2]；卷积输出 shape 为 [2,1,2,4]。
    # 每个空间位置复用同一 kernel；之后才逐元素应用 tanh 并展平。
    let features = conv(reshape(x,[2,2,3,4]),reshape(kernel,[2,1,2,2,2]),bias,[1,2,1],[0,1,1],[1,1,2],2);
    return reshape(tanh(features),[16]);
}
fn loss(prediction,target) { return square(prediction-target).mean; }
```

此例的空间输出长度由公式得到 `[1, 2, 4]`：有效 kernel 长度为 `[2, 2, 3]`，填充后输入长度为 `[2, 5, 6]`。与输入通道无关，输出元素总数为 `2*1*2*4 = 16`，符合声明的 `output y: f32[16]`。

## 限制与验证状态

- 不支持 batch 轴；若有多个样本，调用方需逐样本组织输入。
- 输入和 kernel 使用通道优先 layout；只支持 1 至 3 个空间轴。
- Padding 是对称零填充；没有自动 same/valid 模式、非零 padding 模式或 kernel 翻转。
- 步幅、填充、膨胀和分组是编译期整数常量，不能由张量或模型运行状态动态选择。
- 这是卷积计算原子，不内置激活、池化、归一化或 CNN block。
- 当前 core 的 `tests/convolution.rs` 覆盖 1D、2D、3D 的 reference 与 compiled forward 对照、输入/kernel/bias 梯度对照、有限差分梯度、训练预算和 shape 边界错误。该测试调用的是 `AdaptiveSession`；它不等同于真实 wgtime service 集成验证。
- `wgt-model/tests/convolution.rs` 的三项真实 wgtime service 测试已通过，分别覆盖 1D、2D、3D native-cpu 训练、safetensors 权重保存和恢复、续训及资源释放。该证据限于固定形状卷积 fixture，不包含池化、归一化或完整 CNN 分类器。示例中的参数仍是定义初值，不是这些测试训练后的权重。
