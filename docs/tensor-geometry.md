# 固定 shape 张量几何与多轴 attention

> **状态：指定范围已在本地验证，尚未发布。** 最终 IR 安全校验后 core 7 suites / 30 tests、真实 service 集成 1 项、macro fixture 编译 1 项通过。范围限于本固定 shape attention fixture，不代表完整 Transformer、多头 attention、动态 batch 或 GPU 路径。

## attention 本地验证记录

最终 IR 安全校验后，`wgt check` 已实际通过此专题中的完整 attention 模型；syntax-tour 中抽取的 T10/T11 也通过检查。`tensor-geometry.md` 与 learnable 专题中的三个完整模型均通过解析。workspace all-targets 构建报告 0 errors、4 warnings；standard/custom ops 相关检查 8 项通过。这些结果来自主会话实测记录。

真实 service 的 native-cpu 路径完成 100 步 SGD，MSE 从 `0.046146184` 降至 `0.00000076705595`，并验证权重 roundtrip、同 SGD 续训和资源释放。debug 单机 1000 次 evaluate 共 `2.033213 ms`（约 `2.03 μs/次`，含返回 `Vec`）；managed tensor 计量为 1128 B，其中参数 96 B、tape 1056 B，charged work 为 98。这些数值只描述该次运行，不是 benchmark 优势、进程 RSS 或真实峰值内存。

当前 kernel 使用朴素 f64 累加而非 BLAS，并复用 `AxisGroups`。将 `let k = q` 已验证输出和梯度相同且 work 更少；目前没有全局 CSE 或活跃区间复用，compiled workspace 各 register 常驻，session 仍保留 reference buffers，因此不宣称内存最优。


固定 shape 几何算子用于模型内部张量计算。WGT 支持局部张量类型注解和命名轴，例如 `let q: f32[sequence=8, hidden=32] = ...;`。注解在编译期检查 rank、各轴长度和重复轴名；命名轴用于矩阵收缩、Softmax 归约与轴置换的一致性检查，不增加运行时数值数据。未命名维度可写为 `f32[8,32]`。

## 直接在张量后面写形状

WGT 0.5 的 `value as [d1, ..., dn]` 将目标形状放在值后面，适合函数实参中的高频 reshape。它直接展开为既有 `reshape(value,[d1,...,dn])`，保持元素顺序和 dtype，并清除结果轴名；需要命名轴时使用局部类型注解。目标为 1..8 个正整数字面量轴，每轴 1..16384，总元素数不超过 16777216，且与输入相同。所有维度显式给出，变量、算术表达式和 `-1` 自动推断均不在此合同内。

```wgt
let activity = ff.dense_relu(weight as [2, 2], bias as [2, 1], input as [2, 1]);
let flat = activity as [2];
```

上面的第一行中，weight 的行表示输出单元、列表示输入特征，bias/input 是列向量；完整可运行定义见 [FF model.wgt](../examples/native_training/non_backprop/ff/model.wgt)。输出 activity 为 `[2,1]`，返回向量时写 `activity as [2]`。

形状后缀与 `.T`、索引和调用按书写顺序结合，优先于一元及二元运算。例如 `x as [2, 3].T` 先得到 `[2,3]` 再转成 `[3,2]`；`a @ b as [3, 2]` 只重塑 `b`，重塑整个乘积写 `(a @ b) as [6]`。`reshape(x,[...])` 继续支持，二者共用 Reshape IR 和后端执行路径；Rust 与 Python 都通过共享前端读取该语法，typed 操作保留无 AD 合同。

## 矩阵运算与命名轴

WGT 提供原生矩阵乘法运算符 `@` 与 rank-2 转置后缀 `.T`，它们直接 lowering 到既有 Matmul/Transpose IR，不需要调用 `matmul(...)` 或 `transpose(...)`。例如：

```wgt
wgt 0.5
model NamedGeometry
input x: f32[6]
output y: f32[4]
fn forward(x) {
    # 同一输入按查询和键两种角色命名；这里没有参数初始化或训练。
    let queries: f32[queries=2, hidden=3] = x;
    let keys: f32[keys=2, hidden=3] = x;
    # 收缩 hidden，保留 queries/keys，沿 keys 独立归一化。
    let scores = queries @ keys.T;
    let probabilities = softmax(scores,"keys");
    return reshape(probabilities,[4]);
}
```

`@` 与内置 `matmul` 共用 `TensorType` 的矩阵合同：两侧必须是 rank-2 张量，且 `A:[M,K]`、`B:[K,N]`；输出 shape 为 `[M,N]`。当两侧收缩轴都命名时，名称必须一致；输出保留两个未收缩轴的名称。输出轴名也必须唯一，因此命名的查询/键使用示例中的 `queries` 和 `keys` 区分。不支持 batch broadcasting 或隐式维度扩展。`softmax(x,"keys")` 可直接引用这些传播到结果的名称。`@` 始终表示内置矩阵乘法，不受名为 `matmul` 的子模型实例影响。
通用轴置换写为 `permute(x,[heads,sequence,dim])`，列表表示输出轴依次来自输入的哪个轴；也可使用零基整数轴索引。轴必须构成完整排列。
`batch_matmul(a,b)` 支持 rank-3 `[B,M,K]` 与 `[B,K,N]`，输出 `[B,M,N]`；两个 batch 维相等，已有名称必须一致，任一侧提供的 batch 名称都会保留到结果，不支持 batch 广播。`.T` 仅接受 rank-2 张量并交换两个轴，同时移动轴名。矩阵乘法优先级与 `* / %` 相同，左结合；后缀 `.T` 优先于二元运算。旧的 `matmul(a,b)` 和 `transpose(x)` 调用继续兼容。

保形单输入算子和 `let copy = value;` 保留轴名；同形逐元素运算按位置合并匿名与命名轴，拒绝名称冲突及合并后重复的名称。同一个寄存器的赋值和条件分支也遵守这一合同，已有名称不会被后续匿名结果清除。显式 `reshape` 产生未命名轴，新绑定的类型注解可重新声明名称。LayerNorm 的 gamma/beta 必须是所选轴宽度的向量；当仿射向量与归一化轴都命名时，名称必须一致。这些源级规则复用 [Tensor Contract](tensor-contract.md)，正反例见 [源级 Tensor 回归](../crates/wgt-language/tests/tensor_source_contracts.rs)。

`gate`/`route` 的同形概率张量也参与按位置的命名轴合并与冲突检查，`route` 先检查两支值的合同。原有 `f32[1]` 门保持逐元素应用语义；`f32[1,1]` 等形状不会自动变成标量门。

模型服务入口仍只接受一个 `f32` 向量；需要矩阵输入时，当前仍须在 forward 内显式 `reshape`。返回值仍须为向量，因此矩阵输出应显式 reshape 回一维。当前版本尚未实现多轴索引、动态 Shape 和通用广播。原生 CPU interpreter 将 `reshape`、`.T` 与 `permute` 表示为带 base/offset/shape/strides 的 TensorView；`@` 与 `batch_matmul` 直接消费 strided view，不复制输入数据。其他不支持 strided 输入的算子会按需物化。AOT compiled-forward 目前使用物化回退路径，因此零拷贝保证仅适用于 interpreter 的 view-aware 路径。

| 表达式 | 合同 | 限制 |
| --- | --- | --- |
| `x as [d1,...,dn]` / `reshape(x,[d1,...,dn])` | 按行优先元素顺序重解释 shape，清除结果轴名 | 目标为 1..8 个正整数字面量轴，每轴 1..16384，元素总数不超过 16777216；元素数须与输入相同；所有维度显式给出 |
| `x.T` / `transpose(x)` | rank-2 张量交换两个轴 | 不接受其他 rank；两种写法等价 |
| `permute(x,[...])` | 按输出轴顺序置换任意 rank 1..8 的张量轴 | 轴索引必须构成完整排列；interpreter 使用零拷贝 strided view，其他算子按需物化；AOT 路径当前物化 |
| `a @ b` / `matmul(a,b)` | `[M,K] @ [K,N] -> [M,N]`，保留未收缩轴名 | 两边必须是 rank-2，内维及已有内轴名相等，结果轴名唯一；无广播；两种写法等价 |
| `batch_matmul(a,b)` | `[B,M,K] @ [B,K,N] -> [B,M,N]`，保留 batch 及未收缩轴名 | rank-3，batch 维和已有名称相等，无 batch 广播 |
| `softmax(x,axis)` | 沿指定整轴独立归一化；稳定计算，沿轴使用 f64 累加 | axis 为范围内的非负整数字面常量或已声明的轴名字符串；`softmax(x)` 与 `log_softmax(x)` 仍只接受 rank-1 向量 |
| `scale(x,literal)` | 以有限有符号数值字面量缩放张量 | 不广播，shape 保持不变 |

实际使用新几何算子或多轴 softmax 时使用 adaptive IR v3；旧 IR v1/v2 的读取规则不因 AST 中出现新 intrinsic 而全定义升级。参数 `read` 与裸参数引用始终产生 flat vector；矩阵参数必须在局部显式写 `reshape(parameter,[...])`。这里没有承诺动态 shape、广播、任意 dtype 或 rank-2 以上 matmul。

## 完整模型示例

以下完整模型与 `crates/wgtime-core/tests/fixtures/attention/model.wgt` 一致：两 token、每 token 三特征；Q/K 共用 `[3,2]` 投影参数，V 和输出投影为 `[3,3]`。参数读取始终是 flat vector，矩阵权重在局部显式 reshape。参数是定义中的初值，不代表训练后的权重。已提供的本地验证覆盖 attention 四项、native-cpu daemon 端到端一项（100 步 loss 下降、权重 roundtrip、同 SGD 续训及资源释放）和 macro fixture 编译一项；核心补充 IR 安全校验后须再次回归。

```wgt
wgt 0.5
model TinyAttention
input x: f32[6]
output y: f32[6]
# Two tokens, three features; Q and K deliberately share a [3,2] projection.
# These are declared initial values, not trained weights. V has three channels.
parameter qk: f32[3,2] = [0.3,-0.2,0.1,0.4,-0.3,0.2]
parameter value: f32[3,3] = [0.2,-0.1,0.3,0.4,0.2,-0.2,-0.1,0.3,0.2]
parameter output: f32[3,3] = [0.5,0.1,-0.2,-0.1,0.4,0.2,0.3,-0.2,0.5]
fn forward(x) {
    # Flat parameter reads are explicitly reshaped before matrix operations.
    let tokens = reshape(x,[2,3]);
    let projection = reshape(qk,[3,2]);
    let values = reshape(value,[3,3]);
    let outputs = reshape(output,[3,3]);
    let q = matmul(tokens,projection);
    # Q/K share the projection weights; reusing q here shares a computed value, not a new weight mechanism.
    let k = q;
    let v = matmul(tokens,values);
    # [2,2] scores reduce over keys (axis 1), independently for each query.
    let scores = scale(q @ k.T,0.7071067811865476);
    let attention = softmax(scores,1);
    let context = matmul(attention,v);
    # Output projection preserves tokens; the host contract stays one vector.
    return reshape(matmul(context,outputs),[6]);
}
fn loss(prediction,target) { return square(prediction-target).mean; }
```

将完整模型块抽取到独立 `model.wgt` 时，保留 `wgt 0.5` 版本头与完整参数/函数定义。以下资料同步了同一组规则：

- [语言参考](reference.zh-CN.md) 的 forward/loss 张量表达式契约。
- [Rust 后端契约](native-backend.md) 的 IR 版本和 compiled forward 验收状态。
- [语法总览](../language/syntax-tour/model.wgt) 的 T10 注释示例；不改活跃主模型。
- [wgt-development 技能](../.agents/skills/wgt-development/SKILL.md) 的固定 shape 与多轴规则。
- [prelude.json](../language/prelude.json) 中张量 intrinsic 的合同元数据。

## AdaptiveSession batch inference API

`AdaptiveSession::evaluate_batch(inputs, batch_size, max_steps)` accepts a contiguous
row-major `[B,input_len]` buffer and returns `[B,output_len]`. The method validates
the complete input size and finiteness before execution, uses one shared parameter
snapshot, preserves sample order, and charges one total instruction budget across
all samples. Each sample is currently executed independently in sequence; this is
a host-level batch API, not a fused batched operator or a batch-aware convolution
kernel. Rank-3 `batch_matmul` remains a separate tensor operation.
