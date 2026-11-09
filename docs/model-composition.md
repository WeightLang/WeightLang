# 模型组合与可学习控制

> **状态：WGT 0.5 本地实现，尚未发布。** 控制语法核心测试覆盖前向、梯度、独立实例、影子权重恢复及错误输入；一项真实 wgtime service 组合测试已通过 native-cpu 训练、权重恢复、续训一致性和资源释放。证据限于测试中的固定形状模型，不扩展为任意动态控制流或完整 MoE 的承诺。

模型计算使用固定 shape 张量，没有全局 broadcast。有限数值字面量乘张量复用 `scale`；平方支持 `**2`，等价 `square`。任意标量与张量广播、任意指数幂均不由此提供。

## 可学习参数

`learn name: domain = initializer;` 声明可训练参数。当前 lowering 接受 `float`、`float > low`、`float in low..=high`、`int in low..=high` 和 `bool`。数组形式把 shape 写在统一类型之后，例如 `int[2] in 0..=3`；shape 是 `1..=16384` 的固定宽度整数。初始化可写单个标量并填充所有元素，或写长度必须恰好为 `N` 的列表：

```wgt
learn gate: float[2] = 0.5;
learn bias: float[2] = [0.1, -0.1];
```

数值初始化按每个元素分别校验 domain。`bool` 初始化只接受 `true` 或 `false`；普通浮点初始化须有限。整数边界和初值须为可精确表示的 f32 整数，范围限于 `-16777216..=16777216`。数组 shape 当前 lowering 为一维 `[N]`；不得据此推断多维 learn 数组支持。

## 混合、条件与重复

### 加权混合

语法是 `blend { weight => tensor; ... }`，至少包含一个以分号结尾的臂。Lowering 将每个臂转为 gate，再逐项相加；权重不自动归一化。每个 gate 的权重 shape 必须为 `[1]` 或与其张量完全相同，混合臂结果 shape 完全一致。仅当两臂权重是同一个简单名称 `p` 与 `(1-p)` 时，lowering 会优化为互补 `Route`；其他表达式保持一般 gate 加和。不同 shape 即使元素数相等也拒绝混合。权重须有限且位于 `[0,1]`；核心测试覆盖非归一化加权、非法权重和混合参数梯度。

### 条件选择

表达式形式为 `if condition { yes } else { no }`，两个分支结果 shape 必须相同。条件可为 bool，或 scalar 比较 `>`、`>=`、`<`、`<=`、`==`、`!=`。任意 float 张量裸值不能直接作为条件。`learn bool` 参数可作为学习条件；`learn int` 与整数字面量比较 `depth >= threshold` 也会走学习条件 lowering。普通比较没有条件梯度。Learn bool 前向硬读取，反向使用直通估计器（STE）传递连续影子参数的梯度。推理可跳过未选中的内联分支；训练为学习条件保留候选计算。

### 有界重复

形式为 `repeat depth [indexed i] carrying state=initial { body }`；`indexed i` 可省略，且索引名与 carrying 名不能相同。编译期按 `0..maximum` 展开，索引绑定为零基迭代常量，携带值绑定为当前状态；每轮结果须保持携带张量的 shape。常量深度必须是 `0..=256` 的整数。学习深度必须是有界 `learn int`，其声明区间须落在 `0..=256`；候选迭代按上界展开。学习深度候选步使用阈值硬门与 STE：训练计算全部候选，推理跳过未启用的内联步骤。核心测试以会溢出的未启用循环体验证推理跳过，并验证训练仍会计算该候选且报错；降低有效深度不会减少编译期展开规模。

### 张量运算限制

- 二元张量加、减、乘要求长度和 shape 完全一致，没有全局 broadcast。
- `**` 仅接受右侧字面量 `2`，并转换为 `square`。
- `scalar * tensor` 或 `tensor * scalar` 中的 scalar 必须是有限数值字面量，转换为 `scale`。

## 子模型定义与实例

以下普通 forward 组合规则保留兼容。具有 state 或 typed 函数的模型另支持具名操作互调、嵌套有状态实例及整体事务，见[模型操作](model-operations.md#操作与更新合同)。两种入口都使用共享 tensor lowering；无 AD 的操作不会借助旧可微子模型路径隐式求导。

源文件可以同时包含顶层 `model Root` 定义、子模型块 `model A { ... }` 和函数。子模型块逐行写声明与函数；当前 lowering 读取子模型内部的模型声明，并要求子模型提供 `forward`。根模型通过 `experts: A[8];` 声明实例数组。子模型首期 forward 只接受一个输入，并限于 immutable `let` 与 `return`；不支持可变赋值、嵌套子模型定义、在子模型内声明其他模型实例、递归实例或子模型 loss；可以调用根文件显式导入的纯 helper。每次调用都校验其声明的 input/output 向量合同；未被调用的定义以及 schema-only 验证只检查定义，不初始化实例或权重。

数组实例通过静态索引调用，例如 `experts[i](x)`；标量实例可写 `encoder(x)`。索引接受编译期非负整数表达式或 repeat 索引，范围上限检查到已声明实例数。实例化为每个索引复制独立参数，稳定命名形如 `experts_0_weight`。同一索引的重复调用引用同一组参数；不同索引有各自参数。生成参数名与已有参数发生碰撞时拒绝定义。

## 完整组合模型草稿

以下片段展示 learn 数组初始化、子模型数组、静态 repeat 索引调用、混合、条件、平方与 loss 的组合。该完整模型已通过本地定义解析，尚未单独执行训练验收。注释中的维度说明向量长度；初始化值不是训练后权重。

```wgt
wgt 0.5

# 每个专家将四元素向量映射为四元素向量。
model Expert {
input x: f32[4]
output y: f32[4]
# 子模型声明的初始值；实例化后每个 expert 拥有独立命名的参数。
parameter weight: f32[4] = [0.5, 0.5, 0.5, 0.5]
fn forward(x) {
    # 逐元素缩放，输出仍为四元素向量。
    return x * weight;
}
}

model Root
input x: f32[4]
output y: f32[4]
# 根模型拥有八个 Expert 实例；不同静态索引对应不同参数。
experts: Expert[8];
# 等长列表初始化两个独立元素；索引 0 用作混合概率。
learn mixture: float[2] in 0..=1 = [0.5, 0.5];
learn enabled: bool = true;
learn depth: int in 0..=8 = 2;

fn forward(x) {
    # 按零基索引展开；每轮调用对应的独立专家并更新携带向量。
    let repeated = repeat depth indexed i carrying h=x { experts[i](h) };
    # 第 0、1 个专家已在前两轮使用；再次调用分别复用各自的参数。
    let first = experts[0](repeated);
    let second = experts[1](repeated);
    let mixed = blend { mixture[0] => first; (1 - mixture[0]) => second; };
    # 条件两臂均返回四元素向量。
    return if enabled { mixed } else { repeated };
}

fn loss(prediction, target) {
    # 逐元素平方误差后取均值，得到标量损失。
    return 0.5 * ((prediction - target) ** 2).mean;
}
```

示例中 `mixture` 的声明展示有界 learn 数组及按索引读取；具体条件域、数组权重梯度、重复专家调用与子模型组合的完整行为仍待运行时训练验收。当前 lowering 会在展开实例参数时检查 `experts_0_weight` 一类命名与已有参数冲突。

## 卷积

卷积签名、布局、shape 公式、参数范围和示例见[卷积文档](convolution.md)。该文档记录 core 的 1D/2D/3D reference 与 compiled forward、输入/kernel/bias 梯度和边界测试通过情况；三项真实 wgtime service 测试也已通过训练、权重恢复、续训及资源释放，范围限于固定形状的卷积 fixture。

## 标准库边界

Loss、attention、block 与求解算法通过显式 `import` 使用；不添加预导入 preset。语法和必要的张量原子属于内建能力。各标准模块的可用函数与签名以模块源码为准，本页不虚构 API。

## 验证范围

`wgtime-core/tests/model_control_syntax.rs` 的十项测试已通过，包含参考执行与原生 C 执行、混合和矩阵形状、普通硬分支、学习条件、repeat 跳过、数组影子权重恢复、独立实例和参数共享、导入组合与非法定义拒绝。直接数学式 loss 的单步 SGD 验证了参数 `5 → 5.5`。

`wgt-model/tests/model_control_syntax.rs` 另以真实服务验证组合语法的初始化、前向、训练、safetensors 恢复、续训一致性及关闭释放。本页八专家完整示例已通过定义解析；没有据此声称它完成独立训练或性能测试。
