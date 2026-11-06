# M1：固定形状 Tensor Contract

本变更提供一套可复用的固定形状元数据与校验代码，不是新增一套张量数值存储。`wgt_language::tensor` 定义 `TensorType`、`DType`、`ShapeLimits` 和 `Layout`；源码 lowering、Adaptive IR、CPU view 和旧制品通过明确入口复用它。权重身份、梯度所有权、优化器状态与服务实例仍属于原有实现。

**验证状态：实现候选，尚未在交付环境执行 Cargo。** 新增的 Rust 测试、工作区检查与既有回归必须全部运行后，才能确认可合并。Node 数值模拟和清单检查不能代替 Rust、真实服务或 GPU 验收；本文件不构成发布声明。

## 范围与兼容性

WGT 源码版本仍是 0.5；Adaptive IR 仍读取 v1/v2/v3。`Register` 的 JSON 字段仍为 `name`、`len`、`shape`，不添加必填字段。旧制品的 `dtype/shape/axes` 线格式也不改变。

同一类型实现不代表所有入口接受相同能力。原生寄存器要求正整数固定维度；旧制品元数据仍允许标量与空轴。入口使用不同 `ShapeLimits`，而不是把旧制品规则直接放宽到执行核心。

| 入口 | Rank | 单轴 | 元素数 | 标量 / 空轴 |
| --- | --- | --- | --- | --- |
| `REGISTER` | 1–8 | 1–16384 | ≤16777216 | 拒绝 |
| `PARAMETER` | 1–8 | 正整数，受总数约束 | ≤16777216 | 拒绝 |
| `VECTOR` | 1 | 1–16384 | ≤16384 | 拒绝 |
| `ARTIFACT` | 旧格式兼容 | 可含 0 | `usize` 可表达且乘法不溢出 | 保留 |

所有维度乘法、地址乘加都检查溢出。旧制品在宿主索引宽度下不可表示的维度明确拒绝；含零轴的元数据元素数为 0，不因维度排列顺序而误报中间乘法溢出。标量 `[]` 元素数为 1，但不因此成为原生向量入口。

## 类型、命名轴与算子

`TensorType` 的字段不可直接修改，只能通过校验后的构造与转换创建。逐元素规则要求相同 dtype 和 shape；命名轴按位置兼容，两个已命名轴必须同名。未命名轴可以被明确名称细化；细化后出现重复名称则拒绝。不存在隐式改名、按名称重排、类型提升或广播。

`matmul` 保持 `[M,K] × [K,N] → [M,N]`；`batch_matmul` 保持 `[B,M,K] × [B,K,N] → [B,M,N]`，不广播 batch。矩阵收缩轴和 batch 轴都检查维度及已有名称。`permute` 要求完整排列；`reshape` 要求元素数不变。LayerNorm 的 gamma/beta 必须为所选轴宽度的向量。

`DType` 枚举表示逻辑元数据，不代表后端支持。当前原生路径仍执行 f32。f16、bf16、i8 等枚举成员不开放对应 WGT 端口，不隐式转换，也不定义 INT8/INT4 量化格式；旧制品适配器仍只接受原有 f32、i64、bool。

源码类型注解现在 lowering 为明确的 Reshape/view，而不是修改已经产生结果的算子的输出形状。例如：

```wgt
wgt 0.5
model Annotated
input x: f32[6]
output y: f32[6]
fn forward(x) {
    # tanh 仍产生 [6]；matrix 是其 [2,3] 视图，轴名用于编译期检查。
    let matrix: f32[rows=2, cols=3] = tanh(x);
    # 转置先保持 [3,2] 几何，再按逻辑顺序得到向量。
    let flat: f32[6] = matrix.T;
    return flat;
}
```

这个完整例子同时进入语言与运行时回归测试；当前交付未执行这些 Rust 测试。注解显式形成视图时使用 IR v3；不会把所有未使用张量几何的旧程序自动升级。

## Layout 与物化

`Layout` 只定义元素索引如何映射到调用者已有的存储，不持有数值，也不分配 Weight identity。校验包括 rank/stride 数量一致、偏移、最大访问地址、存储边界与整数溢出。CPU 内部仍保留固定大小数组的 `TensorView`，其入口调用同一无分配的布局校验函数；不会为每次 view 增加一个堆上的 Layout 对象。

连续 reshape 可以共享原存储。非连续 reshape 必须先按逻辑顺序物化，不能只修改 shape 来伪装零拷贝。连续性判断忽略长度为 1 的轴的任意 stride。transpose/permute 保留原存储索引关系；Matmul/BatchMatmul 继续直接读取已支持的 strided 输入；AOT 的物化边界不改变。

零 stride 可表示只读重叠索引，但不自动开放广播操作。`is_non_overlapping` 仅提供保守的“该视图内部不重叠”证明，不判断两个不同 view 是否共享内存，也不授权原地写入。参数共享与 Tensor view 仍是不同概念。

## 可检查的算子清单

[`language/tensor_contracts.json`](../language/tensor_contracts.json) 为当前 44 个 Adaptive `Instruction` 逐项记录最低 IR 版本、shape、layout、梯度与副作用边界。Rust `Instruction::contract()` 使用穷尽 match，新 enum 变体不能无声漏掉。

```sh
node scripts/check_tensor_contracts.cjs --write
node scripts/check_tensor_contracts.cjs
```

检查器同时对照真实 enum 和生成的 Rust 清单。它不把 `language/primitive_specs.json` 的旧制品 graph 原语混作 WGT 内置函数，也不把目录存在解释为 GPU 或 AD 已验收。

`Program::validate_tensor_contracts()` 在源码 lowering 结束和 CPU prepare 入口调用。共享 pass 覆盖寄存器、参数形状、几何/逐元素/归约/卷积/池化检查和引用存在性；初始化、控制流、内存句柄、参数可写性与事务语义仍由已有运行时校验负责。这不是一个完整的新 Semantic/Execution Graph 系统。

## 既有编译缺陷与梯度边界

补丁针对基线 CI 的缺失 `bail!` 导入、三个 Permute 模式字段数错误、workspace 自由函数误用 `self`、不存在的 `Tensor.name`、LayerNorm VJP 重叠借用提供修复；同时补齐 workspace 对两个 pooling 变体的匹配。是否消除所有工作区错误仍以实际 Cargo 结果为准。

LayerNorm 的输入、gamma、beta 可以引用同一个 tape node。VJP 从不可变原值计算标量贡献，再分别累加到对应梯度；不假定三个 ID 不同，不引入 unsafe 或未经预算的临时梯度 Vec。新增测试覆盖全部别名对、非末轴以及有限差分。

自定义 `UpdateStrategy` 的参数名称从已有 Session 名称索引获得，不新建第二份权重身份系统。该入口目前只能替换 resident 参数；对 View/Add/File/Packed 显式拒绝，避免把共享存储分离或替换为空数组。拒绝发生在梯度计算和策略提交前，参数、步数、策略状态保持不变。普通训练及原有 View/Add 功能没有被移除。

## 验收入口与尚未完成的部分

新增测试分别位于三个 crate 的 `tests/tensor_contracts.rs`，以及 core 的 `tests/tensor_strategy_contracts.rs`。它们覆盖限制边界、溢出、命名轴、dtype 拒绝、布局地址、排列属性、旧 JSON 兼容、注解生产者形状、独立数值参考、有限差分和失败不提交。既有 `permuted_batch_view_feeds_batched_matmul_without_materialization` 回归保留。

工作区编译检查、语言测试、CPU/AD/compiled 回归与旧制品测试已加入 CI 候选变更；不取消原有编辑器/prelude 检查。交付包的 `verify.sh` 列出完整命令，并要求该具名回归确实存在，不能用筛选到零项测试代替通过。

动态/符号 Shape、通用广播、多 dtype 内核、JVP、自定义梯度注册、跨后端别名分析、GPU 性能、Weight ABI 运行时绑定与统一执行图不在本实现候选中。M1 的运行时验收也仍待执行；不能把本补丁描述为整个架构路线图完成。
