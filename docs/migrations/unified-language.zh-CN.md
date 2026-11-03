# 统一 WGT 前端：迁移契约

目标是保留动态模型的表达能力，以 `model/input/output/block/layer` 张量图作为
唯一正式模型语言。`backend torch-dynamic` 暂作为旧源码和实验快照的兼容入口。
本页记录迁移工作，不表示已经完成统一；不能仅重命名关键字、统一入口函数，
或把整段 Python forward 包装成一个算子来宣称完成。

## 边界

- WGT 描述参数、配置、张量计算和显式状态；Python 管理数据、损失、优化器、
  采样、设备调度和跨请求状态生命周期。
- 统一前端不执行任意 Python、`extern` 回调或用户提供的 Torch 方法。
- 参数化模块使用 block、显式参数身份和共享规则；控制流必须有明确张量契约。
- 后端选择不能切换文件语法。旧 tensor-program 制品和保留的实验源码不改写。
- 先迁移并通过行为验收，再退出旧入口；编辑器高亮随后针对统一语言维护。

## Qwen3.5 源码差距

审计对象：`examples/bootstrap/qwen/model.wgt`，包含基础文本网络、Surgery、
Refinement 和 EditedTail。Python 宿主中的流式加载也依赖这些类及参数路径。

| 计算/契约 | 当前声明式能力 | 迁移要求 |
| --- | --- | --- |
| Linear、Embedding、SwiGLU、残差 | 已有 linear/embedding/silu/mul/add/block | 保持参数路径、共享 LM head 和前向/梯度 |
| offset RMSNorm | 已有 rmsnorm(offset=1) | 核对乘法及 cast 顺序，不能只比公式 |
| L2Norm、门控 RMSNorm、衰减 | 归约/广播已有；基础数学尚不全 | 可组合 square/rsqrt/exp/softplus/标量平移等 |
| 因果 depthwise Conv1d | 已有 conv1d(groups,causal) | 对照 padding 后截断的输出、前缀与梯度 |
| 分头、拼接、GQA 扩展 | 固定 slice/transpose/cat；有界 reshape 不完整 | 不把 batch 混入样本轴；动态长度和非连续张量 |
| 分块 gated delta rule | 有 scan_state 和显式状态；缺完整原子计算 | 保留 chunk 内更新、衰减、初态/终态及梯度；不可换成不等价 RNN |
| 部分 RoPE、带门控 GQA | 现有 rope_attention 不能直接等同该实现 | 部分旋转、Q/K offset norm、投影门、精度顺序 |
| 配置与层列表 | 有标量 config、block 和独立 repeat | 固定配置展开异构层；拒绝不支持的配置 |
| Refinement | 有共享 recur、残差和门控 | 零初始化、循环内共享、不同层独立、混合精度 |
| EditedTail / 子模型入口 | 旧类构造器提供切片入口 | 用明确的模型/子图接口替换类名绑定；冻结和流式调度不丢失 |
| 权重与实验快照 | 两种制品并存 | 显式参数映射；旧快照保持可重放；不覆盖任何保留模型 |

## 实施与退出条件

1. 补齐通用张量原语，并同步 Python/Rust 契约、运行时、独立导出和文档。
2. 将数学子图迁入标准库，逐个比对旧源码的前向和梯度。
3. 补齐形状变换、参数张量、分块递推、部分 RoPE 和精度契约，再迁移完整网络。
4. 迁移普通加载、分块加载、训练与评测调用；保留显式旧制品读取路径。
5. 同一组参数比较长度 1/7/65、因果前缀、梯度、优化一步、共享/冻结及保存重载；
   同时覆盖 Refinement/EditedTail 的零初始化和独立参数。
6. 当前示例不再使用第二套语法、普通入口不再自动切换到 Python 执行后，才称为统一完成。

小配置数值检查不代表 4B 实验验收或模型已经训练；不得为语言迁移启动完整训练。

## 当前增量

已补入 `square/rsqrt/exp/log/sin/cos/softplus/shift/cumsum`，以及
`std/tensor_math` 的无参数 L2Norm/RMS 子图。通用数学运算走原有声明式解析、
张量图、制品和导出链，不调用旧动态前端。Python/Rust 共用原子签名目录；
`shift.value` 增加有限浮点常量类型，其余轴/大小参数仍严格为整数。

本地检查：`tests/test_tensor_math.py` 与既有 `tests/test_primitives.py` 共 38 项通过；
Rust `primitives` 两项通过。覆盖前向/梯度、非连续输入、标量/空轴、非法 dtype/轴、
数值域、符号长度、累积衰减的分块等价，以及源码/制品/独立导出/反编译四条路径。
尚未替换 Qwen 源码或任何训练入口；当前仍有两套前端。

与[原子能力路线](primitive-roadmap.zh-CN.md)对齐：完整精度策略、旧算子统一 IR、
buffer 身份、部分打包和迁移扩展仍未完成。新增 f32 原语不改变这些边界。

随后相关回归 69 项通过；新增 Qwen 原源码 L2Norm 对照及空轴/大值专项后，
数学专项共 23 项通过。Qwen helper 与原生子图的 f32 前向逐位一致，梯度相符；
这只证明该 helper 的迁移可行，不代表整个 Qwen 模型已迁移。
