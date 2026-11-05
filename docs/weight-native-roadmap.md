# Weight-Native 架构路线图

> 状态：架构提案与工程排序，不是当前能力承诺。本文以现有 WGT 0.5、Rust core、wgtime service 和原生制品为基础，采用增量演进，不推倒重写。

## 目标

WeightLang 的目标是 **Weight-Native Programming Language**：权重具有一等语义，Tensor 是数值计算基础，可微模型表示保留数学语义，执行计划由多个后端负责。WGT 描述模型内部结构与计算；Rust 应用负责数据、I/O、训练/搜索循环和服务编排；唯一 wgtime service 持有可变模型实例。

这不是重新实现一个 PyTorch，也不是把现有语法换皮。目标是定义 WeightLang 自己稳定的参数语义、张量语义和模型语义，同时通过明确的导入/导出边界兼容 PyTorch 等生态。

## 设计不变量

1. **Weight != Tensor。** Weight 是有稳定身份、绑定关系、训练策略和持久化语义的参数对象；Tensor 是计算过程中的带类型数值对象。Weight 可以在计算中读取为 Tensor，但不能因此丢失参数身份。
2. **共享是身份关系，不是视图。** 两个路径引用同一 Weight 时，梯度累积、优化器状态和更新提交都必须归属于同一个参数身份。Tensor view 只描述同一数值存储的索引/布局关系。
3. **定义、实例、权重、状态和 checkpoint 分离。** 解析定义不能初始化实例；权重文件不隐式携带运行状态或优化器状态；checkpoint 的内容必须明确。
4. **数学语义与执行计划分离。** Semantic Graph 保留操作、shape、dtype、梯度规则和来源；Execution Graph 可以融合、重排、分配内存和选择 kernel，但必须保持可观察语义。
5. **后端不拥有语言语义。** CPU、WGSL/WebGPU、CUDA、ROCm、Metal 等是执行目标，不得把某一后端专属概念泄漏为核心语言的必要条件。
6. **优化策略可替换。** Optimizer、搜索算法或用户自定义更新器通过显式参数更新协议工作，不应被固化成语言内核唯一更新方式。
7. **能力声明必须有证据。** Parser 接受、IR 构建、数值正确、梯度正确、真实 service 执行、GPU 执行、性能结果是不同验收等级。

## 当前基础与边界

以下是现有仓库中已经存在的构件，不代表所有路径都已完整验收或发布：

| 领域 | 已有基础 | 不能据此推断 |
| --- | --- | --- |
| Tensor 几何 | 固定 shape 张量类型、局部类型注解、命名轴、rank-2 @、.T、permute、rank-3 batch_matmul；见 tensor-geometry.md | 任意 rank、动态 shape、通用 broadcasting、任意 dtype 或通用 layout ABI |
| Weight 身份 | 参数身份、显式共享/别名校验及权重类型约束相关实现 | 完整独立 Weight 对象生命周期、跨版本 ABI 稳定性或所有训练路径的共享状态正确性 |
| 制品与恢复 | model.json、checkpoint.json、manifest.json 分离；见 artifact.md | 跨版本、跨设备、跨后端的通用权重互操作格式 |
| 自动微分 | 部分数值计算路径具有自动微分和梯度校验 | 所有 typed operations、任意自定义算子、JVP/VJP 或跨状态求导 |
| 神经网络算子 | 固定范围卷积、attention、激活和 loss 等实现 | 完整 CNN/Transformer 算子覆盖、FlashAttention 或 GPU 原生 kernel |
| 执行后端 | Rust/native CPU 路径与 WGSL 相关实现 | CUDA/ROCm/Metal 完整后端或统一异步设备内存合同 |

## 工程阶段与依赖

### P0-A：Tensor Contract

目标是将当前固定 shape 能力整理成统一、可验证的 Tensor 类型合同。

- 统一 Shape 表达：静态维度、符号维度和约束维度分别建模；先实现静态维度，动态维度单独立项。
- DType 与 Layout 分离：dtype 表示元素数值格式；layout 表示 strides、offset、设备与存储连续性。
- Named Axes 是 shape 维度的语义标签，不应成为运行时数据。
- Shape inference 负责从算子签名推导输出；显式注解用于约束和错误定位。
- Broadcasting 必须定义轴对齐优先级：命名轴按名称对齐，未命名轴按明确的尾轴规则对齐；禁止含糊混合。
- View 算子只在 strides/offset 可表达时零拷贝；否则明确 materialize，不能承诺所有 reshape 都零拷贝。
- 编译期拒绝 rank、收缩维、轴名、dtype 和 layout 不兼容操作。

验收：每个算子有输入/输出类型规则；正反例测试覆盖轴名、广播、stride、零维/单维边界；错误包含源位置；随机 shape 属性测试与数值参考一致。

### P0-B：Weight Contract 与 Weight ABI

目标是让参数身份和存储格式成为稳定、可迁移的项目接口。

- Weight identity 是稳定逻辑 ID，不依赖模块路径；模块路径只是可读 binding name。
- Weight binding 允许架构定义与权重集分离；加载时做 shape、dtype、约束和语义版本校验。
- Sharing 使用 canonical owner ID；别名不重复分配参数或 optimizer state。
- Trainability（trainable/frozen/inference-only）与运行时是否执行训练分开定义。
- Initializer 记录算法及参数；初始化是实例创建时显式行为，不是解析副作用。
- Constraints 在绑定、更新候选和加载时统一校验；投影/拒绝策略必须明确。
- Versioning 记录 schema version、model revision、weight revision 与 lineage；不可变快照优先。
- Quantization 元数据描述量化方案、scale/zero-point、group/block 维度和校准来源。
- 权重数据与 manifest 分离；大权重使用二进制 shard，不将数值数组塞进 JSON。
- 具体格式草案见 weight-abi.md。

验收：同一权重可绑定到两个兼容架构；共享参数 roundtrip 后仍只有一个 canonical storage；损坏 hash、未知版本、错误 shape/dtype、非法约束和别名环均 fail-closed；保存失败不留下可误认的完整制品。

### P0-C：Typed Semantic IR 与 AD Contract

目标是为前端、优化器和后端建立稳定边界。

- Typed Model IR 节点携带 shape、dtype、layout、设备约束、来源位置和副作用分类。
- Semantic Graph 保留模型可观察语义；Execution Graph 表达具体 kernel、融合、临时 buffer 和依赖。
- 每个可微算子声明 VJP；需要高阶/敏感度能力时声明 JVP。未声明梯度规则的算子明确不可微。
- Custom VJP/JVP 必须通过形状、dtype、有限值和梯度一致性检查。
- 梯度累积按 Weight identity 汇总，不按读取路径重复创建 optimizer state。
- Checkpointing 是执行计划优化，不改变数学图；重计算必须遵循随机数和副作用语义。
- Graph serialization 必须版本化，不能把后端私有 kernel ID 当作唯一语义表示。

验收：核心算子中心有限差分；共享/冻结/分支/广播梯度；VJP 与 JVP 对照；图 roundtrip；无效依赖、环和副作用顺序测试。

### P1：通用算子与模块组合

- 先补齐通用 primitive：reduce、concat/split、gather/scatter、einsum、pooling、normalization、embedding。
- Conv、Attention 等作为标准库组合和可优化 pattern，不创建模型类别专属张量系统。
- Module 定义与 Module instance 分离；参数化模块的实例化规则必须显式。
- Module import、version pin、parameter binding 和共享参数语义统一。
- 标准库模块公布完整 shape/dtype/gradient/state 合同及最低运行时版本。

验收：CNN 与 Transformer 都使用同一 Tensor、Weight 和 IR；同一模块多次实例化与共享实例的参数身份有独立测试。

### P1：Execution Optimizer 与后端

- 先做常量折叠、死代码消除、代数简化，再做融合和内存规划。
- Buffer reuse 需要基于 liveness 和 alias analysis；不能仅凭图节点顺序复用。
- 算子选择使用 shape、dtype、layout、设备和工作量成本模型。
- 后端统一异步提交、事件/同步、设备 buffer 生命周期、错误传播和 host/device transfer。
- Native CPU 作为参考后端；WGSL/WebGPU 作为可移植 GPU 路径；CUDA/ROCm/Metal/Vulkan 按独立能力矩阵接入。
- 自定义 kernel 必须声明输入输出、布局、设备要求和梯度实现或不可微边界。

性能验收：同时报告 compute、memory、transfer、launch、compile/warmup；与正确性参考对比；区分端到端、kernel-only 和冷启动延迟。不得只报告 FLOPS。

### P2：训练与高级模型能力

训练循环由 Rust 应用编排，参数更新通过可插拔策略接口实现。官方提供 SGD、Adam、AdamW 等参考策略，但不限制用户定义自有更新算法。训练系统逐步加入 scheduler、loss、batch/data pipeline、distributed execution、gradient clipping、AMP、checkpoint/resume、metrics 和 reproducibility。

MoE、SSM/Mamba、KV cache、稀疏张量、动态 shape、低比特推理和 speculative decoding 建立在前述合同上，不先引入专用语法。

## 统一更新策略协议（设计草案）

更新器抽象为：读取参数快照、可选梯度、更新器私有状态，产生参数候选和新状态。运行时负责校验和原子提交。协议必须允许没有梯度的更新策略，也允许按 Weight identity 更新子集。

~~~text
UpdateContext {
    parameter_snapshot
    gradients?             # optional; search/heuristic updates may omit
    step
    metadata
}

UpdateProposal {
    parameter_updates      # keyed by stable Weight identity
    optimizer_state_update # private to the strategy
}

validate(proposal) -> Result
commit(proposal) -> Result
~~~

这是接口设计草案，不是当前 WGT 语法或已实现 API。提交失败不得部分改变参数或更新器状态。

## 关键决策与反模式

- 不把所有权重都降级成普通 Tensor；也不把每个 Tensor 都变成 Weight。
- 不让 optimizer 成为唯一更新途径。
- 不把语义图和后端执行图合并成一套后端专属 IR。
- 不以 parser 通过代替运行时和梯度验收。
- 不先实现几十种模型专用语法；先把可组合原语和模块边界做好。
- 不宣称零拷贝、自动 GPU 或动态 shape，除非相关路径有可复现测试。
- 不破坏现有 WGT 0.5 源码和制品；新合同采用版本化迁移与兼容读取策略。

## 近期可交付里程碑

| 里程碑 | 交付物 | 完成条件 |
| --- | --- | --- |
| M1：Tensor contract inventory | 算子类型规则、当前限制矩阵、错误诊断测试 | 所有现有 tensor primitive 有可追踪合同 |
| M2：Weight ABI draft | schema、manifest、payload、alias 与校验规则 | 有规范和 roundtrip/拒绝用例 |
| M3：Typed IR boundary | Semantic/Execution graph 责任边界和版本策略 | 至少一个 CPU 路径端到端走新边界 |
| M4：AD contract | VJP/JVP 注册和自定义梯度合同 | 核心算子梯度对照通过 |
| M5：Backend contract | async、device buffer、transfer、custom kernel 合同 | CPU 与一个 GPU 后端共享同一语义图 |

以上里程碑是建议顺序。只有实现、测试和文档三者一致后，才将设计状态升级为已实现或已发布。
