# 先完善语言，再扩大模型迁移

本轮调整：暂停以“迁移模型数量”为目标的扩张。先扫描共有计算需求，完善原子能力、
组合方式和契约，再用少量结构差异大的模型验收；语言稳定后才扩大批量转换。
这份文件是扫描结论和候选设计，**不是新语法承诺或已实现功能清单**。

## 扫描依据与边界

参考本机 Transformers 5.7.0 的 **442 个 modeling_*.py 文件**，AST 解析错误 0。
[机器可读报告](transformers-5.7.0-primitives.json) 保留每个文件 SHA256、分类信号和最多五处行号证据。
[扫描脚本](../../scripts/scan_transformers_primitives.py) 不导入或执行模型，不下载或读取权重。

```sh
python scripts/scan_transformers_primitives.py -o /tmp/weightlang/primitives-new.json
# 也可扫描其他源码目录，版本会标为 custom-source：
python scripts/scan_transformers_primitives.py --root /path/to/transformers/models -o /tmp/weightlang/primitives-custom.json
```

输出必须是新文件。统计是“出现对应调用名称/forward 参数的文件数”，每类每文件只计一次。
分类互相重叠，不能相加；生成模型、变体重复存在，不能视为独立架构数。
包括文件中的辅助函数、损失和训练代码，不是仅统计主 forward；没有追踪 imports、别名、
自定义融合算子或共享库。方法同名可能误报，内核封装可能漏报；需要回看证据决定语义。
例如 attention_kernel 只有 17 个文件命中直接调用名，绝不代表只有 17 个模型使用注意力。
所有文件都有 Python 下标操作，但可能操作列表/配置，不能把 442 全算作张量 slice 需求。

## 高复用需求与当前边界

当前边界来自本地实现快照，不从扫描命中推断语言已经支持；所有实现均指本地代码。

| 需求信号 | 文件数 | 当前能力 / 需要补的契约 |
| --- | ---: | --- |
| reshape/view/transpose 等轴变换 | 426 | 已有 rearrange/flatten 和命名轴；动态长度、跨端口维度约束仍需统一 |
| 张量创建 | 400 | 参数由现有层创建，通用 arange/常量/zeros_like 的静态图表达尚不完整；不能把运行时值写入结构文本 |
| split/chunk/cat/stack | 396 | concat 仅末轴；通用 split、多输出绑定、任意轴 stack/concat 是缺口 |
| 精度转换 | 379 | 静态端口已有 f32/i64/bool；底层 .to/float 与动态后端能做更多，不等于静态图已有 dtype/累积精度契约 |
| 矩阵收缩（含 @） | 377 | linear、注意力内部有矩阵乘法；两个运行时张量之间的通用 matmul/einsum 尚非静态图原子算子 |
| 显式参数/缓冲区 | 370 | 层拥有参数、embedding/head 有 tie_embedding；通用命名参数、buffer、别名和共享调用仍需一致表示 |
| exp/log/rsqrt 等数值运算 | 369 | 多数封装在 RMSNorm/注意力内部；缺少可组合数值原子和明确数值域/精度规则 |
| 归约 | 367 | reduce 支持 mean/sum，静态图不接受标量输出；keepdim、max/var/logsumexp、索引输出需明确 |
| 外部 mask 参数 | 360 | run_sequence 接受特定 token mask；不等于任意布尔/加性 mask、cross-attention 多输入已有统一图契约 |
| expand/repeat 等广播 | 358 | add/mul/where 要求同形；建议增加显式 broadcast，而不偷偷改变已有同形运算语义 |
| 归一化调用 | 312 | LayerNorm/RMSNorm 已有；不同归约轴、GroupNorm/BatchNorm 的状态与训练行为不能简单套用 |
| 位置参数 | 299 | 内部位置生成及部分 RoPE 已有；显式 position_ids、相对位置偏置等仍需端口和组合能力 |
| 跨模态/encoder 输入参数 | 292 | 是参数名称信号，不能等同 292 个跨注意力模型；需要明确多输入和不同序列轴 |
| 缓存/状态参数 | 245 | run_sequence 已支持专用 K/V 状态，scan 已有单向量隐状态；通用显式状态结构尚需统一 |
| 卷积 | 225 | 有受限 Conv2d；Conv1d、分组/深度卷积、dilation、因果边界等需补全 |
| where/masked_fill/tril 等 | 212 | 有同形 f32 条件 where；bool 比较、可广播 mask、全屏蔽行行为尚需明确 |
| 重采样/池化/pad | 210 | 有基础池化；插值方式、边界、align_corners、任意维 pad 不应无差别映射 |
| topk/sort 等选择 | 152 | sparse_moe 内部实现了一种 top-k；值与索引双输出及排序/并列行为尚需通用契约 |
| gather/scatter/index 等 | 143 | 内核内部使用，不是通用静态图路由；重复索引的累积和反向行为需要独立验收 |
| cumsum/cumprod 等前缀计算 | 124 | 不等同现有单状态 scan；需要轴、累积精度及状态表达 |
| 复数/FFT 信号 | 4 | 优先级较低；先保留需求证据，不急于扩大 dtype 和后端范围 |

已阅读的实现入口：
[静态语法与契约](../../src/weightlang/syntax.py)、
[运行时算子](../../src/weightlang/modern.py)、
[专用序列调用](../../src/weightlang/sequence.py)、
[数据流](../../src/weightlang/dataflow.py)、
[动态 Torch 前端](../../src/weightlang/tensor_program.py)。

特别注意两条路径：静态图前端已加入固定形状具名多 input/output，详情见下方增量记录；torch-dynamic 前端可以表达
多参数、元组返回和更多 Torch 操作，但其源码通过宿主绑定执行，需要信任授权，
没有自动获得静态图的形状检查和跨后端算子契约。不能把这条宿主执行路径算作原子能力已完备，
也不需要为已经在宿主层自然完成的数据准备、损失、优化器、采样再扩展 WGT。

## 先补哪些能力

### 第一阶段：统一组合与契约，再加算子

1. **端口和形状**：明确具名多输入/多输出、bool/index/浮点 dtype、标量与符号序列维度、
   可选状态的表示。先做函数边界和约束，避免不同模型各自夹带 mask/cache 参数。
2. **参数与别名**：将参数身份、共享引用、buffer 和初始化声明明确区分；权重值继续留在
   二进制模型制品。保留 repeat 独立参数与共享调用的区别；通用共享不应只靠 embedding 序号。
3. **统一算子契约登记**：梳理解析签名、shape/dtype、参数布局、Python 导出和 Rust 校验的
   对应关系，建立一致性检查。是否生成代码再决定，不先大改全部实现。

验收：错误的维度/类型/状态长度应在清楚的边界报错；同一权重的别名在保存重载和导出后
仍是同一参数且梯度累加正确。新增模型不得靠复制一份参数模拟共享。

### 第二阶段：让张量计算能组合

优先候选：`matmul`、显式广播、slice/split/concat/stack、比较与 bool mask、
带 keepdim 的归约、arange/常量和 cast；随后补 gather/scatter、top-k。
这些名称是候选语义，尚未定义新 WGT 语法。

attention 的 Q/K/V 投影、位置处理、打分、mask、softmax、聚合应能分别描述。
保留现有 fused attention 作为性能实现；不要求把成熟优化内核拆慢，
但需要一份可读的组合参考与前向/反向对照，避免每个模型都增加一个专用大算子。
只在 matmul 的收缩/广播规则明确后，再决定是否需要更宽泛的 einsum。

验收包括非连续张量、不同 batch、Q/K 不同长度、全部被 mask 的行、越界和重复索引、
广播梯度归约、空集合/零长度是否支持；不支持的边界明确拒绝。

### 第三阶段：有证据地扩展空间与状态计算

- Conv1d/groups/dilation/causal padding；用音频和线性注意力共同检验。
- 通用状态递推：多个状态张量、输出与状态分离、跨 chunk 传入传出；
  对比全序列与分块执行及梯度，不将 cache 当模型参数保存。
- 非基础 pooling/interpolation、位置偏置、MoE 路由策略等，由代表模型提出具体契约。
- FFT、复杂多模态专用预处理先延后；训练调度继续归 Python。

## 用少量代表模型检验，不扩大搬运量

| 代表对象 | 检验目的 | 需要保留的证据 |
| --- | --- | --- |
| 已有 Llama/Qwen3/OLMo2/3 | 现有行为稳定、GQA、归一化轴、窗口、参数路径 | 现有数值/梯度回归，旧结构的重载兼容 |
| T5 的最小 encoder-decoder | 多输入、cross-attention、Q/K 不同长度、相对位置偏置 | mask/位置边界以及共享 embedding；源码 t5/modeling_t5.py 的 relative_attention_bias |
| ViT 小模型 | patch 卷积、可学习 token/位置参数、广播/拼接、非 causal 注意力 | 分辨率和位置插值的显式适用范围 |
| Whisper 小组件 | Conv1d 与不同长度的跨注意力 | 因果/非因果边界及音频特征形状；不用复制音频加载管线 |
| Mixtral 的小 MoE block | top-k 双输出、gather/scatter、重复路由、aux 输出 | 路由及全部专家梯度；不能仅以现有 sparse_moe 名称判等价 |
| Qwen3.5 的线性注意力 block | 深度因果 Conv1d、门控 delta 递推、多状态和分块 | 对照 torch fallback 的 chunk/recurrent 公式，不把整模型移植当起点 |

T5 和 Qwen3.5 已抽查安装包源码中的相对位置偏置和 torch causal-conv/gated-delta fallback；
表中其他对象是下一阶段验证对象，**尚未完成迁移或数值验收**。
保留十个现有基础适配器作回归资产，不为每个变体复制新 block。

## 减少未来语言更新的批量改写

- 维护少量手写标准库组合、代表模型和原子行为夹具；模型清单只保存配置、来源哈希和缺口。
- 建议转换先落到规范化图表示，再统一生成文本；共享 block 的升级不改每份模型正文。
- 固定默认值、参数身份和序列化版本；区分“结构可重新生成”和“旧权重制品可正确重载”。
- 只有原子契约、保存重载、导出、错误诊断及数值梯度通过，才宣布该能力支持。
- 不以扫描命中数/语法通过率代替完成度；本轮不承诺何时全部迁完。

## 扫描工具初次核验（语言改动前）

扫描器通过不执行源码的夹具检查；核对全部分类的逐文件计数、证据数量和文档链接。
现有 tests/test_sequence.py 与 tests/test_tensor_program.py 共 12 项通过，
仅证明这些测试覆盖的序列调用/别名/制品/动态状态行为，不能外推为任意多端口静态图支持。
Ruff 与 git diff --check 通过。本轮新增扫描工具和报告，没有修改公开语言语义。


## 实施记录：组合契约第一增量

已落地（本地未发布）：静态具名多输入、平坦字典返回、bool 条件、scan 显式初始状态，
Python 保存/重载/独立导出/结构反编译和 Rust 制品校验；模型参数路径未因包装改变。
例子：[streaming_ports.wgt](../../examples/streaming_ports.wgt)；
行为检查：[test_ports.py](../../tests/test_ports.py)；契约详见 [dataflow](../dataflow.md)。

此增量仅覆盖固定正整数形状和一个 scan 向量状态，**第一阶段尚未完成**。
旧 run_sequence 的 RoPE cache/MoE aux 仍未接入静态端口；不把已有宿主动态执行算作完成。

后续按依赖顺序持续实施，验收通过一项再标记一项：

| 阶段 | 仍需交付 | 验收依据 |
| --- | --- | --- |
| 组合契约 | 符号 T/P、有限整数等式/整除/上界约束、零长度 cache、标量 aux、多状态/多结果 block | 错误长度/类型提前拒绝；全序列与分块前向、梯度相同；cache 不进入参数 |
| 算子规格与 IR | 统一端口、算子参数、输出、权重/别名、精度字段；先一致性检查再考虑生成器 | 解析/形状/执行/制品/导出/Rust 对同一夹具达成一致 |
| 参数身份与共享 | 具名参数引用、身份独立于路径、重命名/结构调整映射报告 | 插入 embedding 不改变别名；保存重载后仍同一参数；冲突不按顺序猜测 |
| 配置与打包 | 类型/默认值/约束的配置接口、结构化实例化、依赖解析与固定源码包 | 应用不用正则改 const 或私有 _prepare_program；重载不依赖可变外部源码 |
| 张量原子与精度 | matmul、显式 broadcast、slice/split/concat/stack、归约/mask、gather/scatter；存储/计算/累加/输出 dtype 及局部覆盖 | 非连续输入、广播梯度、全 mask 行、重复/越界索引；保存与导出数值一致 |
| 空间与状态 | Conv1d/groups/dilation、多个状态、跨 chunk 执行 | 小型音频卷积与线性注意力组件；全序列/分块数值与梯度对照 |
| 适配器与训练断点 | 声明式 LoRA/冻结/合并与解除、独立适配器制品；优化器/RNG/进度/参数身份 | 合并前后等价、解除可恢复；断点续训等价，架构不兼容输出报告 |

十个已有基础模型的回归继续保留；代表组件按 T5 跨注意力、ViT patch/拼接、Whisper Conv1d、
Mixtral 路由、Qwen3.5 线性注意力选择，不以新增模型数量考核完成度。

验证记录：Python 全套 887 passed / 1 skipped（包含十个模型的固定版本参考回归）；
随后 scan 时间轴修正的相关 141 项通过，Rust ports/dataflow/native/sequence 共 21 项通过。
Python 生成的 v4 示例制品经 Rust 实际读取并渲染四个端口；独立导出、
Python 更新保存和带权重引用的反编译均有行为测试。Ruff、示例 check --shapes、
文档本地链接和 git diff --check 通过。没有发布制品，也没有训练模型。


## 实施记录：有界维度与约束

第二个增量已落地（0.3 本地未发布）：`dim T min=1 max=16`、`require` 整数比较、
端口 `P+T` 表达式、输入维度绑定与执行前拒绝，以及同一 scan 模型的不同合法分块长度。
复用原静态检查器，对声明范围内全部合法组合验证；Python/Rust 分别执行验证，
不是只在一个示例长度下尝试。制品 v5 保存完整契约，Python 更新保存、独立导出、
反编译与图查看保留表达式；内部数字层形状明确标为首个合法赋值。

当前刻意限制为正整数维度，最多 4096 个原始组合、200000 个组合乘层节点，
不允许随长度改变参数布局，不支持固定轴计划的 rearrange/reduce/flatten（含相关语法糖）。
完整参数、预算和可运行用法见 [dimensions](../dimensions.md#bounded-symbolic-dimensions-03-localunreleased)，
示例 [variable_streaming.wgt](../../examples/variable_streaming.wgt)。

验证：Python 全套 925 passed / 1 skipped，包含十个参考模型回归；Rust 相关 23 项通过。
另用 Python 生成的真实 v5 制品验证 Rust 图查看及仅在代表长度正确的篡改拒绝。
测试覆盖不等长分块的前向/梯度、整数余数和上界、错误输入在执行层之前失败、
保存/重载/独立导出/反编译、无合法解和维度引起参数宽度变化的拒绝。

剩余优先工作：零长度初始 cache、标量 aux、KV/MoE 多结果图节点、通用 block/scan 多状态，
具名参数身份；再推进张量原子、配置打包、精度和适配器/训练断点。当前并未完成整个组合契约阶段。

最后复核补充：固定轴计划的 transpose/mean/sum 简写也明确拒绝，相关 67 项通过；
符号契约最终专项 19 项通过。文档、WGT 技能和编辑器词表同步，Ruff/链接/diff 检查通过。

## 2026-11-03 增量：组合与原子执行链

上方统计与能力表保留扫描时的历史快照；当前本地实现以此增量和
[组合契约](../composition.md)、[张量原子](../primitives.md) 为准。

| 用户提出的七项 | 当前落实 | 尚未完成 |
| --- | --- | --- |
| 多输入输出与状态 | bool/i64/f32 端口、标量/空轴、block 具名返回、KV/MoE 结果、scan_state 多状态与分块梯度 | 一般可选输入、嵌套记录、record recur 的进一步组合 |
| 符号维度及约束 | 有界整数 T/P、P=0、加法/整除/取模及比较、穷举合法环境和参数布局不变性 | 任意动态形状不在本阶段范围 |
| 统一 IR | 已解析 Layer/Port/results/returns 结构；原子算子签名/default 目录供 Python/Rust 共同消费 | 旧算子形状/权重/精度规则仍有重复，未完成集中 IR |
| 具名共享与参数身份 | parameter/share、真实 Parameter 别名、迁移报告和显式 mapping | buffer 身份和更广泛架构迁移 |
| 模型配置与打包 | int/float/bool 默认值/范围、整数 require、Python config、依赖展开和固定源码包 | 含权重导入/派生/自定义初始化/冻结的 package_model 路径仍明确拒绝，使用 build_bundle |
| 完整精度契约 | 旧精度开关保留原语义 | 存储/计算/累加/输出四部分契约尚未实现；制品仍 F32 |
| LoRA 与训练断点 | 声明式 linear LoRA、冻结、合并/解除、独立适配器、SGD/Adam/AdamW 身份断点与 RNG/进度 | 更广泛目标算子/优化器；跨版本确定性不作承诺 |

张量原子新增 matmul、显式 broadcast、slice、cat/stack、归约、比较、masked_softmax、
gather/scatter_add；空间计算新增 Conv1d/groups/dilation/causal。split 当前可组合 slice，
通用 split/top-k/arange/cast 尚未增加。禁止把源码文件命中数当作功能完成数量。

回归包含原有十种 Transformer、小型 cross-attention、空集合/标量、卷积、
显式 KV+MoE、两个不同状态的递推、参数共享、重命名后的优化器续训，以及四种执行路径
（源码、制品重载、独立 Python 导出、反编译重载）。这些是局部正确性证据，
不是已训练模型、性能结论或发布证明。七项路线尚未全部完成。

本次收口的本地验收：当前工作区 Python 全量 1061 passed、9 skipped（8 项 Lean 环境、
1 项显式网络测试）；十个 Transformer 的专项回归 38 passed；最近 Rust 全量 71 passed。
新增组合例子和本节所链接文档已检查；相关 Python 文件 Ruff 与 git diff --check 通过。
完整 Python 回归还修复了旧 0.2 block 导出缺少形状元数据的问题，只有需要模型 rank 的循环
导出才读取相应元数据。共享工作区还包含其他演进，本记录不把整个测试数量算成本阶段新增测试。

## 统一前端的数学原语增量

保留 Qwen3.5 等动态模型在 WGT 内，迁移目标和退出条件见
[统一前端迁移契约](unified-language.zh-CN.md)。当前先补齐
square/rsqrt/exp/log/sin/cos/softplus/shift/cumsum 以及无参数 L2Norm/RMS 标准库组合。
这些原语沿用声明式张量图，f32 输入/输出；并非完整精度策略或旧算子 IR 统一。
现有两套前端、模型源码、训练调用和保留快照尚未替换，高亮视觉调整延后。
