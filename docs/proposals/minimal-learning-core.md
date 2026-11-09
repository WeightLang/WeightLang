# 设计提案：用小核心支持脉冲网络、局部学习与遗传算法

**状态：设计建议，尚未实现。** 本文盘点基于主干 `8011fb8afc1bcf9913f22b11f995705e6936e5c3`，已包含 Rust 全指令覆盖的 PR #11。下文标为“草案”的语法和 SDK 接口不能当作当前可运行 API。本轮没有改变语言版本、parser、执行器或训练行为。

## 提纲与结论

1. 盘点当前语言的用户概念、模型计划和底层指令。
2. 分开模型结构、学习规则和宿主调度，明确 FF 与无梯度的差别。
3. 优先统一模型状态与组合机制，补少量通用张量能力。
4. 将神经元、局部可塑性和 FF 公式放入模型标准库，将 GA 群体调度放入官方 Rust/Python 库。
5. 给出小型语法示意、迁移顺序和可验收的“原生支持”标准。

建议把模型统一为“参数、状态和一组函数”。纯预测、推进神经状态、局部参数更新都通过普通函数与显式写入表达；SNN、FF、GA 不需要各自增加关键字。当前最影响这一目标的缺口是 Forward 与 Operations 的互斥设计、缺少一般逐元素硬比较、标准库复用状态更新公式不够方便，以及宿主没有完整的隔离候选评估接口。[当前模型计划](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/model.rs)、[具名操作](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/operations.rs)

## 一、当前语言设计盘点

### 1. 三种模型计划

| 当前计划 | 能表达什么 | 当前限制 |
| --- | --- | --- |
| `Graph` | 顺序 `layer`、初始化 seed、简单层组合 | 与普通 forward/参数/helper/loss 声明分开 |
| `Forward` | 根 input/output/target、参数、forward/loss、分支循环、子模型 | 普通服务端口是单个定长 f32 向量；没有 typed stateful 子模型体系 |
| `Operations` | 具名且带类型的函数、零个或多个输入、持久 state、参数/状态更新、void 返回 | 不与根 input/output/target 混用；不支持 loss、子模型、入口互调或 AD |

Graph 是顺序模型便利入口，Forward 偏向计算图与可微程序，Operations 已经能够执行没有自动微分的模型内部更新。这些功能分别有价值，但互斥规则妨碍“带状态单元组成多层模型，再提供局部学习入口”。不能用新增一个 SNN 名称解决这种组合问题。具体限制以[模型前端](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/model.rs)和[操作前端](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/operations.rs)为准。

### 2. 用户实际使用的概念

| 当前设计 | 已有含义 | 建议 |
| --- | --- | --- |
| `wgt 0.5`、`model` | 版本与模型定义 | 保留；新行为采用显式版本迁移 |
| `parameter`、`frozen` | 模型参数及写权限 | 保留所有权区别，避免为优化方法重复声明参数 |
| `learn` | float、范围 float、正数、bool、范围 int 等可调域 | 将合法域与梯度估计方法分开 |
| `prob`、`logits`、`probabilities` | 标量概率参数与分类输出合同 | 保持明确用途，不与 spike 或 bool 混用 |
| `state` | Operations 的跨调用状态 | 扩展到统一模型与子模型 |
| `let`、局部赋值 | 调用中的中间值 | 保留；与持久状态区别明确 |
| `fn`、纯 helper | 普通计算函数与模型入口 | 统一函数能力，推导读写效果 |
| `input/output/target` | 旧普通 forward 接口 | 后续作为函数签名的兼容便利写法 |
| `if/else`、`while`、`break/continue` | 有预算的内部控制流 | 明确硬控制流；不隐含代理梯度 |
| `gate/route/mix/blend` | 软门控、路由及组合写法 | 减少公开同义入口，同时保留惰性与梯度合同 |
| `repeat ... carrying ...` | 有静态上界的重复组合 | 保留一种通用组合表达，不额外增加 SNN 时间循环关键字 |
| 子模型实例与实例数组 | 同实例共享参数，不同实例独立参数 | 推广到拥有独立 state 的可组合单元 |
| `read/write` 与 `<-` | 当前两套显式参数/状态更新入口 | 新设计统一成 `<-` 的事务语义，旧写法继续适配 |
| `memory_alloc/read/write/free` | 一次调用内的临时 bank | 属于低层工作区能力，不等同持久神经状态 |
| `import`、`operators`、`std/` | 纯计算模块与闭合导入 | 扩充通用函数表达能力，再下放常见算法 |

当前计算以固定 shape 的 f32 为主。内部支持多轴张量，普通 forward 端口仍为向量；typed 参数、状态与端口也有自己的向量合同。不存在通用动态 shape、任意 dtype 或任意 NumPy 式广播。部分字面量缩放可以 lowering，但这不代表任意运行时标量都能广播。`learn bool/int` 的梯度路径使用连续影子值与 STE，它们不等于实际布尔或整数函数的真实导数。[张量 lowering](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/adaptive/source.rs)、[learn 实现](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/adaptive.rs)

### 3. 44 项 IR 的完整目录

“44 项”是编译后的指令种类，包含控制流、读写和工作区管理，不能直接当成用户需要学习的 44 个数学原语。`square`、`**2`、`.mean`、`@`、`.T` 等便利写法也不能各算一项独立语义。另一个 `primitive_specs.json` 属于制品 Graph，不能把其中存在的算子当成 WGT 模型源码已经支持的函数。[权威目录](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/language/tensor_contracts.json)

| 类别 | 当前 IR |
| --- | --- |
| 线性计算与卷积 | Linear、Conv、Matmul、BatchMatmul |
| 形状与索引 | Reshape、Transpose、Permute、Slice |
| 逐元素与常量 | Add、Mul、Scale、Minimum、Clip、Neg、Exp、Log、Tanh、Sigmoid、Relu、Detach、Constant |
| 归约、规范化与池化 | Sum、Mean、Softmax、LogSoftmax、SoftmaxAxis、LayerNorm、MaxPool2d、AvgPool2d、ValidateProbabilities |
| 路由与控制流 | Gate、Route、InferenceSkip、Label、Branch、Jump、Return |
| 参数 | Read、Write |
| 工作内存 | MemoryAlloc、MemoryFree、MemoryRead、MemoryWrite、ResourceState |

当前模型标准库包含 `std/activation`、`std/loss`、`std/attention`、`std/numerics`。它们是纯计算函数，不持有模型权重、可变状态或训练循环。当前没有可直接调用的 SNN、FF 或 GA 标准库。[标准库实现与签名](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/docs/standard-library.md)

### 4. 本轮做过的源码检查

使用当前 `wgt compile - --target rust` 检查了三个小定义：带 `state`、标量阈值条件、`step/reset` 的单神经元模型被接受；调用 `ge(current,[1,1])` 的向量阈值模型被明确拒绝；纯 helper 返回 `(x,x)` 的元组被明确拒绝。这些检查证明相关语法边界，不是 SNN 精度、时间行为或训练效果的验收。

`language/syntax-tour/model.wgt` 仍有“compiled 不支持工作内存”“没有实例数组”等过时注释。本提案以实现及行为测试为准；这些注释不构成真实能力限制。历史普通宿主脚本中的训练、I/O、async、agent/environment 也不属于本提案要继续扩大的模型语言。

## 二、把模型与学习方法分开

模型可抽象为 `(输出, 下一状态, 下一参数) = F(输入, 当前状态, 当前参数)`。普通预测可以保持状态和参数；SNN 推进膜电位；STDP 等规则显式调整参数。这个表示描述计算与变化的归属，训练循环仍由 Python/Rust 宿主组织。

| 名称 | 指的是什么 | 对反向传播的要求 |
| --- | --- | --- |
| SNN | 用脉冲和随时间演化的神经状态表示计算 | 训练方法另选，SNN 本身不保证没有反传 |
| Forward-Forward | 每层用正负样本学习局部 goodness | 不需要跨层误差反传；局部更新可以仍是梯度 |
| 直接 Hebbian/Oja/STDP | 用局部活动、脉冲和迹计算参数更新 | 可以直接写规则，不调用 AD |
| 纯选择、变异式 GA | 用群体适应度选择并产生新候选 | 可以不访问导数，也不构造梯度估计 |

本文按 FF = Hinton 的 Forward-Forward 理解。其局部目标可取活动平方和，区分正负数据，向下一层传递前再按长度归一化。下面据该目标推导的解析更新不需要自动微分，但数学上仍是局部梯度。编译器能保证“此路径不构建反向图、不调用 AD”，不能从任意乘加程序中证明“数学上完全没有梯度”。[FF 原论文 §2](https://arxiv.org/html/2212.13345v1)

脉冲模型可以通过代理梯度训练，因此“使用硬脉冲”与“使用无反传算法”必须分别声明。严格无 AD 的实现不得偷偷换成 surrogate gradient 或 STE。[SNN 代理梯度论文](https://arxiv.org/abs/1901.09948)

GA 与 ES 也不应混称：有些 ES 用适应度扰动估计梯度；本提案首期的 GA 使用选择、变异及可选交叉，独立记录算法类型。[Deep GA 原论文](https://arxiv.org/abs/1712.06567)

## 三、建议的小核心

### 1. 统一一个模型表示

保留五类语义：有形状的值；参数和状态存储；函数与模块组合；有界控制流；显式事务写入。Graph/Forward/Operations 后续都转换成同一模型表示，区别由入口签名和效果描述。现有三种入口可以保留兼容适配，不需要立刻删除。

建议普通 `fn` 通过推导得到 `reads_state`、`writes_state`、`writes_parameters` 等效果。`forward/observe`、`step`、`adapt`、`reset` 是推荐命名，不各自成为新的语法关键字。纯函数不能捕获可变状态；有状态操作显式属于模型实例。是否建立 AD 图由调用策略与函数能力决定，不由“用了 typed 签名”一刀切。

新版本推荐 `forward` 为只读入口，时间演化通过 `step`，学习更新通过 `adapt`。现有 WGT 0.5 `forward` 可以提交写入，不能静默改义：旧版保留原行为，新版迁移时给出诊断并建议改名。现有 typed `forward` 已要求只读，应把这一点纳入统一接口。

### 2. 参数域不绑定学习算法

`parameter/frozen` 的核心区别是所有权和写权限，`learn` 的核心价值是合法域。建议统一元数据：名称与共享身份、shape、初值、合法域、是否允许更新；SGD、FF、GA 共同消费它。初期可继续接受 `learn` 拼写，后续再统一声明形式，不必先做纯外观改名。

GA 对 bool 做翻转，对 int 做合法整数变异，对范围 float 做明确的修复或采样。提交层严格拒绝非法候选，不替算法偷偷截断。梯度方法需要 STE 时必须显式选用；不能把 STE 影子值直接当作离散基因。

### 3. 优先补通用能力

| 能力 | 最小合同 | 支持的实际需求 |
| --- | --- | --- |
| 逐元素硬比较 | 首期 `ge(a,b)` 返回同形的 f32 0/1 mask；相等为 1，有限输入；AD 行为必须显式定义 | 脉冲、二值网络、阈值门、选择掩码 |
| 内部元组返回与解构 | 先用于纯 helper 的多结果，不立即扩张服务外部 ABI | 一次返回 `(下一膜电位, 脉冲)` 或 `(新权重, 新偏置)` |
| 明确的标量规则 | 真正的标量可提升到固定张量；其他张量要求匹配 shape；区分标量与一元素向量 | 运行时学习率、阈值、衰减和归一化因子 |
| 数值与归约基础 | 明确 div、sqrt/rsqrt、按轴归约的有限值和 shape 合同，按实际库需求逐项补 | FF 长度归一化、归约、局部更新 |
| 有状态子模型组合 | 参数与 state 的共享/独立规则一致，函数效果跨子模型传播 | 多层 SNN、带局部迹的可复用学习单元 |

`ge` 的首期结果使用已有 f32 存储，不要求立即引入一种脉冲 dtype。需要严格 bool 张量时可另行扩类型系统。首期只在无 AD 路径支持硬比较，要求梯度的路径应明确拒绝或要求显式处理不可微边界，绝不默认代理导数。

`select(mask,a,b)` 可作为下一项通用能力，但必须定义它是已计算张量之间的选择，还是能避免某个表达式求值。元素选择不自动等于控制流短路；不能用软混合 `mask*a+(1-mask)*b` 掩盖非有限分支。初版 LIF 的减阈值复位仅需硬比较和乘加，可先完成这一更小集合。

原语数量不应通过删除稳定数值内核来压缩。softmax、layer_norm、conv 可以保留优化 intrinsic，公开 API 归入标准模块；普通函数展开与后端融合共用同一语义合同。稳定 softmax 不能简单替换成可能溢出的 `exp(x)/sum(exp(x))`，现有 gate 的惰性推理和训练规则也不能在移入库时丢失。

### 4. 标准库分两层

| 层次 | 拟议内容 | 状态与执行归属 |
| --- | --- | --- |
| WGT `std/snn` | IF/LIF、复位、突触滤波与迹更新公式 | 参数和状态由模型持有，函数显式传入并返回 |
| WGT `std/plasticity` | Hebbian、Oja、简化 pair-STDP 等局部更新 | 返回候选参数/迹，模型显式 `<-` 提交 |
| WGT `std/ff` | goodness、长度归一化、特定层的解析局部更新 | 首期明确支持 Dense/ReLU，不泛称任意算子可自动学习 |
| WGT `std/nn` 等 | 激活、损失、归一化、常见块 | 复用共享核心和优化内核 |
| 官方 Rust/Python 库 | FF 层训练调度、GA 群体与评估、数据、设备任务 | 宿主拥有循环和调度；Rust 路径无需 Python |

STDP 库应写清前后脉冲迹的更新顺序和具体规则；它是选定的计算模型，不声称完整复现生物突触。原始实验支持脉冲相对时序与更新方向的联系。[Bi 与 Poo 原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6793365/)

“原生 GA”意味着官方接口、参数域校验、隔离评估和 Rust 实现能够共同工作。它不要求把种群管理放进模型 forward，也不要求恢复旧的宿主脚本语言。
