# 设计提案：用小核心支持脉冲网络、局部学习与遗传算法

**状态：设计基线，首期实现见 [PR #13](https://github.com/WeightLang/WeightLang/pull/13) 与[使用指南](../local-learning.md)。** 本文盘点保留提案提出时的主干 `8011fb8afc1bcf9913f22b11f995705e6936e5c3`，已包含 Rust 全指令覆盖的 PR #11。下文的“当前”指该历史基线，标为“草案”的接口不能替代使用指南中的实际 API。首期保留 WGT 0.5 与旧模型计划的兼容入口；完整计划合并、动态 shape 和拓扑进化不在本次实现范围。

## 提纲与结论

1. 盘点当前语言的用户概念、模型计划和底层指令。
2. 分开模型结构、学习规则和宿主调度，明确 FF 与无梯度的差别。
3. 优先统一模型状态与组合机制，补少量通用张量能力。
4. 将神经元、局部可塑性和 FF 公式放入模型标准库，将 GA 群体调度放入官方 Rust/Python 库。
5. 给出小型语法示意、迁移顺序和可验收的“原生支持”标准。

建议把模型统一为“参数、状态和一组函数”。纯预测、推进神经状态、局部参数更新都通过普通函数与显式写入表达；SNN、FF、GA 不需要各自增加关键字。当前最影响这一目标的缺口是 Forward 与 Operations 的互斥设计、缺少一般逐元素硬比较、标准库复用状态更新公式不够方便，以及公开服务接口尚无统一、无 AD、包含状态隔离合同的候选评估协议。[当前模型计划](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/model.rs)、[具名操作](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/operations.rs)

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

当前原生模型计算使用固定 shape 的 f32 张量。内部支持多轴张量，普通 forward 端口仍为向量；typed operations 的 state 和入口端口为固定宽度向量，普通 parameter 可以声明多轴 shape。不存在通用动态 shape、任意 dtype 或任意 NumPy 式广播。部分字面量缩放可以 lowering，但这不代表任意运行时标量都能广播。`learn bool/int` 的梯度路径使用连续影子值与 STE，它们不等于实际布尔或整数函数的真实导数。[张量 lowering](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/adaptive/source.rs)、[learn 实现](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-language/src/adaptive.rs)

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

本文按 FF = Hinton 的 Forward-Forward 理解。其局部目标可取活动平方和，区分正负数据，向下一层传递前再按长度归一化。下面据该目标推导的解析更新不需要自动微分，但数学上仍是局部梯度。拟议的执行策略需要保证“此路径不构建反向图、不调用 AD”；但不能从任意乘加程序中证明“数学上完全没有梯度”。[FF 原论文 §2](https://arxiv.org/html/2212.13345v1)

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

## 四、三个具体例子

### 1. SNN：先让状态和阈值足够自然

下面是**语法草案，当前不能直接编译**。主体采用已有 Operations 形式，新增的计算能力只有 `ge`，版本号省略，避免把提案冒充 WGT 0.5 功能。

```wgt
model SpikingCell

state v: f32[2] = [0, 0];

fn step(current: f32[2]) -> f32[2] {
    let u = v * [0.95, 0.95] + current;
    let spike = ge(u, [1, 1]);
    v <- u - spike;
    return spike;
}

fn reset() {
    v <- [0, 0];
}
```

一次 `step` 对应宿主约定的一次离散时间步。这里的 0.95 是预先选定的泄漏系数，可以来自 `exp(-Δt/τ)`；`current` 是这一时间步已经离散化的输入增量。达到阈值就发出 0/1 脉冲，减去阈值完成软复位。每个时间步最多输出一个脉冲，过强输入可能让复位后的电位仍超过阈值；这属于该离散规则的明确定义，不能暗称连续系统精确求解。当前示例没有不应期、延迟、突触电流状态或多脉冲编码。[LIF 动力学与阈值/复位](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

在内部元组与标准库就绪后，同一更新可以简化为以下**后续草案**：

```wgt
let (next_v, spike) = snn.lif(v, current, [0.95, 0.95], [1, 1]);
v <- next_v;
return spike;
```

标准库负责可复用公式，模型负责明确状态属于谁、何时提交；多层模型可以组合多个这样的单元。神经元没有必要成为语言内建对象，时间步没有必要成为专用循环语法。

加入 STDP 时，膜电位、前后脉冲迹、不应期计数等都是普通 state。必须固定“先使用旧迹计算更新，再把当前脉冲写入迹”等顺序，并定义同一时间步的前后脉冲如何处理。每个 episode 的 reset 应覆盖所有相关状态，权重是否重置另外声明。初版可以提供一条明确定义的 pair-STDP 规则，之后通过标准库添加变体。

稠密的 f32 脉冲模拟不自动成为事件驱动执行，也不自动获得低能耗。事件队列、稀疏计算、真实时间调度和专用硬件后端是另外的执行能力，应在需要时有独立合同与测量。

### 2. FF：局部目标可以直接算更新

首期建议只承诺一种容易验证的 Dense/ReLU FF 规则。对一层，设输入为列向量，`W` 的 shape 为 `[输出宽度, 输入宽度]`：

\[
h=\operatorname{ReLU}(Wx+b),\qquad
g=\sum_j h_j^2,\qquad
p=\sigma(g-\theta).
\]

设正样本的标签为 `t=1`，负样本为 `t=0`，固定阈值 `θ`，最大化局部二元对数似然。将输入视为常量，直接求导可得：

\[
\Delta W=2\eta(t-p)h x^\top,\qquad
\Delta b=2\eta(t-p)h.
\]

这是根据原论文目标做的简化解析推导，不是新的通用学习定理。ReLU 为零时取零更新；公式暂不包括正则项、动量和其他层结构。正负样本贡献在同一份旧参数上计算，按明确的 batch 求和或平均规则合并，再通过一次 `adapt` 原子提交。标准库返回更新候选，执行路径不需要 `backward()`、反向 tape 或自动微分器。[FF 原论文](https://arxiv.org/html/2212.13345v1)

goodness 使用归一化前的 `h` 计算；下一层接收长度归一化的活动，例如 `h / max(||h||₂, ε)`，其中 `ε > 0`。零向量保持为零，长度小于 `ε` 时输出并非单位向量。这不是现有 `layer_norm` 的减均值、除标准差操作；必须补齐正确的数值合同，不能为了复用名字而改变算法。当前层只更新自己的参数，下一层目标不能跨层回传；共享参数跨层使用的更新归属也需单独规定。

正负样本如何构造、层间训练顺序、数据加载和迭代次数属于宿主 FF 训练器。模型库提供该层的 goodness、归一化和更新公式。首期不承诺任意卷积块、循环结构或硬脉冲单元都能直接使用上式；硬脉冲需要明确的学习规则、可观测量或资格迹。也可以用 GA 优化某个 FF 目标，但那是目标与优化器的一种组合，不能直接称为原论文更新方式。

如果“无反向传播”的意思是完全不调用 AD，上述解析实现符合目标；如果要求数学上完全不用梯度，则需要选择明确的直接可塑性规则或纯 GA，而不能将 FF 的局部梯度改称无梯度。

### 3. GA：原生支持由候选生命周期决定

遗传算法首先需要一个可执行模型和一组可变参数，并不需要可微计算。对于固定拓扑的 SNN、普通网络或混合模型，都可以复用同一 GA 引擎。

下面是**拟议宿主接口的伪代码，不是当前 SDK API**；Python 和 Rust 应有等价能力，Rust 实现无需 Python 参与：

```python
search = GA(model.genome_schema(), seed=7, population_size=64)

for generation in range(100):
    candidates = search.ask()
    scores = evaluate_isolated(
        model,
        candidates,
        reset_operation="reset",
        episode_seeds=[11, 23, 37],
        fitness=fitness,
    )
    search.tell(scores)

model.update_parameters(search.best.parameters)
```

`ask/tell` 把候选产生和评估分开，使本地、分批或分布式执行复用同一算法。官方库应提供基础选择、变异、精英保留和可选交叉，而不把这些基本工作全部留给调用者。模型语言只负责候选在自身参数、状态和输入下如何执行。

## 五、原生 GA 必须补齐的合同

### 1. 哪些能力已经可复用

Rust 当前已有参数枚举和 `update_parameters`；批量更新会检查名称、shape、有限值与 learn 合法域，并提交一个候选的参数集合。这里的“批量”不是同时评估整个种群。现有更新排除 frozen 与概率参数，不能把“没有 frozen”直接当作可变基因；公开 schema 必须显式描述更新资格。外部普通参数更新还会重置优化器状态。[Rust 模型接口](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgt-model/src/lib.rs)、[参数更新校验](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgtime-core/src/adaptive/reload.rs)

Core 已有 `forward_with_parameters` 的临时参数求值能力，会在结束后恢复原存储；该路径建立 Tape，尚不是这里建议的统一无 AD 候选评估接口。后续可以复用其快照经验，不能将公开协议的缺口误述为底层完全没有候选求值。[Core 求值实现](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgtime-core/src/adaptive/autodiff.rs)

typed 操作对参数与状态的写入具有单次调用原子性，失败回滚；这不等于整个 episode 自动回滚。当前普通参数更新路径涉及重新准备与加载，不应提前承诺只是低成本内存拷贝；是否需要候选批处理或缓存复用应由测量决定。[Operations 执行器](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/crates/wgtime-core/src/operations.rs)

Python 还需要与 Rust 对齐的公开原子候选更新接口。直接对参数 `copy_` 或保存未克隆的 `state_dict()` 不能保证候选隔离；在现有实现里装载 typed 权重也不会自动重置神经状态。不同候选应使用独立模型，或使用经过验证的完整快照恢复机制。[Python 具名操作](https://github.com/WeightLang/WeightLang/blob/8011fb8afc1bcf9913f22b11f995705e6936e5c3/python/weightlang/operations.py)

### 2. 首期实现合同

| 合同 | 必须明确的行为 |
| --- | --- |
| 基因编码 | 稳定参数路径、shape、合法域、共享存储身份与 schema hash；共享参数仅编码一次 |
| 更新资格 | 显式标出可由候选提交的参数；frozen、prob 等现有限制不能被绕过 |
| 离散参数 | bool/int 保持实际合法值；变异算法显式修复或重采样，提交层验证而非偷偷投影 |
| 候选隔离 | 首期使用独立模型实例；参数、神经状态与学习迹不得跨候选污染 |
| 每次评估 | 设置候选参数，调用显式 reset，重置环境及 RNG，执行约定 episode，再计算 fitness；默认 reset 和评估操作不得写参数 |
| 适应度 | 声明最小化/最大化、有限值、失败策略和稳定平局规则；score 绑定候选 ID |
| 随机性 | 种群、变异、环境各自保存宿主整数 PRNG 状态；不能把大整数 seed 塞进 f32 state |
| 失败隔离 | episode 中途失败后丢弃或重新构造受影响 worker，不能只回滚最后一个 step |
| 提交最优 | 完整评估成功后，仅将最优参数原子提交到目标模型；状态保留或重置由调用方明确选择 |
| 可恢复性 | checkpoint 包含参数、种群、代数、fitness、算法配置、schema/版本及相关 RNG 状态 |

复现承诺先限定在相同算法版本、运行时和后端；跨后端的浮点结果使用明确容差。使用共同的 episode seeds 有助于公平比较候选，但不能替代记录完整环境状态。首期利用具名操作的 `writes_parameters` 效果检查默认禁止评估过程写参数。若显式启用评估中的局部学习，还必须规定每个 episode 从哪个参数快照开始，以及适应后的参数是否遗传，不能静默改变 GA。

首期的独立 worker 可以使用各自的模型与连接，不能假定当前 Rust 模型句柄能跨线程共享。先保证正确性，再根据剖析结果设计并行与批量调用。

### 3. 固定拓扑与拓扑进化分开交付

首期固定同一个模型定义、参数名称与 shape、state schema 和操作签名；只进化允许的参数值。可以支持权重、偏置、合法 bool/int 配置以及已有静态上界内的开关或深度。它们可能改变执行路径，但不改变编译后的存储结构；如需限制可变配置，可显式使用参数白名单。增加节点或连线的存储、改变层宽通常需要新的模型 schema，不能假装只是一次参数更新。

NEAT 一类拓扑进化还需要结构基因、结构交叉规则与重新编译/实例化流程。它可以建立在相同宿主协议上，但属于独立阶段，不应写进第一版 GA 的已支持范围。[NEAT 原论文](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## 六、实施顺序和验收

建议从一个小的可运行闭环开始，同时确定统一模型的目标；不要为等待整体重构而推迟所有实际功能。

| 阶段 | 交付 | 验收重点 |
| --- | --- | --- |
| 0：语义合同 | 定义参数域、读写效果、step/reset、无 AD 策略及版本迁移 | 新旧含义可区分；没有隐式代理梯度或隐式状态重置 |
| 1：最小 SNN 与 GA | 补 `ge`；Operations 上实现 LIF/选定 STDP；Rust 官方固定拓扑 GA；Python 原子更新对齐 | 硬阈值、状态复位、合法域、候选隔离及无需 Python 的 Rust 全流程 |
| 2：可用 FF 库 | 补所需数值/标量能力；Dense/ReLU 解析更新；长度归一化；正负样本与层调度 | 解析更新正确、只更新当前层、明确 batch 合同、训练路径无 AD |
| 3：统一与复用 | 合并模型表示；内部元组；有状态子模型；旧语法兼容 lowering | 同实例共享、异实例独立、子调用效果传播与整次调用事务 |
| 4：按瓶颈优化 | 候选参数快速切换、快照、批量时间步或候选评估 | 先测量开销，再优化；保留隔离、版本与数值合同 |

阶段 1 可以先将简短公式写在操作体中；阶段 3 再以元组函数和状态子模型消除重复。这样最先增加的语言计算原语可以只有 `ge`，算法名称全部留在库层。后续补充的除法、范数与归约是通用数值能力，不能靠改名或把未支持功能藏进“库”就省略实现。

### 必须通过的行为验收

1. **比较合同。** 覆盖等于阈值、正负极值、shape 错误与非有限输入政策；Rust 与 PyTorch 都返回精确 0/1。直接比较而非先相减，避免两个有限值相减溢出。AD 请求明确失败或经过显式不可微处理。
2. **脉冲动力学。** 用独立标量参考计算逐时间步核对电位和脉冲；验证连续两次 episode 在 reset 后不串状态。声称支持不应期、延迟或迹时，分别增加边界案例，不能用一次 forward shape 检查代替。
3. **FF 局部性。** 对简化层的解析更新用独立导数或有限差分验证；测试框架可以使用 AD 作参考，但生产无 AD 路径不得调用它。验证正负贡献共享旧参数、长度归一化、其他层参数不变以及确定的 batch 缩放。
4. **GA 正确性。** 用可穷举的小 bool/int 搜索空间检验编码、选优与最优保留；设置已知最优的玩具目标，记录多个种子、预算和命中率，不要求随机算法保证有限步找到全局最优。测试失败候选、reset、非法参数和恢复 checkpoint 后继续运行。
5. **组合与事务。** 验证共享子模块不重复编码参数，各实例 state 隔离；任一子调用失败时，整个父操作不提交部分状态。WGT 0.5 已有 forward 写入行为不能被静默改掉。
6. **完整链路。** 新能力需要源语法、类型/shape 校验、序列化、Rust 解释与编译执行、PyTorch 和 SDK 同步验收。现有 44 项 IR 的覆盖不能自动证明未来 `ge` 或状态子模型已被覆盖。

精度和性能另做小规模端到端实验。至少报告数据与 episode 设置、训练预算、fitness 调用次数、硬件、种子及时间；不同算法不能只比较迭代数。没有实验之前，不承诺 FF 优于 BP、GA 更快或 SNN 更省电。

## 七、证据与自审

### 主张—证据表

| 主张 | 证据 | 当前结论边界 |
| --- | --- | --- |
| 当前有三个互斥计划和 44 项 Adaptive IR | 固定提交的前端与 tensor_contracts 清单 | 属于现状，不代表新增算法库已存在 |
| 当前能表达标量状态单元，缺少通用向量 `ge` 和 helper 元组 | 本轮三个编译入口检查 | 仅证明语法/转换边界，未验证 SNN 数值与学习效果 |
| FF 可使用层内目标，不需要跨层误差回传 | Hinton 原论文；本文展示目标与解析推导 | 首期限定 Dense/ReLU，仍是局部梯度，不泛化到任意硬脉冲层 |
| SNN 不等于无反向传播 | LIF 与 surrogate-gradient 一手资料 | 模型架构与学习方法分别选择 |
| GA 可不依赖导数训练固定拓扑网络 | Deep GA 原论文 | 不据此保证任意任务效率，也不包含 NEAT 式动态拓扑 |
| 少量共用机制比增加算法关键字更适合本项目 | 现有组合缺口与三个例子的共同需求 | 架构建议，需上述实现及行为验收验证 |

### 五项自审

- **贡献清晰度：** 提议统一状态、参数、函数与效果，新增算法关键字为零；避免把 API 数量当成语言复杂度的全部。
- **表达清晰度：** 明确区分现状、语法草案、数学推导和未来宿主 API；显式说明 FF 的局部梯度与无 AD 的区别。
- **实证依据：** 当前能力有固定源码与编译边界检查；算法机制有原论文。本文没有新算法性能数据。
- **评估完整性：** 给出数值、局部性、隔离、事务、兼容和跨后端检查；实现完成前不把这些计划写成已通过。
- **设计合理性：** 保持模型语言负责内部计算和更新，宿主负责数据、群体、设备及调度；第一步只补实际缺失的通用能力。

本次交付是语言设计评审稿。Rust 全指令覆盖是此前实现的基线，新语言能力和算法库仍需按阶段开发。
