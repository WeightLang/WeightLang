# WeightLang 训练策略与模型生命周期

> 本文记录当前可用边界与后续工程合同。文中标为“设计目标”的内容不代表已经实现或发布。

## 设计原则

WeightLang 不把权重更新算法绑定到模型定义，也不把训练限制为 SGD/Adam。训练系统拆成三个可组合阶段：

1. **Differentiate**：执行模型与 loss，返回 loss、参数快照、梯度和版本号；不修改模型参数。
2. **Update strategy**：用户策略根据梯度、参数、历史状态及额外评估，计算候选参数和下一份策略状态。
3. **Commit**：验证快照仍有效、参数名称/shape/有限性/learn 域、状态大小与预算；全部通过后原子提交参数和策略状态。

这一边界允许梯度优化、无梯度搜索及混合策略共存。模型定义只声明计算和可学习参数，不声明唯一 optimizer。

## 当前自定义更新能力

CPU forward 路径已有 WGT 自定义更新接口，详见 [原生训练](native-training.md#WGT-自定义更新算法CPU-forward)：

- `differentiate(model, inputs, targets)` 计算批次平均 loss 和非冻结参数梯度，不提交更新。
- `differentiate_at(gradient, parameters)` 在临时绝对参数上重新求导，真实模型保持不变。
- `gradient_snapshot(gradient)` 读取参数、梯度、名称、shape 和用户状态。
- `commit_gradients(gradient, parameters, state)` 原子提交候选参数及 JSON 兼容策略状态。
- `release_gradient(gradient)` 释放未提交的梯度快照。
- Checkpoint v2 保存模型、用户策略状态和自定义提交步数。

普通 WGT host 函数本身不自动微分。它可以读取梯度并编写任意更新公式。自定义路径目前限定 CPU forward；不应将它描述成 GPU 自定义优化器插件。

### 自定义 Momentum 示例（算法伪代码）

下面描述策略的数学逻辑；具体 WGT 容器/索引语法以语言参考和可执行 fixture 为准：

```text
g = differentiate(model, batch)
snapshot = gradient_snapshot(g)

for each trainable parameter i:
    velocity[i] = beta * old_velocity[i] + gradient[i]
    candidate[i] = parameter[i] - learning_rate * velocity[i]

commit_gradients(g, candidate, { velocity, beta, learning_rate })
```

把 `beta` 设为 0 得到 SGD；保留 velocity 得到 Momentum。Adam 可在用户状态中保存一阶、二阶矩和步数。AdamW 可将权重衰减与梯度方向分开。Lion 可保存符号更新所需状态。SAM 可先调用 `differentiate`，构造扰动参数，再用 `differentiate_at` 求第二次梯度，最后只提交一次。

### 无梯度与混合优化

同一更新边界可支持：

- **Evolution Strategies**：生成多个候选参数、评估目标、按排名或加权回报更新。
- **遗传算法**：选择、交叉、变异并提交候选。
- **随机搜索 / 坐标搜索**：不需要反向传播。
- **Bayesian Optimization**：外部代理模型选择下一批候选。
- **混合算法**：先用梯度法预训练，再用搜索策略优化离散结构或非光滑目标。

限制：当前 `differentiate_at` 仍是给定批次上的一阶梯度重算，不是任意高阶自动微分；策略自己承担候选生成、额外评估、状态版本管理和算法正确性。

## Learn 参数如何进入反向传播

### 连续 float

普通 float 参数直接参与计算图，梯度按算子 VJP/链式法则回传。SGD/Adam 或自定义策略都可以使用这个梯度。

### int / bool：Shadow + STE

离散值本身通常不可直接使用有意义的连续导数。当前实现维护连续 shadow：

- int 前向读取使用 round(shadow)。
- bool 前向读取使用 shadow >= 0.5。
- 反向对离散化节点使用 identity straight-through estimator：把上游梯度原样传给 shadow。
- 优化器更新 shadow；int/bool 的前向值由 shadow 再次离散化。
- bool shadow 限制在 [0,1]；有界 int/float 在更新后按声明域处理。

STE 是代理梯度，不是 round 或阈值函数的真实导数。它允许优化器获得训练信号，但不保证离散架构搜索的全局最优性。对更复杂的离散决策，应允许 Gumbel-Softmax、REINFORCE、进化策略、贝叶斯优化等替代方法，而不是强制 STE。

## Checkpoint 与模型权重的区别

| 制品 | 内容 | 用途 |
|---|---|---|
| Model Definition | WGT 结构、参数 schema、初始化声明 | 重新解析和构建架构 |
| Weights | 模型参数 | 推理或初始化训练 |
| Training Checkpoint | 参数、优化器/策略状态、步数等 | 恢复训练 |
| Experiment Record | 配置、数据版本、指标、环境、制品引用 | 复现实验 |
| Inference Package | 定义/执行契约、权重、Runtime 元数据 | 部署推理 |

普通 safetensors 权重不是训练 checkpoint。自定义策略恢复时必须使用相同算法及兼容的策略状态 schema；checkpoint 保存状态，不会自动恢复用户策略代码、数据游标、外部环境或 RNG，除非应用明确把这些内容纳入自己的恢复合同。

## 项目工程化目标

推荐项目布局：

```text
project/
  wgt.toml
  wgt.sum
  model.wgt
  src/                 # Rust 数据管线、训练循环、评估和推理入口
  modules/             # 可复用 WGT 模块（如项目需要）
  data/                # 数据清单/版本引用；大型原始数据可外置
  experiments/
    baseline.toml
    runs/
  weights/             # 可发布的 safetensors
  checkpoints/         # 训练恢复点
  reports/             # 评估摘要
  tests/               # 前向、梯度、训练、恢复和推理回归
```

建议每次运行记录：源码 commit、WGT 版本、架构/参数 schema hash、数据集版本及 split、预处理、随机种子、训练配置、Runtime/设备版本、开始结束时间、指标、停止原因、权重/checkpoint SHA256。

不要把大型数据、临时 tape、缓存或可重建文件提交到 Git。正式权重与 checkpoint 应通过明确的制品策略管理，不应被普通缓存清理删除。

## 目标生命周期

```text
wgt new / sync
    -> weightc check
    -> inspect model schema
    -> initialize model
    -> load data and create experiment
    -> train (built-in or user strategy)
    -> evaluate on held-out data
    -> save weights + resumable checkpoint
    -> restore and verify continuation
    -> export inference package
    -> load in independent inference process
```

## 尚需完成的工程工作

以下是明确的后续开发目标，不是现有能力声明：

1. **统一训练策略协议**：把内置 SGD/Adam 与 WGT 自定义策略统一到同一生命周期；策略声明版本、状态 schema、所需梯度、是否需要多次评估及支持的设备。
2. **稳定公共 API**：提供受版本约束的 Rust `TrainingStrategy` / `GradientSnapshot` / `UpdateCandidate` 接口，或等价的协议；不能要求所有算法实现成固定枚举。
3. **事务和并发**：梯度快照绑定模型 revision；提交必须拒绝 stale snapshot；一次提交原子更新参数和策略状态。跨多个请求的策略执行不是自动事务。
4. **Checkpoint vNext**：记录模型指纹、策略 ID/版本、策略状态 schema、参数、全局步数、优化器状态、scheduler、RNG 和数据 sampler 状态；对未知字段/版本 fail closed。
5. **Experiment Manager**：稳定的 run ID、配置快照、指标追加、失败/中断状态、制品引用、恢复来源与比较。
6. **独立推理包**：从训练 checkpoint 导出只含推理所需内容的制品，验证架构指纹和参数 schema。
7. **设备能力合同**：CPU/GPU 对前向、反向、内置更新、自定义策略分别声明能力；不得因检测到 GPU 就推断整个训练策略在 GPU 上运行。
8. **端到端验收**：CNN 与 Transformer 分别完成数值参考、梯度有限差分、训练、checkpoint 恢复、独立推理；另测一个 WGT 自定义梯度策略和一个无梯度搜索策略。

## 验收要求

- 对每个可微算子：Forward 对照独立参考，VJP 对照中心有限差分。
- 对每个更新策略：验证候选有限性、参数域、策略状态 schema、失败原子性和确定性范围。
- 对 checkpoint：保存、关闭进程、恢复、继续训练，并核对权重、策略状态、步数和后续结果。
- 对推理包：在不加载 optimizer/tape 的独立进程中加载，验证模型指纹和预测结果。
- 对实验记录：能够从记录追溯源码、数据、配置、环境和输出制品。

性能报告必须区分纯前向、反向、参数更新、数据准备和 IPC 开销；不能把单次总时间当作 Kernel 性能。
