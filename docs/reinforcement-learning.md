# 原生强化学习语法（WGT 0.5，当前源码）

WGT 负责环境、轨迹、优势、损失和训练循环，Rust 执行张量求导与优化器。
当前支持 CPU 固定向量 forward 模型上的离散动作 PPO 示例；不是已发布功能或通用环境框架。
没有新增 Python 回调、外部模拟器、连续动作分布、replay buffer 或 GPU forward 训练。

## 显式训练语句

```wgt
wgt 0.5
model Policy
input observation: f32[2]
output logits: f32[2]
parameter weights: f32[2,2] = [0,0,0,0]
# 两个动作的 one-hot、旧 log_prob、常量 advantage。
target trajectory: f32[4]
fn forward(observation) { return linear(observation, weights); }
fn loss(prediction, trajectory) {
    let selected = sum(log_softmax(prediction) * slice(trajectory,0,2));
    let old = detach(slice(trajectory,2,1));
    let advantage = detach(slice(trajectory,3,1));
    let ratio = exp(selected-old);
    return -minimum(ratio*advantage, clip(ratio,0.8,1.2)*advantage);
}
fn main() {
    let policy = Policy("adam");
    # 这个固定批次只演示一次更新；完整环境采集见 PPO 示例。
    train policy on ([[1,0]], [[0,1,-0.6931472,1]]) rate 0.03 clip 0.5;
    return policy([1,0]);
}
```

完整签名：`train MODEL on (INPUTS, TARGETS) rate RATE clip MAX_NORM;`。
MODEL、INPUTS、TARGETS、RATE、MAX_NORM 均为必填普通脚本表达式，按此顺序各求值一次。
语句降低为 `train(model,inputs,targets,rate,max_norm)`，函数形式返回更新前的批次平均损失；
语句形式丢弃返回值。train 是内置名称，on/rate/clip 在该语句对应位置作为分隔词。
它仅适用于含 `fn loss`、优化器为 sgd/adam 的 forward 模型；普通顺序模型使用原有 fit/reinforce。
不支持在 forward/loss 中训练，也不提供可跨调用保存的梯度对象或单独 backward/step 语句。

inputs 是非空样本列表，每个样本为模型输入宽度的有限 f32 向量；targets 数量必须相同。
rate、max_norm 转为 f32 后必须为有限正数，均无默认值。
整批先从同一组参数求导并取平均，然后对所有非冻结参数的梯度一起计算 L2 范数：
大于 max_norm 时统一缩放，随后提交一次 SGD/Adam 更新。这里的 clip 是梯度范数上限，
与损失内 `clip(ratio,low,high)` 的 PPO 比率裁剪不同。
任一样本、求导、预算或候选更新失败，权重和优化器状态均不提交；操作预算仍按实际执行扣除。
沿用现有每样本模型/损失 tape 共享 64 MiB 预算和指令计费，不是整个进程的内存/FLOP 上限。
检查、编译、构造和加载不执行 train。

## 独立损失目标

`target NAME: f32[N]` 是可选模型声明，只出现一次；N 为 1..16384 固定整数。
NAME 必须与 `fn loss(prediction,NAME)` 的第二个参数同名，必须存在 loss。
未声明时保持原契约：目标宽度等于模型输出宽度。声明后，第二参数及传入 targets 使用 N，
第一参数仍为模型输出宽度；loss 始终返回 f32[1]，按批次取平均。
该声明只改变自定义损失（train 或 fit 的 "loss"）；内置 MSE 的目标仍须与输出等宽。
目标数据不参与参数更新，target 不增加 forward 的输入端口，不改变 predict/sample 输入。
loss 不能访问模型参数、普通辅助函数或 I/O；不能在 main 中直接调用。

PPO 的旧概率、动作和优势来自固定采样批次。先完整采集，再更新；多轮优化期间不重算旧概率。
普通函数可用 `let {action,log_prob}=sample from policy(observation);` 从同次只读模型评估取得动作和稳定旧概率，
不会再执行一次策略调用或 RNG；也支持 `action:a`、`log_prob:lp` 别名。
概率为实际抽样分布的自然对数；logits 使用 f64 log-sum-exp，显式 probabilities 用实际总质量归一化解释。
完整例子见 [sample_record/model.wgt](../examples/native_training/sample_record/model.wgt)。
[PPO 示例](../examples/native_training/ppo/model.wgt) 用 `collect` / `yield` 同时收集 observation、显式 f32[4] target、value_target，
`for minibatch in rollout size 16` 按同一行范围切分所有字段，不默认洗牌且保留尾批。
每批 48 条、4 个 epoch 保持 12 次策略更新；记录只是有界数据容器，不自动转换为可微具名目标。
环境终止时不 bootstrap；时间截断时使用最终观测价值，而非 reset 后的观测价值。
现有 gae 接收单条顺序轨迹，不能直接传交错的并行环境数据。

## forward/loss 可微表达式

以下全部为当前固定 f32 向量运算，无隐式广播、无默认参数。
普通 main 中已有 log/exp/sum/clip/log_softmax 仍遵循普通数值/列表契约，不建立梯度图。

| 表达式 | 完整契约与梯度 |
| --- | --- |
| `[1,-2,0.5]` | 1..16384 个有限 f32 数字字面量，允许负号；不接受变量、嵌套列表或空列表；常量梯度为零 |
| `detach(x)` | 返回同形同值向量，截断此路径梯度；仍计算 x，不省略其错误或预算 |
| `exp(x)` | 同形逐元素指数，导数 exp(x)，溢出拒绝 |
| `log(x)` | 同形逐元素自然对数，x 必须严格为正，导数 1/x；非有限结果/梯度拒绝 |
| `log_softmax(x)` | 同形，沿唯一向量轴稳定计算自然对数概率；VJP 为 g-softmax(x)*sum(g)，无需先求 softmax 再 log |
| `sum(x)` | 沿唯一向量轴求和，输出 f32[1]，梯度广播到各元素 |
| `minimum(a,b)` | 等长逐元素最小值；较小者导数 1，较大者 0，相等时各分配 1/2 |
| `clip(x,low,high)` | 同形逐元素截断；low/high 为有限、有符号数字字面量，low<=high；开区间内导数 1，端点及区间外 0；上下界不求导 |

离散动作采样本身不反传。用 `onehot[N] action` 在普通脚本中构造动作掩码，
传入 target 后通过 `sum(log_softmax(prediction)*mask)` 选取动作的可微 log_prob。
熵可写成 `-sum(softmax(prediction)*log_softmax(prediction))`。

## 可运行示例与验证

[完整 PPO 示例](../examples/native_training/ppo/model.wgt) 包含两步延迟奖励环境、独立价值网络、GAE、
PPO 比率裁剪、熵项、四轮小批次更新和全局梯度裁剪。模型从均匀随机策略开始，随机策略成功率为 0.25。
这是有限状态合成环境，不证明真实任务泛化或大模型 RL 能力；小批次顺序固定，未实现并行环境。

```sh
cargo test --locked --offline -p weightlang-runtime --no-default-features --test rl_syntax
cargo test --locked --offline -p weightlang-wgt --no-default-features --test script_compiler compiled_ppo_uses_native_training_syntax
cargo run --locked --offline -p weightlang-wgt --bin wgt -- run examples/native_training/ppo/model.wgt
```

测试覆盖数值梯度、detach、极端 logits、端点导数、裁剪、错误回滚、多个随机种子和解释/AOT 一致性。
save/restore 保存新增损失程序和优化器，save_model 导出推理参数与程序；新增 IR 指令要求匹配版本运行时，
不保证旧可执行文件能加载新制品。完整模型/报告保留于 weights/<experiment>/，不覆盖现有文件。
断点仍不保存环境、脚本 RNG、采集进度或整批轨迹；它证明模型/优化器续训，不是整个 RL 进程恢复。
