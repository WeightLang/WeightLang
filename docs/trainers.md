# 原生训练

WGT 0.5 脚本通过显式调用训练函数，模型加载和检查不会自动训练。训练在当前 Rust 实现中使用 CPU 和 f32 张量，解释与编译模式共享训练运行时。

| 签名 | 作用 |
| --- | --- |
| `mlp(widths,activation,seed,optimizer)` | 初始化一个顺序网络，返回可复用的模型句柄。 |
| `train model on inputs against targets { rate: lr; loss: mse; }` | 对一个批次更新一次，返回更新前的 f32 批次平均损失；`rate` 必填，`loss` 默认 `mse`，也可选 `cross_entropy` 或自定义 `loss`。 |
| `reinforce(model,observations,actions,advantages,lr)` | 使用显式动作和优势进行一次策略梯度更新。 |
| `save checkpoint VALUE to PATH;` / `load checkpoint from PATH` | 保存和恢复含优化器状态的训练断点。 |
| `save model VALUE to PATH;` | 保存不含优化器的推理模型。 |

SGD 无 momentum；Adam 固定 beta1=0.9、beta2=0.999、eps=1e-8，均无 weight decay。同一模型句柄保留优化器状态。可运行例子见 [supervised/model.wgt](../examples/native_training/supervised/model.wgt) 和 [reinforce/model.wgt](../examples/native_training/reinforce/model.wgt)。

完整参数、维度、限制、数据接口和训练断点规则见[原生训练](native-training.md)。加载或初始化结果不能当作已训练或任务质量已验收。

声明式训练可写 `train model on inputs against targets { rate: lr; loss: loss; clip: max_norm; }`；`clip` 可选且目前仅 CPU 支持，对批次平均梯度做全局 L2 裁剪后原子更新。forward 自定义损失与独立目标形状见[原生 RL](reinforcement-learning.md)。`train model on inputs against targets using strategy` 调用同步用户策略并返回其有界纯数据结果（最大深度 32、计量 1 MiB；允许 null、拒绝资源句柄）；策略可零次或多次显式提交，训练语句不隐式提交，失败不撤销此前已成功的显式提交。两种形式的模型、输入、目标及配置/策略参数按源码顺序各求值一次；旧 fit/train 函数调用与 tuple 形式均拒绝。
