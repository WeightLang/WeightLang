# 原生训练

WGT 0.5 脚本通过显式调用训练函数，模型加载和检查不会自动训练。训练在当前 Rust 实现中使用 CPU 和 f32 张量，解释与编译模式共享训练运行时。

| 签名 | 作用 |
| --- | --- |
| `mlp(widths,activation,seed,optimizer)` | 初始化一个顺序网络，返回可复用的模型句柄。 |
| `fit(model,inputs,targets,lr,loss)` | 对一个批次更新一次，返回更新前的损失；`loss` 为 `"mse"` 或 `"cross_entropy"`；forward 模型另支持 `"loss"`。 |
| `reinforce(model,observations,actions,advantages,lr)` | 使用显式动作和优势进行一次策略梯度更新。 |
| `save(model,path)` / `restore(path)` | 保存和恢复含优化器状态的训练断点。 |
| `save_model(model,path)` | 保存不含优化器的推理模型。 |

SGD 无 momentum；Adam 固定 beta1=0.9、beta2=0.999、eps=1e-8，均无 weight decay。同一模型句柄保留优化器状态。可运行例子见 [supervised.wgt](../examples/native_training/supervised.wgt) 和 [reinforce.wgt](../examples/native_training/reinforce.wgt)。

完整参数、维度、限制、数据接口和训练断点规则见[原生训练](native-training.md)。加载或初始化结果不能当作已训练或任务质量已验收。

forward 自定义损失另支持 `train model on (inputs,targets) rate lr clip max_norm;`，全局梯度裁剪后原子更新；独立目标形状与 PPO 示例见[原生 RL](reinforcement-learning.md)。
