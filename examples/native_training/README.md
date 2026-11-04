# 原生 WGT 示例入口

每个示例独占一个目录，入口统一为 **`model.wgt`**。示例名称写在目录名中；
`fn main()` 是文件内的执行函数，不应据此把入口文件命名为 main.wgt。
模型声明、forward、loss 与训练 main 可放在同一入口中，检查/编译不会执行训练。

从仓库根目录运行：

```sh
weightc check examples/native_training/working_memory/model.wgt
wgt run examples/native_training/working_memory/model.wgt
wgt run examples/native_training/working_memory/model.wgt --compiled
```

| 示例入口 | 用途与运行条件 |
| --- | --- |
| [gpu/model.wgt](gpu/model.wgt) | 显式 GPU 顺序模型训练；需要 gpu feature 和真实硬件 |
| [wgsl/model.wgt](wgsl/model.wgt) | 原生 WGSL 编译与调度；固定 f32 输入/输出 ABI |
| [supervised/model.wgt](supervised/model.wgt) | 合成数据监督训练，读取共享的 `../samples.jsonl` |
| [build/model.wgt](build/model.wgt) | 返回训练后的模型，供 `weightc legacy-build ... -o weights/<新目录>` 使用 |
| [reinforce/model.wgt](reinforce/model.wgt) | 原生策略梯度训练 |
| [soft_routing/model.wgt](soft_routing/model.wgt) | 可微路由训练 |
| [lazy_routing/model.wgt](lazy_routing/model.wgt) | 推理分支短路 |
| [prob_mix_loss/model.wgt](prob_mix_loss/model.wgt) | 概率参数、mix 与自定义损失 |
| [sample_branch/model.wgt](sample_branch/model.wgt) | 普通脚本概率抽样与独热编码 |
| [working_memory/model.wgt](working_memory/model.wgt) | 可微工作内存与分步推理 |
| [compute_policy/model.wgt](compute_policy/model.wgt) | 学习选择固定计算能力 |
| [cost_aware_policy/model.wgt](cost_aware_policy/model.wgt) | 成本感知策略训练 |
| [custom_budget/model.wgt](custom_budget/model.wgt) | 自定义能力和子预算 |
| [checkpoint/model.wgt](checkpoint/model.wgt) | 保存重载；先创建 `weights/native_training/`，目标须不存在 |
| [compiled/model.wgt](compiled/model.wgt) | 编译后训练与保存；先创建 `weights/native_script/`，目标须不存在 |
| [inline/model.wgt](inline/model.wgt) | 单文件依赖注释 |
| [huggingface/model.wgt](huggingface/model.wgt) | 读取在线数据，需要网络 |
| [postfix_mean/model.wgt](postfix_mean/model.wgt) | 后缀平均、前向归约和自定义损失训练，无需外部数据 |
| [host_resources/model.wgt](host_resources/model.wgt) | 文件读取，需要宿主授权 data 别名 |
| [host_write/model.wgt](host_write/model.wgt) | 文件写入，需要宿主授权 result 别名及额度 |
| [device_queue/model.wgt](device_queue/model.wgt) | 设备推理，需要可用后端与 accelerator 授权 |

所有相对路径均从各自 `model.wgt` 所在目录解析。共享样本文件保留在本目录，
保存到仓库根目录 weights 的示例使用 `../../../weights/...`，不会写入 examples 下。
保留模型、断点与报告遵循[存储规范](../../docs/storage.md)，不覆盖既有结果。

这是源码入口约定；制品归档中的 `source.wgt`、模型 JSON 与优化器断点保持原格式。
标准库模块仍按功能命名，不作为独立模型入口。其他示例为
[自更新模型](../self_modifying/model.wgt)和[包调用模型](../package_usage/model.wgt)。

- [ppo/model.wgt](ppo/model.wgt)：WGT 原生训练语句、独立目标形状、两步延迟奖励、GAE 与 PPO 小批次更新；合成环境验证，非真实任务泛化证明。
