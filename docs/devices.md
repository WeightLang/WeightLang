# 原生执行设备

WGT 原生训练脚本目前在 CPU 上执行，使用 f32 张量。独立原生 IR 推理入口支持 CPU，以及编译并运行环境提供的 GPU 后端；列出本机可用设备：

```sh
wgt-native --devices
```

`wgt-native MODEL.json BUDGET_BYTES` 的 `--device` 可选 `auto`、`cpu`、`gpu` 或具体 `BACKEND:N`；默认 `auto` 在支持的候选中选择。`--check` 只准备并检查会话，不执行输入；`--strict-zero-alloc` 限定 CPU 执行，不能与显式 GPU 一起使用。实际可用后端取决于构建功能、设备和驱动，设备枚举不证明模型已经在该设备通过数值或性能验证。

自更新 `wgt-native adaptive` 当前只支持 CPU。运行中的权重视图见 [weightfs](weightfs.md)；GPU 实机编辑仍需在目标设备验证。内存预算覆盖范围、零分配契约和 GPU 边界见[原生执行后端](native-backend.md#内存与生命周期契约)。

Rust API 另有实验性顺序 IR GPU 训练：**Rust API 实验性，WGT 绑定尚未接入**。
启用 runtime `gpu` feature，以 `GpuTraining::new(device,max_batch,budget_bytes)` 指定当前枚举的精确
硬件 ID，再调用 `Trainer::set_backend(TrainingBackend::Gpu(...))`。不自动测速、选卡或回退 CPU。
支持 f32 batch linear/relu/tanh/sigmoid/identity、MSE 反传与原子 SGD 更新；不支持 GPU Adam、
交叉熵、REINFORCE、多 GPU 或任意 WGT forward 程序。完整签名、预算、容差和实机命令见
[实验性 Rust GPU 训练 API](native-training.md#实验性-rust-gpu-训练-api)。

每次事务从已提交 CPU 权重上传，GPU 保留完整 tape 并计算候选；所有计算和回读校验通过后才提交。
CPU 副本使设备错误后仍可导出最后成功权重，恢复断点默认 CPU。当前实现同步回读、逐步分配，
不承诺训练加速或显存常驻；`predict` 仍是 CPU。原 WGT fit/reinforce 和 CLI 设备参数保持原行为。
