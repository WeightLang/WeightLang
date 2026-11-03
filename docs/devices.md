# 原生执行设备

WGT 原生训练脚本目前在 CPU 上执行，使用 f32 张量。独立原生 IR 推理入口支持 CPU，以及编译并运行环境提供的 GPU 后端；列出本机可用设备：

```sh
wgt-native --devices
```

`wgt-native MODEL.json BUDGET_BYTES` 的 `--device` 可选 `auto`、`cpu`、`gpu` 或具体 `BACKEND:N`；默认 `auto` 在支持的候选中选择。`--check` 只准备并检查会话，不执行输入；`--strict-zero-alloc` 限定 CPU 执行，不能与显式 GPU 一起使用。实际可用后端取决于构建功能、设备和驱动，设备枚举不证明模型已经在该设备通过数值或性能验证。

自更新 `wgt-native adaptive` 当前只支持 CPU。运行中的权重视图见 [weightfs](weightfs.md)；GPU 实机编辑仍需在目标设备验证。内存预算覆盖范围、零分配契约和 GPU 边界见[原生执行后端](native-backend.md#内存与生命周期契约)。
