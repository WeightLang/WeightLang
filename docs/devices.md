# 原生执行设备

当前 WGT 语言仍保留训练脚本中的 `training_device`，但新的 Rust 模型 facade 契约尚未实现设备选择 API。设备枚举、独立 IR 推理 CLI `wgt-native` 已删除；不要把底层 `wgtime-core` 能力描述成新 facade 已公开的功能。新模型定义、实例与服务边界见[原生后端契约](native-backend.md)。

底层 `wgtime-core` 源码仍实现 CPU 及可选 GPU 执行路径；GPU 能力取决于构建 feature、设备与驱动。设备枚举不证明模型已在该设备通过数值或性能验证。自更新底层会话当前只支持 CPU。运行中的权重视图见 [weightfs](weightfs.md)；GPU 实机编辑仍需在目标设备验证。内存预算覆盖范围、零分配契约和 GPU 边界见[原生执行后端](native-backend.md#内存与生命周期契约)。

底层 core 中保留顺序 IR GPU 训练和 WGSL 能力，WGT 脚本路径的 `training_device` 与 `wgsl_kernel`/`wgsl_run` 见 [WGSL 与 GPU 训练](wgsl.md)。这些不是新 Rust facade 的公开设备选择 API；新的 facade 需由 core/service 集成承接后再提供稳定选择接口。底层支持的算子、预算、精度和限制以 core 实现及其测试为准；不据此承诺多 GPU 或任意 WGT forward 程序。

底层 core 的 GPU trainer 事务从已提交 CPU 权重和优化器矩上传，在 GPU 计算候选；验证成功后再提交权重、矩与步数。该路径保留 CPU 副本，并同步回读，不承诺训练加速或跨步权重常驻。它目前没有接入新 Rust 模型 facade/service 的设备选择接口；这些底层实现细节不构成 facade 能力承诺。
