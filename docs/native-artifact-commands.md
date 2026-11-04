# 原生模型导出和目录打包

`wgt export INPUT -o OUTPUT.json` 导出已有原生模型的推理 IR，不初始化源码、不执行 `main()`、不训练。INPUT 是 `weightc build` 创建的完整原生构建目录，或自包含的原生顺序模型 JSON。目录输入先验证 manifest、文件哈希、接口及 checkpoint 与模型一致性；校验会读取训练断点，但导出结果只含推理结构和参数，不含优化器状态，也不生成 checkpoint。

`wgt package --entry BUILD_DIRECTORY [-o OUTPUT_DIRECTORY]` 复制并校验完整原生构建制品，默认输出 `dist/native-package`。保留 manifest 指定契约内的源码快照、推理模型和独立训练断点；不执行源码。输出是目录包，不能用 `.zip`、`.wgtpkg`、`.tar`、`.gz`、`.tgz` 或 `.zst` 扩展名冒充归档文件。需要发布资产时继续使用 `wgt release-assets`。

```sh
wgt package --entry weights/my-model -o dist/my-model
wgt export dist/my-model -o dist/inference.json
```

两个命令都拒绝已有目标，包括符号链接；不支持 `--force`。package 不支持旧 `--weights` / `--reference`；export 不支持 `--trainer` 或设备放置参数。输入 payload 必须是普通文件，拒绝符号链接和依赖外部参数文件的模型，避免把目录外的材料隐式加入包。当前只支持原生顺序模型，不支持 forward Program IR、旧制品、源码初始化和 checkpoint 作为单文件推理输入。

迁移旧 `wgt export model.wgt`：先显式执行训练或构建，在获得保存的推理模型后执行 export。检查和加载源码不应触发训练。export 不替代训练断点保存；恢复 Adam 等优化器状态仍须使用独立 checkpoint。

发布前先校验源目录，将文件复制到同盘临时目录并复验。目录发布以 manifest 最后写入作为完成标记，不宣称整个目录原子替换；若发布途中失败，无完整 manifest 的部分目录保留供诊断。模型文件导出沿用原生 no-clobber 写入方式，不覆盖已有内容。

行为测试见 `crates/wgt/tests/native_artifact_commands.rs`，完整训练至导出推理测试见[生命周期示例](../examples/lifecycle/README.md)。
