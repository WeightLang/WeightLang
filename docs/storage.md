# 模型与数据存储

完整模型放在 `weights/<model>/`；保留的实验基线、候选结果和报告放在 `weights/<experiment>/`，作为持久数据一起保留。不要将它们当缓存清理，也不要覆盖既有结果；只有提供显式 `--force` 或原生 `save ... overwrite;` 的操作才可以按其具体契约替换。原生构建和兼容保存函数仍拒绝已有目标；原生 save 语句默认拒绝覆盖，显式 overwrite 仅原子替换普通可写文件，拒绝目录和符号链接，详见[文件 IO](file-io.md)。

| 内容 | 位置 | 用途 |
| --- | --- | --- |
| 模型与保留的实验 | `weights/` | 持久保存 |
| 大型数据、下载和构建中间文件 | 磁盘上的项目路径或 `.cache/weightlang/` | 按需缓存 |
| Rust 编译缓存 | `target/` | 仅编译器缓存，不是模型输出 |
| 小型临时文件 | `/tmp/weightlang/` | 仅本任务产生的可丢弃内容 |

预计写入超过 100 MB 前，检查目标目录、剩余空间、实际挂载和冗余内容。大文件不能放在 tmpfs，包括 `/tmp` 或 `/dev/shm` 的内存挂载。结束后只清理本任务产生且不再需要的临时文件，不删除已有权重、实验、缓存或其他工作。

`weightc legacy-build` 默认向 `weights/<source stem>/` 输出 `model.json`、`checkpoint.json`、`source.wgt` 和完成标记 `manifest.json`。这些文件应作为完整目录保留。`save_model` 写入推理模型；`save_checkpoint` 写入包含优化器状态的训练断点，两者不能互换。调用者还需保存数据及脚本进度；断点不自动保存脚本 RNG 和外部依赖。详见[原生训练](native-training.md#保存与续训)。

原生编译的 Cargo 中间文件使用当前目录下的 `.cache/weightlang/aot/`；编译不执行训练或生成权重。单文件脚本的依赖锁位于包缓存中的 `scripts/<脚本规范路径的 SHA256>/wgt.sum`；项目脚本使用项目的 `wgt.sum`。完整路径及依赖规则见[原生训练](native-training.md#单文件依赖注释与直接执行)。

其他项目使用 `/tmp/<project>/` 保存小型临时文件，并使用各自的磁盘缓存。

## 运行中的权重视图

[weightfs](weightfs.md) 自动挂载于 `$XDG_RUNTIME_DIR/weightlang/weightfs/<pid>-<session>/`
（未设置时基于 `/run/user/<euid>`）。这是直接访问进程内权重的 FUSE 内存文件系统，
是本规范明确允许的易失权重视图，不是完整模型制品、下载或大型临时文件。
编辑只改变运行中的模型；退出不自动持久化，仍须显式保存至 `weights/` 的新目标。
