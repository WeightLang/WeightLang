# 原生模型制品

`weightc legacy-build SOURCE -o DIRECTORY` 显式执行脚本的 `fn main()`，要求返回普通模型句柄，然后写入完整的原生模型目录；`build` 是兼容别名。脚本是否训练由 `main` 本身决定，检查或编译源码不会执行训练。

| 文件 | 内容和用途 |
| --- | --- |
| `model.json` | 原生推理模型，包含结构和权重；可用 `model(path,optimizer)` 加载或由 `wgt-native` 运行。 |
| `checkpoint.json` | 模型和优化器状态，可通过脚本 `restore(path)` 续训。 |
| `source.wgt` | 入口脚本快照，不包含外部数据或依赖。 |
| `manifest.json` | `weightlang.native-build.v1` 元数据、文件 SHA256、输入输出契约、操作预算、更新步数和指标；最后写入，作为构建完成标记。 |

应完整保留目录。已有输出目录或符号链接会被拒绝，原生构建不提供覆盖选项；如果发布中途失败，可能保留不完整目录供诊断，不能当作完整构建。`initialized` 表示没有优化器更新，`updated` 表示有更新历史；后者不证明真实任务质量或泛化能力。

训练脚本也可以显式用 `save_model(model,path)` 保存推理模型，用 `save(model,path)` 保存训练断点；两者是独立文件，不能互换。断点不包含数据流位置、脚本局部变量、随机数状态或环境进度，完整复现需要调用者另外保留这些信息。完整接口和恢复示例见[原生训练](native-training.md#保存与续训)。

[weightfs](weightfs.md) 是运行中的易失权重视图，文件编辑只改变进程内模型，退出不会自动持久化。完整模型和实验基线应保存在 `weights/`，详见[存储规范](storage.md)。
