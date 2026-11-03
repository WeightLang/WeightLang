# wgt 命令行

`wgt` 管理项目、依赖和工具链，并运行 WGT 0.5 脚本。`weightc` 提供脚本检查、编译和原生模型构建。当前命令的完整参数以 `wgt --help`、`wgt <命令> --help` 和 `weightc --help` 为准。

```sh
# 检查只解析，不训练、不构造模型，也不准备依赖。
weightc check examples/native_training/supervised.wgt
# 显式执行 main；也可以增加 --compiled 使用本机编译执行。
wgt run examples/native_training/supervised.wgt
# 直接以脚本路径调用等价于 wgt run。
wgt examples/native_training/supervised.wgt
```

`wgt run SOURCE --check` 检查脚本及依赖注释的语法，但不下载依赖、不执行，也不证明所有函数在运行时有效。`wgt run SOURCE --compiled` 编译后执行；`--max-operations N` 设置脚本操作预算。运行、编译及文件路径规则见[原生训练](native-training.md#同一脚本的解释执行与-rust-编译执行)。

`wgt new PATH` 创建新项目，`wgt init` 初始化现有目录；`wgt sync` 同步项目包依赖，`wgt add`、`wgt remove`、`wgt update` 管理依赖。`--offline` 禁止包网络访问，`--locked` 要求沿用现有锁；两者不能保证脚本本身没有网络操作。详见[包与脚本依赖](packages.md)。

`weightc compile SOURCE -o EXECUTABLE` 只编译，不执行训练或保存。`weightc legacy-build SOURCE` 显式执行返回模型的 `main`，构建模型目录且拒绝覆盖；它与训练脚本主动调用 `save` / `save_model` 的工作流不同。`wgt` 中仍列出的旧 `check/build/package/install/train/export/trace/debug` 子命令会报迁移错误，不能作为原生训练入口。原生 `wgt test` 用于运行项目测试，见[原生测试](native-testing.md)。完整工具链和发行契约见[工具链说明](weightc-migration.md)。
