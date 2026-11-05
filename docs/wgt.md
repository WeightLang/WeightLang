# wgt 命令行

`wgt` 管理项目、依赖和工具链，并运行 WGT 0.5 脚本。完整命令说明见本文其余章节。

> 架构设计入口：[Weight-Native 架构路线图](weight-native-roadmap.md)、[Weight ABI v1 草案](weight-abi.md)。两份文档是未来设计，不代表当前实现或发布承诺。

`wgt` 管理项目、依赖和工具链，并运行 WGT 0.5 脚本。原有命令和模型源码入口规则保持不变。当前命令的完整参数以 `wgt --help`、`wgt <命令> --help` 和 `weightc --help` 为准。

```sh
# 检查只解析，不训练、不构造模型，也不准备依赖。
weightc check examples/native_training/supervised/model.wgt
# 显式执行 main；也可以增加 --compiled 使用本机编译执行。
wgt run examples/native_training/supervised/model.wgt
# 直接以脚本路径调用等价于 wgt run。
wgt examples/native_training/supervised/model.wgt
```

`wgt run SOURCE --check` 检查脚本及依赖注释的语法，但不下载依赖、不执行，也不证明所有函数在运行时有效。运行、编译及文件路径规则见[原生训练](native-training.md)。

`wgt new PATH` 创建新项目，`wgt init` 初始化现有目录；`wgt sync` 同步项目包依赖，`wgt add`、`wgt remove`、`wgt update` 管理依赖。详见[包与脚本依赖](packages.md)。

模型源码入口必须命名为 `model.wgt`，每个独立模型/训练示例一个目录。CLI 对既有 `wgt.toml` 中显式配置的 `model.source` 保持兼容。

## 架构可视化

`wgt graph ARTIFACT --open` 将已有模型制品生成离线 HTML 并打开；不执行模型脚本或训练。组件接口和样式变量见 [Viewer 文档](../web/viewer/README.md)。

`weightc compile SOURCE -o EXECUTABLE` 只编译，不执行训练或保存。原生构建、测试和迁移边界见[工具链说明](weightc-migration.md)与[原生测试](native-testing.md)。
