# wgt 命令行

> 架构设计入口：[Weight-Native 架构路线图](weight-native-roadmap.md)、[Weight ABI v1 草案](weight-abi.md)。两份文档是未来设计，不代表当前实现或发布承诺。
> M1：[基线盘点](tensor-contract-inventory.md)与[固定形状契约实现候选](tensor-contract.md)。

`wgt` 管理项目、依赖和工具链，并运行 WGT 0.5 脚本。`weightc` 提供脚本检查、编译和原生模型构建。当前命令的完整参数以 `wgt --help`、`wgt <命令> --help` 和 `weightc --help` 为准。

```sh
# 检查只解析，不训练、不构造模型，也不准备依赖。
weightc check examples/native_training/supervised/model.wgt
# 显式执行 main；也可以增加 --compiled 使用本机编译执行。
wgt run examples/native_training/supervised/model.wgt
# 直接以脚本路径调用等价于 wgt run。
wgt examples/native_training/supervised/model.wgt
```

`wgt run SOURCE --check` 检查脚本及依赖注释的语法，但不下载依赖、不执行，也不证明所有函数在运行时有效。`wgt run SOURCE --compiled` 编译后执行；`--max-operations N` 设置脚本操作预算。运行、编译及文件路径规则见[原生训练](native-training.md#同一脚本的解释执行与-rust-编译执行)。

`wgt new PATH` 创建新项目，`wgt init` 初始化现有目录；`wgt sync` 同步项目包依赖，`wgt add`、`wgt remove`、`wgt update` 管理依赖。`--offline` 禁止包网络访问，`--locked` 要求沿用现有锁；两者不能保证脚本本身没有网络操作。详见[包与脚本依赖](packages.md)。

模型源码入口必须命名为 `model.wgt`，自定义可视化样式为同目录的 `model.css`。
每个独立模型/训练示例一个目录，用目录名区分功能；例如 `examples/native_training/supervised/model.wgt`。
新项目模板已经使用 `model.wgt`；维护和迁移项目时也应对齐这一规范。
CLI 对既有 `wgt.toml` 中显式配置的 `model.source` 保持兼容，但这不改变新建与维护模型的入口规范。

## 架构可视化

`wgt graph ARTIFACT --open` 将已有模型制品生成离线 HTML 并打开；不执行模型脚本或训练。
HTML 样式依次选择：显式 `--style PATH`、制品目录内的 `model.css`、
当前项目（或 `-C` 指定目录）中模型源文件同目录的同名 CSS。
源文件取 `wgt.toml` 的 `model.source`，缺省为 `model.wgt`；仅当源文件存在时查找其 CSS。
都不存在时使用内置样式；已找到但无法读取的 CSS 会报错，不静默忽略。
JSON、SVG、Mermaid 等其他格式不自动读取 CSS；显式 `--style` 仅支持 HTML。
制品内部已有的 `source.wgt` 文件名保持不变；自定义 CSS 不自动加入发布包。

```sh
wgt graph weights/my-model --open
wgt graph weights/my-model --style presentation.css -o architecture.html
```

显式输出文件必须不存在。`model.css` 示例：

```css
weightlang-viewer { --selection: #985235; }
weightlang-viewer [data-op="self_attention"] { --node-fill: var(--io-fill); }
```

导出的 JSON 与 HTML 附带 `sources`：制品内的 `source.wgt`，以及它（递归）`use` 的内置 `std/...` 模块；
第三方包源码不内嵌。网页把紧贴声明上方的 `#` 注释和行尾注释挂到对应的模型、输入输出、block 和 layer 上：
节点卡片显示注释首行，点击节点的浮窗显示完整注释、所调用和所在 block 的注释，以及声明所在的源码行。

组件接口和样式变量见 [Viewer 文档](../web/viewer/README.md)。

`weightc compile SOURCE -o EXECUTABLE` 只编译，不执行训练或保存。`weightc legacy-build SOURCE` 显式执行返回模型的 `main`，构建模型目录且拒绝覆盖；它与训练脚本主动使用 `save model` / `save checkpoint` 语句的工作流不同。`wgt` 中仍列出的旧 `check/build/package/install/train/export/trace/debug` 子命令会报迁移错误，不能作为原生训练入口。原生 `wgt test` 用于运行项目测试，见[原生测试](native-testing.md)。完整工具链和发行契约见[工具链说明](weightc-migration.md)。
