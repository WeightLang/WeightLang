# wgt 命令行

> 架构设计入口：[Weight-Native 架构路线图](weight-native-roadmap.md)、[Weight ABI v1 草案](weight-abi.md)。两份文档是未来设计，不代表当前实现或发布承诺。
> M1：[基线盘点](tensor-contract-inventory.md)与[固定形状契约实现候选](tensor-contract.md)。

`wgt` 创建模型项目、管理依赖和工具链，并通过 uv 运行可选的 PyTorch 模板。`weightc` 提供原生源码检查、编译和兼容模型构建。当前命令的完整参数以 `wgt --help`、`wgt <命令> --help` 和 `weightc --help` 为准。

```sh
# 小型语言模型：创建、训练、读取已保存的模型生成文本。
wgt init text-model --template tiny-lm
wgt -C text-model train
wgt -C text-model infer --prompt "Hello"

# 原生 WGT：创建纯模型定义，检查时不准备依赖或执行模型。
wgt init native-model --template native
wgt -C native-model check
weightc check native-model/model.wgt
```

`wgt templates` 列出内置模板，`wgt init --interactive` 提供交互选择。Python 模板的架构、训练和推理参数位于标准 `pyproject.toml` 的 `[tool.wgt]` 中；`wgt check` 验证配置，`wgt train` 和 `wgt infer` 显式运行项目中的 Python host。`wgt run -- COMMAND ARGS` 通过 `uv run` 在项目环境中运行命令，例如 `wgt -C text-model run -- python -c "import torch; print(torch.__version__)"`。完整流程见[项目快速上手](project-quickstart.md)。

`wgt new PATH` 创建新目录，`wgt init` 也可初始化现有目录；两者都拒绝覆盖模板输出。`wgt sync` 同步 WGT 模型包及已配置的 uv 环境，`--no-env` 只同步模型包。`wgt add`、`wgt remove`、`wgt update` 管理模型包，Python 库使用 uv 原生命令管理。`wgt --offline --locked sync` 限制包网络访问并要求沿用已有锁；用户自己的程序仍自行决定是否联网。详见[包管理说明](packages.md#项目配置与-python-依赖)。

原生 WGT 模型源码入口必须命名为 `model.wgt`，自定义可视化样式为同目录的 `model.css`。
每个独立模型/训练示例一个目录，用目录名区分功能；例如 `examples/native_training/supervised/model.wgt`。
原生模板使用 `model.wgt`；PyTorch 模板使用 TOML 与 `workflow.py`，不会生成未参与计算的 WGT 文件。
CLI 对既有 `wgt.toml` 中显式配置的 `model.source` 保持兼容，但这不改变新建与维护模型的入口规范。

## 编译到 PyTorch

`wgt compile model.wgt --target torch` 复用原生前端，向 stdout 输出版本化架构 JSON；`-o architecture.json` 写入新文件并拒绝覆盖。`wgt compile - --target torch` 从 stdin 读取源码。编译不会执行 forward、训练或准备 Python 环境。

输出包含解析后的纯计算导入、接口、参数初值和计算图，可通过 [Python SDK](python-sdk.md) 的 `weightlang.from_ir` 或 `weightlang.load` 加载为 PyTorch 模型。目标接受无可变状态的直线计算图，未支持的状态、动态控制流与受约束 learn 参数会明确报错。

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

`weightc compile SOURCE -o EXECUTABLE` 只编译，不执行训练或保存。`weightc legacy-build SOURCE` 显式执行返回模型的 `main`，构建模型目录且拒绝覆盖；显式保存语句的工作流见[原生训练](native-training.md)。原生模型的应用入口使用 Rust API 和 `wgtime` 服务；`wgt train`、`wgt infer`、`wgt run` 使用已声明的 Python 项目环境。`wgt --list` 列出内置命令与 PATH 中的 `wgt-<name>` 扩展。完整工具链和发行契约见[工具链说明](weightc-migration.md)。
