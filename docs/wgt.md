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

`wgt templates` 列出内置模板，`wgt init --interactive` 提供交互选择。项目、模型配方、运行设备、训练和推理参数统一放在 `wgt.toml`；Python 模板还会生成只保存 Python/uv 设置的 `pyproject.toml`。Rust 应用的依赖继续放在 `Cargo.toml`。旧的 `[tool.wgt]` 配置仍可整份使用；同时提供两份 WGT 配置会报错，普通 `pyproject.toml` 可与 `wgt.toml` 共存。

`wgt check` 验证配置，`wgt train` 和 `wgt infer` 执行 `[python].runner` 指定的 host。`wgt run -- COMMAND ARGS` 通过 `uv run` 在项目环境中运行命令，例如 `wgt -C text-model run -- python -c "import torch; print(torch.__version__)"`。CLI 用 `WGT_CONFIG_FILE` 向 host 传入所选 WGT 配置的绝对路径；模板中的相对数据和输出路径以配置目录为基准。显式命令行参数覆盖文件中的对应参数，可选参数缺省时使用模板默认值。完整流程与旧 runner 的迁移方式见[项目快速上手](project-quickstart.md)。

Python 模板默认自动选择兼容的 PyTorch 构建与可用加速设备，无可用 GPU 时回退到 CPU。首次环境准备通过 uv 的后端探测记录标准包来源，后续依赖仍由 `uv.lock` 管理；`--device` 可覆盖运行设备，`wgt sync` 可重新选择自动管理的安装源。

`wgt new PATH` 创建新目录，`wgt init` 也可初始化现有目录；两者都拒绝覆盖模板输出。`wgt sync` 同步 WGT 模型包及已配置的 uv 环境，`--no-env` 只同步模型包。`wgt add`、`wgt remove`、`wgt update` 管理模型包，Python 库使用 uv 原生命令管理。`wgt --offline --locked sync` 限制包网络访问并要求沿用已有锁；用户自己的程序仍自行决定是否联网。详见[包管理说明](packages.md#项目配置与-python-依赖)。

原生 WGT 模型源码入口必须命名为 `model.wgt`，自定义可视化样式为同目录的 `model.css`。
每个独立模型/训练示例一个目录，用目录名区分功能；例如 `examples/native_training/supervised/model.wgt`。
原生模板使用 `model.wgt`；PyTorch 模板使用 `wgt.toml` 中的模型配方与 `workflow.py`，不会生成未参与计算的 WGT 文件。
CLI 对既有 `wgt.toml` 中显式配置的 `model.source` 保持兼容，但这不改变新建与维护模型的入口规范。

## 格式化源码

`wgt format`（别名 `wgt fmt`）递归规整项目内 `.wgt`，默认排除权重、构建、缓存、依赖和隐藏目录。`wgt format model.wgt` 选择显式文件，`wgt format --check` 只检查并在存在差异时返回非零；`wgt format -` 从 stdin 读取并输出格式化文本。它与 LSP 共用 token 保留式 formatter，保留注释、字符串、数字拼写和声明顺序，验证失败的源码不写入；不会初始化参数或升级历史语法。完整范围、退出码与排除规则见[源码格式化](format.md)。

## 编译到 Rust

`wgt compile model.wgt --target rust -o src/model.rs` 使用共享前端导出可被 Rust `include!` 的模型模块，公开 `SOURCE` 与 `definition()`。纯计算 imports 随输出嵌入；生成 Rust 应用运行时无需原始 WGT 文件。省略 `-o` 输出到 stdout，源文件 `-` 表示 stdin，已有输出文件拒绝覆盖；`--crate-path renamed_wgt` 可匹配 Cargo 中改名的 Rust SDK 依赖。

导出不启动服务、不初始化实例、不训练。模型实际执行通过 `wgt-model` SDK 和 `wgtime` 原生服务；当前支持三种模型计划和完整的 44 指令集合。生成的模块仍会解析嵌入架构，服务初始化仍需要本机 C 编译器。完整示例、覆盖清单与边界见 [Rust SDK](rust-sdk.md)。

## 编译到 PyTorch

`wgt compile model.wgt --target torch` 复用原生前端，向 stdout 输出版本化架构 JSON；`-o architecture.json` 写入新文件并拒绝覆盖。`wgt compile - --target torch` 从 stdin 读取源码。编译不会物化初始化配方、执行 forward、训练或准备 Python 环境。未绑定参数可编译，实例化时必须得到完整外部权重。

输出包含解析后的纯计算导入、接口、参数与状态的初始化配方、子模型身份及完整模型 IR，可通过 [Python SDK](python-sdk.md) 的 `weightlang.from_ir` 或 `weightlang.load` 加载为 PyTorch 模型。目标支持顺序图、forward 程序及 typed operations，保留控制流、learn 约束和状态事务；非法模型与未知 IR 会明确报错。

`from_ir` 明确执行声明配方；`weightlang.restore(definition, weights)` 从完整 `WeightSet` 直接绑定参数，跳过参数配方，状态单独初始化。合并和迁移使用宿主 `Plan`，没有隐式补值或独立的权重操作 CLI；详见[权重快照与迁移](weights.md)。

SDK 的加载接口使用显式传入的模型路径或架构数据以及设备参数，不隐式应用 `wgt.toml` 的训练、推理和设备设置。由 CLI 启动的 host 可读取 `WGT_CONFIG_FILE` 指向的文件；其他 host 显式选择配置文件，再将相应值传给 SDK 与 PyTorch。源码编译仍遵循编译器的导入和工具链查找规则。`tiny-lm` 和 `mlp` 模板仍通过自己的 TOML/Python 配方构造模型。

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
