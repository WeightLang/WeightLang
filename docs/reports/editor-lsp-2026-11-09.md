# WeightLang 编辑器与 LSP 验收记录

日期：2026-11-09。范围：更新现有编辑器高亮，实现原生 `wgt lsp`，接入 VS Code，并验证参数提示、张量形状、模型参数量及错误诊断。

## 实现结果

`wgt lsp` 使用标准输入/输出提供语言服务，复用 `wgt-language` 原生模型前端进行检查。分析接受编辑器当前文本和已打开依赖的未保存文本，不初始化权重、不执行模型、不启动服务或准备网络依赖。该命令使用当前可执行文件，不被项目旧工具链固定配置转发。

| 用户能力 | 实现与范围 |
| --- | --- |
| 当前语法高亮 | 数字 `as [...]`、命名轴类型注解、`.T` / `.mean` / `.probabilities`、元组、`<-`、`learn float/int/bool`；同步 TextMate、Vim、Emacs、Kate、GTK、Tree-sitter、Zed 资源 |
| 函数参数提示 | 签名、活动参数、实参名称提示；正确区分嵌套调用、形状数组的逗号；模型调用不混入宿主函数和 layer 声明的签名 |
| 张量形状 | 编译器提供 `forward`、`loss` 和 typed operations 的输入及局部绑定形状、命名轴、元素数；悬停和行内提示展示 |
| 模型参数量 | 统计存储参数元素；总量、可训练、冻结和状态分别解释。多轴参数悬停区分存储布局与扁平读取形状 |
| 原生错误 | 语法、参数数量、矩阵乘法、reshape 元素数、命名轴、返回值和状态更新合同；编辑修复后清空旧诊断 |
| 补全与导航 | 当前可见声明、适用的内置函数、标准库与直接本地导入；定义、当前文件引用、高亮、保守重命名 |
| 结构 | 文档大纲、已打开文档的符号搜索、块折叠、语义高亮 |
| 快速修复 | 根文件缺失或不支持的版本头；不会为导入文件的问题改写正确的根版本头 |
| VS Code | 自动启动 `wgt lsp`、可执行路径配置、开关、重启命令、依赖文件监听；客户端和依赖许可证完整打入 VSIX |

协议位置使用 UTF-16，支持中文、emoji、CRLF 及同批顺序增量修改。诊断带文档版本；重命名返回带版本的编辑。依赖缓冲区失去同步时停止提供其旧形状，不回退到磁盘旧内容。相对导入保持原生目录边界与大小限制，并在打开前拒绝非普通文件。

## 一并修复的实际问题

导入解析原先无法重新序列化带命名轴的张量注解与字符串轴参数，导致合法模型在导入标准库后失败。本次补齐已有语法的链接输出，并用“标准库导入、矩阵命名轴、按名称 softmax”的组合测试验证。

typed `<-` 更新的形状和冻结检查移到相应语句的 lowering 中，错误能够包含更新行以及期望、实际形状。最终 IR 检查继续保留。纯 helper 检查也附带原始语句行号。

词法索引按连续 `let a = ..., b = a;` 的实际顺序登记绑定，避免重命名漏掉后续初始化器中的引用。对已经通过检查的模型，候选重命名重新经过原生前端；仅修改 legacy `forward` 输入参数却破坏端口关联的操作会被拒绝。

## 验证结果

以下项目均已在本次工作区实际运行，不把跳过的检查计为通过。

| 检查 | 结果 |
| --- | --- |
| `cargo check --locked --workspace --all-targets` | 通过；保留仓库原有 unused-unsafe/dead-code 警告 |
| `weightlang-wgt --lib lsp::` | 15 项通过：索引、连续 let、遮蔽、签名、补全、导入定义、行内范围、UTF-16 等 |
| `weightlang-wgt --test lsp` | 11 项通过：真实 stdio 进程、生命周期、形状错误与清除、参数量、未保存依赖、版本化重命名、缓冲区同步保护等 |
| `wgt-language --test ide` | 12 项通过：形状/命名轴/统计、typed operations、标准库和相对导入、源码定位、设备和目录拒绝 |
| `wgt-language --lib` | 8 项通过，包括分析不物化参数与状态的验证 |
| 相关语言回归 | 26 项通过：model_definition 3、model_modules 3、operations_composition_validation 5、shape_syntax 8、tensor_source_contracts 7 |
| 相关原生运行时回归 | 25 项通过：operations 8、composition_lowering 8、local_learning 9 |
| VS Code 客户端 | 7 项通过：启动、信任、重启、错误清理、配置、退出、文件监听与释放 |
| TextMate | scope 断言通过，遍历 18 个仓库 WGT 文件 |
| Tree-sitter | 20 项 corpus、63 项共享 conformance，以及新旧 fixture/query 检查通过 |
| 生成资源与词汇 | generator `--check`、prelude 检查通过；183 项 prelude、41 项 tensor intrinsic |
| 安装包 | `npm run package` 成功；13 个归档条目，完整客户端 bundle 与第三方许可证存在 |
| 非普通文件导入 | 对真实 FIFO 发起 LSP 分析，5 秒超时约束内立即拒绝并正常退出，无阻塞 |
| 示例与文档 | 高亮 fixture 和使用指南完整模型通过 `wgt check`；指南错误变体被原生前端拒绝 |
| 差异检查 | `git diff --check` 通过 |

CI 已加入 LSP 单元与 stdio 集成测试、VS Code 客户端测试、VSIX 打包和安装包 artifact 上传。远端 Actions 的运行状态以对应提交的 GitHub 记录为准。

### 验证边界

本地没有完整的 VS Code Extension Host，因此未执行编辑器 UI 自动化；客户端生命周期使用行为测试，服务通过真实进程和 JSON-RPC 协议验证。完整多编辑器脚本在需要 `xmllint` 时因工具缺失停止；XML 另用 Python 解析验证，Vim/Neovim 实际高亮未在本地执行。CI 配置保留这些工具的安装与检查。

## 使用与交付

完整说明见[语言服务器指南](../lsp.md)。在仓库根目录安装当前工具：

```sh
cargo install --path crates/wgt --bin wgt --locked
wgt lsp --help
```

在 VS Code 中通过 **Extensions: Install from VSIX** 安装 `weightlang-0.2.0.vsix`，打开可信工作区的 `.wgt` 文件即可启动。可用 `weightlang.server.path` 指定 `wgt` 的绝对路径。安装包使用编辑器扩展版本 0.2.0，集成后的 Rust 工具链使用主分支版本 0.5.0，两者独立编号；此次生成安装包不代表 Marketplace 或新 Release 已发布。

本次安装包大小：492,258 字节。SHA-256：

```text
c9dc8a28f0c8aee5084d60eac116d6e9be2774475234fd9fdc7a401e20694add
```

代码通过 GitHub 分批提交。高亮提交为 `3e79b5ecb2f46ed9e1715fc1d1ffcc69eb8960bd`，编译器 IDE 分析提交为 `56b16672dcbe589f545a7ea043ad4db12e01c395`；服务器、客户端和 CI 的最终提交见本记录所在的 Git 历史。集成时保留同期 `main` 的安装脚本改动。

## 当前功能边界

中间形状和模型统计仅在模型静态验证成功后提供。尚未确定的多态 helper 输入不会被猜测为具体矩阵；错误文件不继续展示旧版的已确认形状。每个文件先报告原生前端返回的首个错误，修复后继续检查。

引用和重命名限定当前文档；导出的算子函数需要同步导入者时会拒绝重命名。工作区符号限定已打开文件。深层子模型错误保守定位到根模型声明，并保留错误原因。Zed 扩展已更新高亮，但尚未自动接入 `wgt lsp`。

语言服务遵守当前固定形状合同，不分析任意 Python/PyTorch 动态执行、运行时输入值或训练效果。单文件最多 1 MiB，同时最多跟踪 64 个打开文档。

实现依据：[共享分析入口](../../crates/wgt-language/src/ide.rs)、[LSP 服务](../../crates/wgt/src/lsp/)、[协议测试](../../crates/wgt/tests/lsp.rs)、[静态分析测试](../../crates/wgt-language/tests/ide.rs)、[客户端](../../editors/vscode/extension.cjs)、[CI](../../.github/workflows/ci.yml)、[LSP 3.17 规范](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)。
