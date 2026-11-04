# 编辑器语法高亮

仓库提供 WGT 0.5 `.wgt` 的离线高亮配置，不需要 Python、PyTorch 或运行模型。
颜色跟随编辑器主题。支持关键字、内置算子、类型、数值、双引号字符串和 `#` 注释；
支持 `fn`、控制流、训练与自更新前向内置函数、布尔/null 字面量、JSON 字符串转义及花括号。
TextMate 还区分函数定义、参数、调用名称、标点和运算符。高亮不判断版本是否合法，也不做形状检查。
词表以当前 Rust 实现为准；旧模型材料不代表 0.5 可执行功能。
当前工作区的词表包含标量数学、计算预算和设备队列函数；这些本地更改不代表已安装的 0.2.0 扩展已更新。

## 支持范围

| 编辑器 | 文件/方式 | 状态 |
| --- | --- | --- |
| VS Code、VSCodium，以及兼容 VS Code 扩展的编辑器 | `editors/vscode/`，本地 VSIX | TextMate 引擎测试通过；当前扩展版本 0.2.0 |
| Sublime Text | `editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage` | 格式已验证，未逐一 GUI 验收 |
| TextMate | `editors/WeightLang.tmbundle/` | 同上 |
| JetBrains IDE（启用 TextMate Bundles 插件） | 导入 `editors/WeightLang.tmbundle/` | 同上 |
| Vim、Neovim | `editors/vim/` | 两者无界面实际高亮测试通过 |
| Zed | `editors/zed/`，独立 Tree-sitter 扩展 | 原生解析/查询测试和 WASM 编译通过；未 GUI 验收、未上架 |
| Emacs | `editors/emacs/wgt-mode.el` | 已提供 mode；未在 Emacs 执行验证 |
| Kate、KWrite | `editors/kate/wgt.xml` | XML 检查通过；未 GUI 验收 |
| GtkSourceView 编辑器，如 GNOME Text Editor | `editors/gtksourceview/wgt.lang` | GtkSourceView 5 实际加载/高亮测试通过 |

VS Code 使用 [TextMate grammar](https://code.visualstudio.com/api/language-extensions/syntax-highlight-guide)；
[Sublime 支持 tmLanguage](https://www.sublimetext.com/docs/syntax.html)，
[JetBrains 支持导入 TextMate bundle](https://www.jetbrains.com/help/idea/textmate.html)。
因此同一套规则能覆盖多种编辑器，但这不代表所有编辑器都能直接读取相同配置。

## 安装

以下路径相对仓库根目录。安装后重新打开 `.wgt` 文件；必要时重启编辑器。
本项目不会自动改写你的编辑器配置，也不会覆盖已有插件。

### VS Code / VSCodium / 兼容扩展的编辑器

```sh
cd editors/vscode
npx @vscode/vsce package --no-dependencies
code --install-extension weightlang-0.2.0.vsix
```

也可以在编辑器中执行 **Extensions: Install from VSIX**，选择生成的文件。
VSCodium 用 `codium` 替代 `code`；其他兼容编辑器使用自己的安装入口。
扩展提供 `.wgt` 自动识别、注释切换、括号和双引号配对，无可执行扩展代码。
发布者为 `LIghtJUNction`，安装 ID 为 `LIghtJUNction.weightlang`。
Marketplace 的可安装版本以发布页的验证状态为准；本地 VSIX 可直接安装。
[Marketplace 页面](https://marketplace.visualstudio.com/items?itemName=LIghtJUNction.weightlang)
提供公开版本。Open VSX 尚未发布。

### Sublime Text / TextMate / JetBrains

- Sublime：菜单 **Preferences → Browse Packages**，创建 `WeightLang` 目录，复制
  `editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage` 到其中。
- TextMate：打开 `editors/WeightLang.tmbundle` 安装 bundle。
- JetBrains：确认 TextMate Bundles 插件已启用，在 **Settings → Editor → TextMate Bundles**
  添加 `editors/WeightLang.tmbundle` 文件夹。此操作只增加词法高亮。

### Vim / Neovim

在 vimrc 或 Neovim 的 `init.vim` 中加入（替换绝对路径）：

```vim
set runtimepath^=/absolute/path/to/WeightLang/editors/vim
filetype plugin on
syntax enable
```

Lua 配置对应：

```lua
vim.opt.runtimepath:prepend('/absolute/path/to/WeightLang/editors/vim')
vim.cmd('filetype plugin on')
vim.cmd('syntax enable')
```

也可以把 `editors/vim/` 目录作为整个 Vim package 安装到 `pack/*/start/weightlang/`。
高亮采用 Vim 自带语法引擎，不依赖 Tree-sitter。

### Zed

在命令面板运行 **`zed: install dev extension`**，选择本仓库的 `editors/zed/`。
0.2.0 的 manifest 已固定到包含 WGT 0.5 解析器的远端提交。
Zed 会从 manifest 固定的提交下载并编译 `editors/tree-sitter-wgt/` 中的解析器；
首次需要网络及 WASI 工具链，之后高亮可离线工作。打开 `.wgt` 文件后语言应显示 **WeightLang**。

扩展提供高亮、括号配对、注释切换、平铺声明大纲，函数大纲，以及括号和花括号内部的缩进。
尚无 LSP、按缩进划分的折叠或嵌套大纲；`wgt init 1` 算法脚本和
`backend torch-dynamic` 不在该 Tree-sitter 语法支持范围内。
插件尚未提交 Zed 扩展仓库；本地开发扩展安装不等于上架。
详见 [Zed 扩展说明](../editors/zed/README.md)。

### Emacs

```elisp
(add-to-list 'load-path "/absolute/path/to/WeightLang/editors/emacs")
(require 'wgt-mode)
```

自动关联 `.wgt`；支持 font-lock 和 `#` 注释操作。

### Kate / KWrite

将 `editors/kate/wgt.xml` 复制到用户数据目录中的
`org.kde.syntax-highlighting/syntax/`。Linux 通常是
`~/.local/share/org.kde.syntax-highlighting/syntax/`，可用
`qtpaths --paths GenericDataLocation` 查询。见 [KDE 官方说明](https://docs.kde.org/stable_kf6/en/kate/katepart/highlight.html)。

### GtkSourceView

将 `editors/gtksourceview/wgt.lang` 复制到
`~/.local/share/gtksourceview-5/language-specs/`；使用 GtkSourceView 4 的应用对应
`gtksourceview-4/language-specs/`。目录不存在时先创建。
Flatpak 应用可能需要使用其沙箱内的用户数据目录。
见 [GtkSourceView 官方说明](https://gnome.pages.gitlab.gnome.org/gtksourceview/gtksourceview5/lang-intro.html)。

## 如何继续扩大覆盖

- **分发**：后续发布 VSIX 到 Marketplace/Open VSX、提交编辑器的语言仓库，让用户不用手动复制。
- **Tree-sitter**：已实现于 `editors/tree-sitter-wgt/`，Zed 包复用相同查询。
  Helix、Neovim 可复用解析器，但仍需各自的语言注册；尚未提交这些编辑器的上游仓库。
- **LSP**：复用 WGT 解析器提供诊断、补全、定义跳转等。它是下一层语义工具，当前没有实现。
- **GitHub / 网页 / 文档**：需要分别集成 Linguist、Shiki、Pygments 等；仓库带了 grammar
  不会自动让 GitHub 识别新语言。此处尚未注册这些集成。

## 维护与验证

使用 Node 生成器，不依赖已移除的 Python 实现：

```sh
node scripts/generate_editor_support.cjs
node scripts/generate_editor_support.cjs --check
npm test --prefix editors/vscode
(cd editors/tree-sitter-wgt && tree-sitter generate --abi 14 && tree-sitter test)
node scripts/test_editor_support.cjs
```

生成器使用 native-builtins.json 中的原生函数快照；存在 Rust BUILTINS 源码时检查两者一致。
编辑器可独立发布，不依赖尚未合入的语言迁移；显式维护 0.5 声明、控制流和 forward 运算。
修改生成器及 scripts/editor-templates/，不要直接修改生成的词表、TextMate、Vim、Emacs、
Kate、GtkSourceView 或高亮查询。Tree-sitter 结构规则在 grammar.js；结构变更后提交生成的 C/JSON。
编辑器解析器允许部分旧语法以便阅读迁移材料，不替代 weightc check。
`operators NAME` 有独立声明节点，`target` 与 `operators` 均标记为关键字。
`.5`、`//` 等宽松词法规则仅用于编辑器结构恢复，不表示 Rust 接受这些源码表达式。
[上下文规格](../language/source_capabilities.json) 与[共同语料](../language/conformance.json)
分别声明 Rust 接受与编辑器结构接受的预期；语义负例无需在编辑器中产生 ERROR。
运行 `cargo test -p weightlang-runtime --test source_conformance` 验证 Rust 一侧，
`node scripts/test_source_conformance.cjs` 验证编辑器一侧（也包含在编辑器总检查中）。

测试覆盖具体 TextMate scope、仓库源码遍历、Tree-sitter 语料和四种查询。
新增 0.5 样例覆盖训练循环、自更新 forward、嵌套花括号、逻辑运算与字符串转义。
未逐一 GUI 验收；Open VSX 与 Zed 扩展仓库未发布。
