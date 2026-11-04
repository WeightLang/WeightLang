# 编辑器语法高亮

仓库提供 WGT 模型 DSL 的 `.wgt` 文件离线高亮配置。Rust 主程序区域使用 Rust 编辑器工具；WGT 文件由仓库现有的 WGT 语法配置支持。
编译期读取的 WGT 文件和运行期读取的 WGT 文件都仍是 WGT 源文件，编辑器按文件类型提供高亮，不据此判断其加载时机或执行语义。
颜色跟随编辑器主题。WGT 文件规则覆盖关键字、可公开调用的函数、Tensor intrinsic、类型、数值、双引号字符串和 `#` 注释；高亮不验证 DSL 语义、版本或形状。
Rust 的 `model!` 宏接收 WGT 源码 Rust string literal；其中内容不是 Rust token DSL。目前没有该源码字符串的 WGT 语法注入、高亮或补全支持，不能假设会自动继承 `.wgt` 文件的语法支持。`include_model!` 接收文件路径字符串，注入支持不适用于路径字符串。

## 支持范围

| 编辑器 | 文件/方式 | 状态 |
| --- | --- | --- |
| VS Code、VSCodium，以及兼容 VS Code 扩展的编辑器 | `editors/vscode/`，本地 VSIX | WGT 文件使用 TextMate；Rust 主程序使用编辑器的 Rust 支持；当前扩展版本 0.2.0 |
| Sublime Text | `editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage` | WGT 文件规则；格式验证状态以本轮检查结果为准 |
| TextMate | `editors/WeightLang.tmbundle/` | WGT 文件规则；未确认 Rust 宏内嵌注入 |
| JetBrains IDE（启用 TextMate Bundles 插件） | 导入 `editors/WeightLang.tmbundle/` | 仅增加 WGT 文件词法高亮 |
| Vim、Neovim | `editors/vim/` | WGT 文件使用仓库语法；Rust 主程序使用 Rust 支持 |
| Zed | `editors/zed/`，独立 Tree-sitter 扩展 | WGT 文件由 Tree-sitter grammar/query 支持；本地 corpus 13/13 与编辑器检查通过，覆盖顶层 `learn` 声明和 `depth_gate` 调用高亮；未 GUI 验收、未上架 |
| Emacs | `editors/emacs/wgt-mode.el` | 已提供 WGT mode；未在 Emacs 执行验证 |
| Kate、KWrite | `editors/kate/wgt.xml` | WGT 文件规则；未 GUI 验收 |
| GtkSourceView 编辑器，如 GNOME Text Editor | `editors/gtksourceview/wgt.lang` | WGT 文件规则；不提供 Rust 宏内嵌注入 |

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
首次需要网络及 WASI 工具链，之后可为 `.wgt` 文件提供高亮。打开 `.wgt` 文件后语言应显示 **WeightLang**。
该 grammar 只注册 WGT 文件类型；Rust 宏中的内嵌 DSL 注入尚未实现。

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
- **Rust 宏内嵌 WGT**：`model!` 和 `include_model!` 接收 Rust string literal，不是 token DSL。要为字符串内容提供 WGT 高亮，需要 Rust grammar 的 injection query、宏匹配规则和 WGT 注入语言注册；这尚未实现。宏内补全也未实现。
- **Tree-sitter**：解析器位于 `editors/tree-sitter-wgt/`，Zed 扩展复用其查询。最近一次本地检查中，`tree-sitter generate --abi 14` 及全部 13 项 corpus 检查通过；查询覆盖顶层 `learn` 声明与 `depth_gate` 调用高亮；Helix、Neovim 仍需各自语言注册。
- **LSP**：诊断、补全和定义跳转当前没有实现。
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

生成器从当前 prelude metadata 生成 WGT 文件高亮词表；这些规则服务于 WGT DSL 文件，不代表 Rust 宏字符串已启用注入。
修改生成器及 scripts/editor-templates/，不要直接修改生成的词表、TextMate、Vim、Emacs、
Kate、GtkSourceView 或高亮查询。Tree-sitter 结构规则在 grammar.js；结构变更后提交生成的 C/JSON。
编辑器解析器允许部分旧语法以便阅读迁移材料，不替代 weightc check。
`operators NAME` 有独立声明节点，`target` 与 `operators` 均标记为关键字。
`.5`、`//` 等宽松词法规则仅用于编辑器结构恢复，不表示 Rust 接受这些源码表达式。
[上下文规格](../language/source_capabilities.json) 与[共同语料](../language/conformance.json)
分别声明 Rust 接受与编辑器结构接受的预期；语义负例无需在编辑器中产生 ERROR。
运行 `cargo test -p wgtime-core --test source_conformance` 验证 Rust 一侧，
`node scripts/test_source_conformance.cjs` 验证编辑器一侧（也包含在编辑器总检查中）。

编辑器语法检查与 Rust 主程序、WGT DSL 文件应分别验收。最近一次本地编辑器检查通过：Tree-sitter corpus 13/13、共同语料 conformance 49 项通过，查询、XML 与 Vim 检查均通过；总检查解析了 12 个 WGT 示例，并覆盖顶层 `learn` 声明与 `depth_gate` 调用高亮。宏内嵌 WGT 注入、补全尚未验收或实现。未逐一 GUI 验收；Open VSX 与 Zed 扩展仓库未发布。
