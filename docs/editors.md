# 编辑器语法高亮

仓库提供声明式 `.wgt` 的离线高亮配置，不需要 Python、PyTorch 或运行模型。
颜色跟随编辑器主题。支持关键字、内置算子、类型、数值、双引号字符串和 `#` 注释；
TextMate 还区分参数、调用名称、标点和运算符。高亮不判断版本是否合法，也不做形状检查。
实验性的 `backend torch-dynamic` Python 风格函数体目前不在支持范围内。

## 支持范围

| 编辑器 | 文件/方式 | 状态 |
| --- | --- | --- |
| VS Code、VSCodium，以及兼容 VS Code 扩展的编辑器 | `editors/vscode/`，本地 VSIX | TextMate 引擎测试通过；0.1.0 已提交 Marketplace 验证 |
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
code --install-extension weightlang-0.1.0.vsix
```

也可以在编辑器中执行 **Extensions: Install from VSIX**，选择生成的文件。
VSCodium 用 `codium` 替代 `code`；其他兼容编辑器使用自己的安装入口。
扩展提供 `.wgt` 自动识别、注释切换、括号和双引号配对，无可执行扩展代码。
发布者为 `LIghtJUNction`，安装 ID 为 `LIghtJUNction.weightlang`。
0.1.0 已上传 Marketplace，正在等待平台验证；这不等于已经可以搜索或安装。
[Marketplace 页面](https://marketplace.visualstudio.com/items?itemName=LIghtJUNction.weightlang)
在验证完成后可用。Open VSX 尚未发布。

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
Zed 会从 manifest 固定的提交下载并编译 `editors/tree-sitter-wgt/` 中的解析器；
首次需要网络及 WASI 工具链，之后高亮可离线工作。打开 `.wgt` 文件后语言应显示 **WeightLang**。

扩展提供高亮、括号配对、注释切换、平铺声明大纲，以及括号内部的续行缩进。
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

不要分别修改生成文件。修改 `scripts/generate_editor_support.py`，然后运行：

```sh
python scripts/generate_editor_support.py
python scripts/generate_editor_support.py --check
python -m pytest tests/test_editor_support.py -q
npm ci --prefix editors/vscode
npm test --prefix editors/vscode
```

生成器只用 Python 标准库，以 AST 读取 `syntax.py` 的语句处理器、算子集合和类型；
不导入 WeightLang，不执行或初始化模型。命名轴简写和结构词维护在生成器中。
新增语法时检查词表提取是否覆盖新的声明方式；词表不是语法验证器。
TextMate 测试检查具体 token scope，并遍历 examples、packages 和标准库的 `.wgt` 文件。
Python 测试检查生成文件同步、JSON/plist 等价，以及已安装的 Vim/Neovim 实际行为。
未安装的 Vim/Neovim 测试会显示 skip。

Tree-sitter 的结构规则在 `editors/tree-sitter-wgt/grammar.js`，高亮查询复用词表：

```sh
python scripts/generate_tree_sitter_queries.py
python scripts/generate_tree_sitter_queries.py --check
(cd editors/tree-sitter-wgt && tree-sitter generate --abi 14 && tree-sitter test)
python -m pytest tests/test_tree_sitter_support.py -q
```

新增测试实际编译 C 解析器、遍历声明式模型、执行四种查询，并验证 CRLF/EOF、增量编辑和
不完整声明后的恢复。需要 Tree-sitter CLI（本地使用 0.26.9）和 C 编译器；缺少 CLI 时会 skip。
WASM 另用 `tree-sitter build --wasm` 验证；编译成功不等于已经在 Zed GUI 验收。
