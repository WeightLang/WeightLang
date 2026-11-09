<p align="center">
  <img src="media/icon.png" width="96" height="96" alt="WeightLang logo">
</p>

# WeightLang for VS Code

Edit `.wgt` models with syntax highlighting, native tensor diagnostics and the
WeightLang language server. The extension connects to `wgt lsp` through standard
input/output; model checking does not initialize weights or run training.

![WeightLang syntax highlighting preview](media/preview.png)

The preview shows lexical highlighting. Shape hints and diagnostics appear when
the matching native language server is installed and running.

## Features

| Feature | What appears in the editor |
| --- | --- |
| Current syntax | Typed operations, named axes such as `f32[rows=2, columns=3]`, numeric reshapes `as [2,3]`, `.T`, tuple bindings, `<-` updates and `learn` domains |
| Function parameters | Signature help with the active argument, required argument counts and parameter name hints |
| Tensor shapes | Hover and inline hints for statically checked local values, including named axes |
| Model size | Model hover and an inline parameter count; trainable, frozen and state storage counts are distinguished |
| Diagnostics | Native syntax, argument-count, tensor geometry and model contract errors, including incompatible matrix products and reshapes |
| Completion | Visible declarations, applicable builtins and imported helper functions |
| Navigation | Go to Definition, document references, document highlights and conservative rename |
| Structure | Document outline, symbols from open documents and block folding |
| Semantic highlighting | Resolved declarations and references follow the editor's semantic color theme |
| Quick fixes | Add or update the supported `wgt 0.5` header |

TextMate highlighting, comment toggling and bracket pairing work without the
language server. The color theme controls both lexical and semantic colors.

## Get started

1. Install VS Code 1.82 or later and this extension.
2. Install a native WeightLang toolchain that includes the `lsp` subcommand.
3. Open a `.wgt` file in a trusted workspace. The extension starts `wgt lsp`.

For this source version, build and install the matching executable from the
repository root if your installed release does not yet include the server:

```sh
cargo install --path crates/wgt --bin wgt --locked
```

To install a locally built extension, use **Extensions: Install from VSIX**.
See [BUILD.md](BUILD.md) for the reproducible package commands. Building a VSIX
does not publish it to the Marketplace.

If `wgt` is outside your editor's `PATH`, set an executable path in VS Code:

```json
{
  "weightlang.server.path": "/absolute/path/to/wgt"
}
```

The path is passed directly to process creation; it is not a shell command.
The extension always supplies the `lsp` argument. Remote workspaces run the
server on the remote extension host, so the executable must be available there.

Use **WeightLang: Restart Language Server** after replacing the executable.
Changes to `weightlang.server.path` restart it automatically. To keep only
lexical highlighting, set `weightlang.server.enabled` to `false`.

## Using tensor hints

Hover over a local tensor to see its inferred shape and element count. Hover over
the model name to see parameter totals. A declaration such as
`parameter weight: f32[2,3]` describes six stored values; the current language
reads that storage as `f32[6]`, so `weight as [2,3]` supplies matrix geometry to a
calculation. The hover explains both representations.

Type `(` or `,` in a function call to see its signature and active argument.
VS Code's **Trigger Parameter Hints** command can reopen the popup. Inlay hints
show parameter names and inferred shapes when **Editor: Inlay Hints** is enabled.

Diagnostics appear in the Problems panel and as underlines. The native frontend
checks the current document text, including unsaved changes, without executing
forward passes or starting the model service. Changes to local `.wgt` dependencies
also refresh open documents, including changes made outside VS Code.

## Current boundaries

Inferred intermediate shapes and parameter totals are shown after native model
validation succeeds. An unfinished or invalid model can still provide lexical
completion, signatures, navigation and diagnostics, but its intermediate shapes
are not presented as verified results.

References and rename operate within the current document. Rename refuses
ambiguous or conflicting names. Definition navigation also resolves directly
imported local helpers; this is not a repository-wide refactoring service.
Workspace symbol search includes open documents.

The server follows the native compiler's supported fixed-shape contracts. It
does not infer arbitrary Python/PyTorch code, allocate model parameters, train a
model or check runtime input values. In Restricted Mode, highlighting stays
available and the language server is not started.

## Troubleshooting

Open **Output → WeightLang Language Server** for startup and protocol logs.
If startup reports an unknown `lsp` command, install a matching toolchain. If it
reports `ENOENT`, set `weightlang.server.path` or make `wgt` available in the
extension host's `PATH`, then restart the server.

## 中文说明

打开 `.wgt` 文件即可使用新语法高亮。安装包含 `wgt lsp` 的原生工具链后，
扩展会在可信工作区自动启动语言服务器，提供函数签名、当前实参、参数名称、
矩阵形状、模型参数量和原生编译诊断，并支持补全、定义跳转、当前文件引用与
重命名、大纲、折叠和语义高亮。

形状与参数量来自原生静态检查，不会初始化模型或运行训练。尚未通过检查的
代码不会显示未经验证的中间张量形状。Python/PyTorch 的动态运行时行为不在
这些静态提示的范围内。找不到工具链时，配置 `weightlang.server.path`；更新
工具链后执行 **WeightLang: Restart Language Server**。

[Language reference](https://github.com/WeightLang/WeightLang/blob/main/docs/reference.zh-CN.md)
· [Report an issue](https://github.com/WeightLang/WeightLang/issues)
· [Other editors](https://github.com/WeightLang/WeightLang/blob/main/docs/editors.md)
