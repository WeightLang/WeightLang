---
name: sync-wgt-examples
description: 当 WeightLang/WGT 语法、解析器、版本、模型原语、标准库签名或张量形状写法发生变化时，自动同步整个 WeightLang 仓库的现有示例、测试夹具、文档代码块与编辑器片段，并验证旧语法没有遗留。也用于“更新语法后同步示例”“迁移旧 WGT 语法”“修复过时示例”的请求。
---

# 同步 WeightLang 语法示例

一次语法变更必须覆盖完整示例清单。以实际前端、原语合同和运行后端为准，保留与旧版本兼容性的专用测试。

## 1. 确定变更合同

1. 阅读仓库 `AGENTS.md` 和 `.agents/skills/wgt-development/SKILL.md`。
2. 对比此次变动前后的 `crates/wgt-language/src/`、`language/primitive_specs.json`、`language/prelude.json`、`std/`、执行后端及对应测试，列出新增、删除、废弃和仍兼容的形式。
3. 把“解析器接受”与“原生 CPU 服务、Rust SDK、Python SDK 执行支持”分别确认；不得凭编辑器高亮或提案文档推断功能可用。

## 2. 全仓搜集并逐一迁移

执行以下清单；每次按实际语法变更补充搜索表达式。

- 用 `rg --files -g '*.wgt'` 枚举完整源码：`examples/`、`packages/`、`python/examples/`、`std/`、`language/syntax-tour/model.wgt`、`crates/**/tests/fixtures/`、`crates/wgt/templates/` 和 `editors/`。
- 搜索 README、`docs/` 所有 WGT fenced code、教程中的注释代码、编辑器展示片段，以及 Rust/Python 测试中的内嵌 WGT 字符串。使用 `rg -n` 展开所有旧写法并逐项处理。
- 以最新推荐形式重写所有正向示例，保持结果、数值顺序、shape、轴名、梯度、共享参数、状态写入和控制流语义不变；必要时更新说明形状推导的注释。
- 将独立模型放在 `<示例目录>/model.wgt`；同步移动后引用、工作路径和链接。不要将标准库模块或制品 `source.wgt` 当成模型入口改名。
- 仅在**明确验证兼容或拒绝历史语法**的回归测试里保留旧形式，并让测试名称与注释表明用途。不得留下任何未分类的旧示例。
- 同步 `docs/reference.zh-CN.md`、相关专题文档、`language/syntax-tour/model.wgt`、本仓库的 `wgt-development` 技能与编辑器语法资产。

## 3. 防回退与验证

1. 更新 `scripts/check_example_syntax.cjs` 中的当前禁止示例写法：新语法替代旧语法后，检查器也要演进，覆盖源码与 Markdown 代码块。运行 `node scripts/check_example_syntax.cjs` 和 `node --test scripts/test_example_syntax.cjs`。
2. 运行 `cargo test --locked -p wgt-language --test example_sources`，确保仓库所有真实 `model.wgt` 被共享原生前端解析并 lowering；根据变更再运行相关 `wgt-language`、`wgtime-core`、`wgt-model` 与 Python SDK 回归。
3. 涉及编辑器时运行 `node scripts/generate_editor_support.cjs --check`、`node scripts/test_editor_support.cjs`；涉及原语表时运行 `node scripts/check_prelude.cjs`。
4. 对 WGT code fences、README、Rust/Python 内嵌字符串再做一次 `rg -n` 全局搜索，检查误改和漏改；运行 `git diff --check`。本地缺依赖时保留验证记录并检查远程 CI。
5. 按用户要求分批提交、同步到有权限的远程功能分支并提供 PR 或提交链接。不要绕过受保护分支或访问权限。

## 验收标准

所有活跃示例以当前推荐语法呈现；兼容/负例有明确归类；模型解析、示例防回退、受影响的数值或 SDK 行为检查通过；文档引用指向真实文件。未运行的检查逐项说明。
