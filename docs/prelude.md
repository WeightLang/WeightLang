# 内部内置目录（WGT 0.5，本地未发布）

[`language/prelude.json`](../language/prelude.json) 是原生函数、张量 intrinsic、语法、类型、常量、顺序层及内部 lowering 的统一内部目录。运行时注册、纯算子校验和编辑器词汇从这里读取元数据。它描述已有实现，不提供用户声明原生函数的入口，也不会自动导入 `std/` 模块。

Rust 应用负责 I/O、调度和训练循环；模型内部计算库使用显式 `import`。现有普通脚本运行时继续使用目录中的 `ordinary_function` 项。普通函数与同名张量 intrinsic 分别登记，它们的 dtype、形状、参数和副作用可以不同。完整模型边界见[语言参考](reference.zh-CN.md)和[模型内部标准库](standard-library.md)。

## 字段与消费者

| 字段 | 含义与使用位置 |
| --- | --- |
| `name`、`kind` | 项的名称与类别；两者组成唯一键 |
| `contexts` | 有效上下文，例如普通函数、张量 forward/loss 或模型声明 |
| `signature`、`defaults` | 完整签名及可省略参数；没有默认值的参数必须提供 |
| `purpose`、`implementation`、`documentation` | 用途、原生实现位置及可选专题文档 |
| `arity` | 普通函数及内部宿主入口的最少、最多实参数量 |
| `operator_arity` | 张量 intrinsic 在纯 helper 中的最少、最多实参数量；省略表示不能作为普通纯张量调用 |
| `reserved` | 可选的张量名称保留标记；保留名称仍不成为普通宿主函数 |

`operator_arity` 统一了纯算子的函数集合和参数数量检查。例如 `permute`、`batch_matmul` 分别要求两个参数，`layer_norm` 要求五个，`softmax` 接受一个或两个。形状、轴名、常量参数、状态访问和梯度规则仍由各原语的 lowering 与执行实现校验。`positive` 仅在张量条件位置使用；参数与工作内存读写保留各自的上下文限制。

## 修改与校验

直接修改内部目录和对应原生实现，然后生成编辑器资产：

```sh
node scripts/check_prelude.cjs
node scripts/test_prelude.cjs
node scripts/generate_editor_support.cjs
node scripts/generate_editor_support.cjs --check
cargo test --locked -p wgt-language
```

Rust 构建检查目录版本、唯一名称、上下文及 arity 结构。Node 检查进一步核对普通函数签名与默认值、原生 dispatch 集合、张量 lowering 集合及实现文件；负例检查确保遗漏、虚构、重复或损坏的记录被拒绝。编辑器生成不改动源目录；生成结果过期会由 `--check` 报错。

旧 `std/prelude.wgt` 声明源及其生成链已移除。`prelude::declarations::parse` / `validate` 保留为对显式提供文本和目录的惰性兼容读取器；它们没有执行能力，也不参与当前目录生成。普通模型、脚本及算子解析器继续拒绝 `prelude native` 与 `native fn`，`std/prelude` 也不是可导入模块。

目录加载、静态检查和编辑器生成均不初始化模型、执行训练或保存权重。完整执行验收应按所改动的原语运行行为测试；仅静态检查通过不说明后端已执行模型。
