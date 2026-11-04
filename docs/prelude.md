# 隐式预导入声明（WGT 0.5，本地未发布）

[`std/prelude.wgt`](../std/prelude.wgt) 是公开原生函数和张量 intrinsic 的权威声明源，便于查看参数、默认值、用途、适用上下文和原生实现位置。它是编译器专用的声明单元，不是可执行模型或可导入算子库。无需 import 即可使用当前上下文中的名称；它不会自动导入其他 std 模块。

非函数的语法、常量、类型、顺序层与内部 `__` lowering 保存在 [`language/prelude_metadata.json`](../language/prelude_metadata.json)。运行 `node scripts/generate_prelude.cjs` 将两者合并为 [`language/prelude.json`](../language/prelude.json)，供运行时注册与编辑器生成器读取；不直接编辑生成的 JSON。运行 `node scripts/generate_prelude.cjs --check` 验证同步。内部 lowering、已移除的兼容调用不出现在公开函数声明中。

## 受限原生声明

文件以 `prelude native 0.5;` 开始。普通函数使用 `native fn NAME(PARAMETERS);`；张量专用操作使用 `native intrinsic NAME(PARAMETERS);`。参数名是真实形式参数，可省略参数明确写 `= DEFAULT`；没有默认值的参数必须提供。描述性默认字符串（例如 linear 的 `bias = "absent (no bias)"`）表示缺省行为，不创建运行时字符串实参。所有必填参数须在可选参数之前，当前声明不支持可变参数。

每项前的 `# @contexts`、`# @purpose`、`# @implementation` 标注适用上下文、用途和原生绑定位置；可选 `# @documentation`、`# @position` 指向完整约束与位置限制。注解值使用 JSON 字符串或字符串数组，避免引号、换行等转义歧义；签名、默认值与函数 arity 直接从声明推导，不另写重复注解。同名普通函数与张量 intrinsic 分别声明。

这些声明没有 WGT 函数体，行为由 `implementation` 指向的原生 Rust 实现提供，不是空 WGT stub。用户源码不能定义原生函数：只有独立 prelude 声明读取器接受此格式，普通模型、脚本及算子解析器拒绝 `prelude native` 与 `native fn`；`std/prelude` 不是可导入标准模块。

`wgt check std/prelude.wgt` 和项目扫描识别声明单元并通过同一受限 Rust 读取器验证，不执行声明。构建阶段还校验完整公开集合、全部元数据、参数、默认值、arity 与生成 JSON 强一致，并校验其余元数据一致；不一致会中止构建并要求重新生成。运行时直接读取已验证的注册表，不重复解析声明。检查、编译和注册加载不执行函数、不读取数据、不初始化模型、不训练、不保存文件。

## 上下文与使用

`ordinary_function` 是普通脚本函数；`tensor_forward` / `tensor_loss` 是模型的可微张量函数。非函数元数据中的 `model_declaration` 和 `forward_parameter_declaration` 是声明位置，不能当作普通函数调用。普通列表函数与同名张量版本可能具有不同 dtype、形状和默认值；完整限制以所链接专题和实现为准。

```wgt
wgt 0.5
fn main() {
    # mean 的普通函数版本对非空有限数值列表归约，输出一个标量。
    return mean([1, 2, 3]);
}
```

该示例返回 2，不创建权重。公开目录中有状态或副作用的接口只在脚本显式执行时生效。完整目录与分派一致性检查为 `node scripts/check_prelude.cjs`，负例检查为 `node scripts/test_prelude.cjs`；编辑器一致性另运行 `node scripts/generate_editor_support.cjs --check`。运行时 `prelude_declarations` 测试覆盖格式拒绝、完整契约漂移和普通源码隔离。
