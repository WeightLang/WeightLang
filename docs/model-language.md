# 模型语言：接口、维度和状态

工具包 0.5.2 沿用 `wgt 0.5`，以兼容扩展收敛模型写法。WGT 描述模型结构、参数、张量计算和状态；Rust/Python 管理 I/O、调度、训练循环与制品。旧宿主脚本和顺序 `layer` 定义仍可由各自兼容入口读取，本补丁没有删除它们或改变旧模型的梯度规则。

## 构造时确定维度

`dim NAME = INTEGER` 声明带显式默认值的维度；`dim NAME` 要求宿主绑定。名称以大写 ASCII 字母开头，其余字符只能是大写字母、数字或下划线，值为 1..=16384。声明独占一行，位于版本头之后、任何函数/子模型块之外，不带分号；重复声明、未知绑定、缺项和越界均报错。默认值必须是十进制整数字面量，不接受表达式、负数或自动推断。

维度在检查模型之前绑定，可用于端口、参数、state、局部张量注解、`as [...]`、固定循环/实例数量及初始化表达式。它是整个源文件保留的常量名，不能再作为局部变量、参数、函数、实例或命名轴的名称。字符串、注释和导入路径不替换。文件中的子模型共享这些维度，独立导入的算子库不继承调用方维度。这不是每次 forward 动态改变形状，也不提供每个子模型实例的独立泛型参数。

```wgt
wgt 0.5
dim N = 2
model Projection
# 权重的两个轴分别为输出和输入特征；实例化时创建单位阵。
parameter weight: f32[N,N] = eye()
parameter bias: f32[N] = zeros()
fn forward(x: f32[N]) -> f32[N] {
    return linear(x, weight, bias);
}
# 沿特征轴得到一个标量，宿主显式选择此 loss 和训练循环。
fn loss(prediction: f32[N], target: f32[N]) -> f32[1] {
    return 0.5 * ((prediction - target) ** 2).mean;
}
```

完整源码：[signature_dimensions/model.wgt](../examples/native_training/signature_dimensions/model.wgt)。定义检查只保存初始化配方；绑定维度不会执行初始化、forward 或训练。编译后的定义和保存的模型已经固定尺寸，不能通过加载时再传维度改变已训练权重的 shape；扩维使用显式权重迁移计划。

```sh
wgt check examples/native_training/signature_dimensions/model.wgt --dim N=3
wgt compile examples/native_training/signature_dimensions/model.wgt --target torch --dim N=3
wgt compile examples/native_training/signature_dimensions/model.wgt --target rust --dim N=3
```

`--dim NAME=VALUE` 可重复，重复绑定同名维度报错。Rust 的 `ModelDefinition::parse_with_dimensions(source, &BTreeMap<String, usize>)` 和 `from_file_with_dimensions(path, &BTreeMap<String, usize>)` 使用相同规则；文件导入仍相对原文件解析。默认 `parse` / `from_file` 使用源码默认值。Rust 宏可检查带默认值的模型；需要外部绑定的模板通过显式构造 API 使用。

Python 的 `compile` / `compile_source` / `load` / `loads` 接受可选关键字 `dimensions: dict[str, int] | None = None`，例如 `weightlang.load("model.wgt", dimensions={"N": 3}, device="cpu")`。SDK 将绑定传给 Rust 编译器，没有第二套 WGT parser。仅 WGT 源码可绑定维度；对 JSON/SDK checkpoint 传非空维度会报错，`from_ir` 不接受维度覆盖。`wgt format` / LSP 静态检查需要可解析的默认尺寸；无默认尺寸的模板须先通过显式绑定检查。

## 签名与可微 loss

新的可训练写法将端口直接写在 `forward` / `loss` 签名中，不再重复写根级 `input/output/target`。声明 `loss` 是提供可微训练目标，不会自动选优化器或运行训练。首期合同：

- `fn forward(x: f32[N]) -> f32[M]`，一个输入、一个输出，宽度各为 1..=16384；需要矩阵时在函数内显式 `as [...]`。
- `fn loss(prediction: f32[M], target: f32[T]) -> f32[1]`；预测宽度必须等于 forward 输出，target 可不同。兼容无注解的 `fn loss(prediction,target)`，此时 target 宽度默认等于输出。
- 不混用根级端口、`layer` 或持久 `state`。typed forward 保持纯计算，不允许 `write`/`<-`，需传递状态时通过纯 helper 返回下一状态。
- 其他 helper 仍为无类型注解的纯张量函数；不允许额外 typed 操作入口混入训练模型。
- 无 `loss` 的现有 typed 操作仍保持原有无 AD 合同；本补丁不改变其调用 API。旧根级端口 forward 继续兼容，包括其已有梯度/写入语义。

Rust 原生服务与 Python SDK 都使用相同的定义与 loss。Python `.loss(prediction,target)` 保留梯度；Rust 显式使用 `Loss::Custom`。原生解释、compiled-forward、真实 IPC 和 Python 的行为分别测试，不能从语法通过推断 GPU 加速或跨状态求导。

## 参数、状态和共享

`parameter` 是模型参数，`frozen` 保留现有不可更新合同；没有初值或 `external` 表示等待完整绑定，不默认随机/置零。`state` 是 typed 操作的持久状态，参数权重快照不包含它。模型定义、初始化实例和训练后权重仍然分开。

读取参数直接写名字；修改参数/状态推荐 `name <- value;`，目标必须为裸存储名且 shape 完全匹配。矩阵参数必须显式恢复矩阵形状，不能只凭相同元素数更新。旧 `read` / `write` 仅保留兼容，后者原有扁平写入语义没有被偷偷改变。

纯 helper 可返回 2..16 项扁平张量元组，通过 `let (output,next_state)=helper(input,state);` 显式传递状态；元组不是外部服务端口，不支持嵌套。宿主负责跨调用状态调度；模型内需要持久更新时使用 typed 操作的 `state` 与 `<-`。

`blocks: Cell[3];` 创建三个独立参数实例；多次调用 `blocks[0]` 共享同一实例。复制一个中间张量不会创建参数别名或新的可训练参数。共享合同无需额外的 `share` 关键字。

## 推荐写法与兼容边界

新代码优先使用 `as [...]`、`@`、`.T`、直接参数读取和 `<-`。加权计算优先使用现有 `gate/route` 与纯 helper；不再扩张 `mix/blend` 同义语法。旧调用形式、顺序 layer、prob 参数与 learn 域均继续兼容，本补丁没有合并它们的存储格式，也未改变 `if` 的旧 STE 行为。

后续破坏性收敛需要单独的语言版本与迁移：统一所有无 loss/有 loss 的入口能力、多输入多输出可微端口、独立的梯度策略、布尔 mask/多轴切片以及完全退役旧宿主脚本。本补丁没有以文档描述冒充这些能力已经完成。
