# Lean 公式桥接（本地实验，未发布）

这一版让同一个受限公式定义生成可执行 WGT 和 Lean 数学定义，支持单个线性公式及下文的行局部公式图。
不是通用张量表达式语言，也不支持导入任意 Lean 函数；不改变 `.wgt` 语法。

## 公式与接口

```python
from weightlang.lean import LinearFormula

formula = LinearFormula(3, 2, bias=True)
model = formula.module()  # 内存初始化；没有训练或模型文件输出
print(formula.wgt_source())
print(formula.lean_source())
record = formula.verify(lean="lean")  # 显式调用 Lean，不在 load/forward 中运行
assert formula.matches_record(record)
assert not LinearFormula(4, 2).matches_record(record)
```

`LinearFormula(in_features: int, out_features: int, bias: bool = True)` 为不可变的
封闭公式表示；宽度必须是正整数（不接受 bool），bias 必须为 bool。
运行时 dtype 为 f32，输入 `[rows,in]`，输出 `[rows,out]`，也支持单个 `[in]` 向量。
权重为 `[out,in]`，偏置为 `[out]`，所有行共享参数：

`Y[r,o] = sum_i X[r,i] * W[o,i] + b[o]`。

`bias=False` 时省略加偏置，不创建偏置参数。实际执行沿用已有 WGT linear，
不是另写一个 Python forward；默认 seed=0，初始化不改变调用方 RNG 状态。
Python 负责优化器、loss、数据与调度。矩阵权重如以数学常见的 `[in,out]`
布局提供，需要先转置；桥接不隐式转换调用方权重。

- `canonical() -> str`：含 schema、尺寸、bias、dtype、权重布局、归约和广播规则的稳定 JSON。
- `wgt_source() -> str`：当前 `wgt 0.3` 线性模型，无新语法。
- `lean_source() -> str`：对应尺寸的 Lean 定义及 `WeightLang.linear_rows` 定理。
- `module(*, device="cpu") -> nn.Module`：通过现有 WGT 构建流程内存初始化，eval 模式；可训练参数仍允许求梯度。不会启动 Lean。
- `verify(*, lean="lean", timeout=60) -> dict`：每次实际运行 Lean 4.19.0；每个子进程超时单位为秒。找不到工具、版本不符、编译错误、警告、缺失公理审计或非允许公理均失败，不返回成功记录。只在平台临时目录下的 `weightlang/` 创建小临时源码，结束后删除。
- `matches_record(record) -> bool`：检查记录的公式、生成源码、桥接实现摘要和预期 Lean 版本是否匹配。它是**身份匹配，不是证明验真或当前工具检测**；记录可以被编辑，需要新证据时调用 `verify()`。

## 证明覆盖范围

Lean 使用 `Fin in` / `Fin out` 保留特征轴尺寸，使用 `Sum A B` 表示两组行的拼接。
点积是有限索引列表上从零开始的左折叠。定理证明：

`linear(rows(x,y),w,b) = rows(linear(x,w,b),linear(y,w,b))`。

定理对任意带零、加法和乘法的标量类型成立，不需要结合律等代数假设，因而适用于
实数实例。第一版不引入 mathlib、不自行构造实数类型；它证明的是行局部计算结构。
Lean 对完整定义和证明进行检查，禁止 `sorry` 警告，并审计定理依赖，
只允许标准 `propext`、`Quot.sound`、`Classical.choice` 公理。

**没有证明**：WGT→Torch 映射正确性、GPU 内核、浮点舍入误差、梯度实现、训练效果。
Torch 的归约顺序也不保证与数学定义中的左折叠相同。实际前向、输入/参数梯度、
空行和行拼接由独立的数值测试检查，不把测试说成形式化证明。
生成器、WGT 编译器、Lean 编译器与运行环境仍在信任边界内。

验证记录绑定公式、生成的 WGT/Lean 源码、桥接实现、Lean 版本与可执行文件摘要，
并包含公理审计原文；没有签名，也不绑定已训练权重或其他 WGT 模型。
公式或生成器变化后，旧记录不再匹配。

## 本地运行

安装官方 Lean 4.19.0（可将独立发行版放在磁盘缓存中），不需要 Lake 或 mathlib。
普通 Python 使用不依赖 Lean；证明检查必须显式提供工具。

```sh
PYTHONPATH=src .venv/bin/python -m weightlang.cli check examples/lean/model.wgt --shapes
PYTHONPATH=src .venv/bin/python examples/lean/verify.py --lean /path/to/lean-4.19.0-linux/bin/lean
WEIGHTLANG_LEAN=/path/to/lean-4.19.0-linux/bin/lean .venv/bin/python -m pytest tests/test_lean.py -q
```

完整示例：[WGT](../examples/lean/model.wgt)、[Lean](../examples/lean/Linear.lean)、
[验证脚本](../examples/lean/verify.py)。没有 Lean 时，数值测试仍执行，Lean 集成测试
明确跳过；不能把跳过视为证明通过。行局部公式图见下节；任意张量公式与受限 Lean 公式导入尚未实现。

## 可组合的带类型公式图

`FormulaGraph(in_features: int, nodes: tuple[FormulaNode, ...])` 支持一个 f32 特征
向量输入和最后一个节点输出。它继承上述 `canonical`、`wgt_source`、`lean_source`、
`module`、`verify` 和 `matches_record` 接口；schema 为 `weightlang.formula-graph.v1`。

`FormulaNode(name: str, op: str, inputs: tuple[str, ...], out_features: int | None = None,
bias: bool = True)` 描述一个不可变节点：

| op | 输入数量 | 特征宽度与参数 |
| --- | --- | --- |
| `linear` | 1 | `out_features` 必须是正整数；bias=True 创建偏置，False 不创建 |
| `add` | 2 | 两个输入宽度必须一致；out_features 必须省略，bias 保持默认值 |
| `mul` | 2 | 同宽逐元素乘法，参数约束同 add；不是矩阵乘法 |

节点名匹配 `[a-z][a-z0-9_]*`，不能是 `input`，不能重复。输入引用只能指向
`input` 或先前节点。节点元组非空，所有节点必须参与最终输出；拒绝未知引用、
前向/循环引用、无关节点、隐式广播、不支持的算子和可变列表。
`widths() -> dict[str,int]` 返回包含 `input` 的已推导宽度副本。
这些是构图与生成源码的静态契约。运行时沿用 WGT/Torch 检查，没有额外输入包装器；
尤其纯 add/mul 图不能依靠 Torch 自动拒绝所有错误 dtype/宽度。调用方须遵守声明的 f32 与特征宽度。

```python
from weightlang.lean import FormulaGraph, FormulaNode

graph = FormulaGraph(3, (
    FormulaNode("up", "linear", ("input",), 4),
    FormulaNode("square", "mul", ("up", "up")),
    FormulaNode("down", "linear", ("square",), 3, False),
    FormulaNode("skip", "add", ("down", "input")),
))
model = graph.module()
record = graph.verify(lean="lean")
```

示例表达 `((x W_up^T + b_up)²) W_down^T + x`。平方复用同一个 up 结果，
不会复制参数；两个不同 linear 节点有独立参数，不支持两个投影节点显式共享权重。
生成的 WGT 节点按拓扑次序命名 `n0`、`n1` 等，权重路径例如 `n0.weight`。
用户节点名用于引用、诊断和摘要；重排图可能改变生成的权重路径。
没有 sigmoid 等激活，因此这里的乘法门控不保证 [0,1] 值域。

Lean 从同一节点列表生成带 `Fin width` 的各中间值、显式参数以及 `graph_rows`
定理。证明覆盖整个图的行局部性质；不是仅证明一个孤立 linear。
单样本/空 batch/普通 batch、前向、输入与参数梯度、重复引用、源码加载和制品往返
由测试检查。生成 `.wgt` 可用既有 `load` 和导出流程，不需要额外运行后端。
旧 `LinearFormula` API 保留，旧图和新图的验证记录不能混用。

此子集仍不含任意阶张量、跨行归约、transpose、矩阵输入乘法、softmax、mask、
attention 或 Lean→WGT 公式导入。图验证同样不证明浮点内核和梯度实现。

可运行的 [构图与验证脚本](../examples/lean/graph.py)、
[生成 WGT](../examples/lean/graph.wgt)、[生成 Lean](../examples/lean/Graph.lean)：

```sh
PYTHONPATH=src .venv/bin/python examples/lean/graph.py --lean /path/to/lean-4.19.0-linux/bin/lean
PYTHONPATH=src .venv/bin/python -m weightlang.cli check examples/lean/graph.wgt --shapes
```
