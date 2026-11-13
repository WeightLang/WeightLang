# 一份模型定义，多个执行后端

普通 WGT 模型描述参数、张量计算、子模块、控制流及显式状态；执行后端由宿主或编译命令选择。Rust 原生服务与 Python SDK 复用 Rust 前端及模型计划。使用双方已经实现的计算时，无需为 Python 重写模型。

```sh
wgt check examples/native_training/shared_recurrent/model.wgt --target rust
wgt check examples/native_training/shared_recurrent/model.wgt --target torch
wgt compile examples/native_training/shared_recurrent/model.wgt --target rust
wgt compile examples/native_training/shared_recurrent/model.wgt --target torch
```

`--target rust` 生成嵌入完整定义的 Rust 模块，计算仍由原生服务执行；`--target torch` 生成 SDK 的模型表示。定义检查和编译不初始化参数、不运行 forward、不创建服务实例。后端检查首先检查源码方言边界，再进行模型校验；它不是任意算子的执行能力证明。具体执行支持以[原生后端](native-backend.md)、[Python SDK](python-sdk.md)和行为测试为准。

## 从输入形状绑定维度

以下新增 API 尚未发布，需要包含本地改动的编译器。SDK 按显式 `compiler`、`WGT_COMPILER`、PATH 中的 `wgt` 顺序选择编译器，不自动选用 checkout 的 debug 二进制。

本地新增的 `ModelTemplate` 允许宿主在创建实例时提供逻辑输入形状，由共享 Rust 前端推导声明的命名维度。没有增加另一套 Python 解析器。示例为 [`input_shapes/model.wgt`](../examples/native_training/input_shapes/model.wgt)。

```sh
wgt check examples/native_training/input_shapes/model.wgt --target rust --input-shape x=7,3
wgt compile examples/native_training/input_shapes/model.wgt --target torch --input-shape x=7,3
wgt compile examples/native_training/input_shapes/model.wgt --target rust --input-shape x=7,3 --dim OUTPUTS=4
```

Rust 宿主（edition 2024，`wgt-model` 的库名为 `wgt`）：

```rust
use std::collections::BTreeMap;
use wgt::{Dimensions, ModelTemplate};

let template = ModelTemplate::from_file("examples/native_training/input_shapes/model.wgt")?;
let definition = template.bind_input_shapes(
    &BTreeMap::from([("x".to_owned(), vec![7, 3])]),
    &Dimensions::new(),
)?;
// 此处只得到具体定义；再通过 Client::initialize 显式创建服务实例。
```

Python 宿主：

```python
import torch
from weightlang import torch_api as wgt

model = wgt.load(
    "examples/native_training/input_shapes/model.wgt",
    input_shapes={"x": (7, 3)},
    device="cpu",
)
y = model(torch.ones(21))  # 原有向量 ABI；输出为 14 个元素。
```

`input_shapes` 是输入名到逻辑轴长度的映射；它不执行 reshape、不推测扁平长度的分解。首期要求单个带 `f32[...]` 签名的 forward，不接受子模型声明、根级端口、持久 state、顺序 layer 或额外 typed 操作；输入名、rank、固定轴和重复维度都必须一致。输入推导出的维度优先于声明默认值；与显式 `dimensions` / `--dim` 冲突则报错。仅用于输出或参数的维度必须有默认值或显式绑定。单轴为 1..16384，rank 为 1..8，具体定义继续受现有端口及存储限制约束。

相对导入仍以模型文件所在目录为基准，根模型的维度不注入导入库。保存后的定义与权重保持具体形状；`input_shapes` 不能覆盖编译或保存制品的形状。每次绑定产生独立定义，创建实例会有独立参数及优化器状态。

`bind_input_shapes` 本身只创建具体定义；需要单实例变长时使用新增的 `initialize_dynamic` / `load_dynamic`。该路径在固定参数合同下切换激活计划并保留优化器，见[动态模型](dynamic-models.md)。原生只保留一个活动计划，Python 缓存有明确上限，不通过多个服务实例模拟共享参数。

## 共享循环模型

[`shared_recurrent/model.wgt`](../examples/native_training/shared_recurrent/model.wgt) 使用 `std/recurrent.gru_cell` 表达两步 GRU。输入打包为 `[x0(2), x1(2), previous_hidden(2)]`，输出为最终 hidden `[2]`。两步显式复用四组参数；没有 `torch.*`、Python 宿主绑定或隐式持久状态。

`gru_cell(xr, xz, xn, hr, hz, hn, hidden)` 的七个参数都是完全同形的 `f32[H]`，H 为 1..16384。所有参数必填、无默认值、无广播；返回 `f32[H]`。前三项为输入投影及输入偏置，随后三项为 hidden 投影及 hidden 偏置，顺序都是 reset/update/new。reset 门乘在完整 hidden candidate 投影之后，包含其偏置，对齐 PyTorch GRUCell。此纯函数不创建参数、不保存状态，可微性继承共享张量原语。仅添加标准库公式，未增加 IR 指令；沿用 WGT 0.5，属于本地新增能力。

宿主负责序列调度、初始 hidden、数据和优化器。示例一次调用内的两步参与同一计算图；该 GRU 示例本身不自动提供跨调用训练；新增的 tuple forward 和显式有界序列训练见[动态模型](dynamic-models.md)。通用持久 state AD、数据决定的动态中间长度及完整 train/eval 模块语义仍需分别实现。

## PyTorch 专用兼容扩展

源码中的 `target torch` 仍保留，供直接依赖 Python/PyTorch 表达式的模型使用。它属于[兼容扩展](torch-models.md)，不代表共享模型语义；原生加载和 `--target rust` 明确拒绝并报告行号，不静默调用 Python。

新模型优先使用共享 WGT。既有 Torch 模型逐项迁移其计算、索引、状态及共享参数语义，每项都需要原生与 SDK 的前向、梯度及失败行为验证。删除 `target torch` 声明本身不能完成迁移。
