# 共享模型的变长执行与显式状态求导

本页描述本地实现，不是已发布版本的功能承诺。模型仍使用普通 WGT；Rust 原生服务与 Python SDK 使用同一 Rust 前端和张量 IR。`target torch` 继续保留为 Python/PyTorch 专用兼容路径。

## 同一实例切换输入形状

Rust 的 `Client::initialize_dynamic(template, input_shapes, dimensions, options)` 创建一个服务实例。`DynamicModel::forward(input, input_shapes)` 和 `specialize(input_shapes)` 重新绑定激活形状，保留服务句柄、参数值和优化器矩；不重新运行初始化配方，不创建另一组模型权重。

Python 的 `load_dynamic(path, *, input_shapes, dimensions=None, device="auto", dtype=None, compiler=None, timeout=30, max_specializations=8)` 与 `loads_dynamic(source, ...)` 提供同样的源码形状绑定入口。宿主创建一次标准优化器，形状变化时继续使用相同的 `torch.nn.Parameter`。

```python
import torch
import weightlang as wgt

model = wgt.load_dynamic(
    "examples/native_training/input_shapes/model.wgt",
    input_shapes={"x": (2, 3)}, device="cpu",
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for rows in (2, 7, 3, 2):
    optimizer.zero_grad()
    prediction = model(torch.ones(rows, 3))
    prediction.square().mean().backward()
    optimizer.step()
```

输入采用显式逻辑形状，不能把任意长度的向量自动分解为矩阵。普通单输入模型的结果继续展平；tuple 模型按其输出合同返回各张量。单轴 1..16384，rank 1..8，打包后的输入、输出分别最多 16384 项。参数的名称、声明顺序、shape、冻结标记、learn 域和初始化配方必须保持一致。参与参数 shape 的维度不能在一个实例中改变。

原生侧只保留一个活动计划，准备候选计划时先检查当前与候选的瞬时内存，并纳入编译 workspace。准备、编译或 schema 校验失败，不替换旧计划，不修改参数或优化器。成功切换计划与随后一次 forward/train 是两个明确操作；执行失败不意味着回退一次已经成功的形状切换。

Python 保留参数唯一的 base 模块，额外形状采用有界 LRU；`max_specializations` 包含 base，序列化 IR 总预算为 16 MiB。缓存不是持久状态。train/eval、dtype/device 转换及参数对象重载使候选缓存失效。并发调用或宿主修改需由宿主同步。

`ModelTemplate` 在第一次成功绑定时冻结导入源码闭包；根源码在创建模板时已捕获。`source_fingerprint()` 是根源码、文件导入和实际内置标准库源码的 SHA-256，与具体输入形状无关。CLI 的形状绑定导出携带 `template_fingerprint`。Python 候选图必须与首个定义指纹相同，避免把同形权重误用于另一份已编辑的模型。

动态 Python 模块的 `state_dict` 是参数检查点；恢复时先用相同源码创建动态模块，再加载参数，并在之后创建优化器。当前 `weightlang.save(dynamic_model, ...)` 明确拒绝，以免把当前具体形状误存为完整动态模型。显式 `save(dynamic_model.base, ...)` 只导出 base 的静态模型。优化器检查点仍由宿主单独保存。

## 可微多输入与 tuple 输出

普通 WGT 的 tuple-return forward 是显式的纯 Forward 路径：

```wgt
wgt 0.5
model ExplicitState
parameter gain: f32[2] = [0.5,0.5]
fn forward(x: f32[2], hidden: f32[2]) -> (f32[2], f32[2]) {
    let next_hidden = x + hidden * gain;
    return (next_hidden, next_hidden);
}
```

`forward` 的全部形参必须有 `f32[...]` 合同；返回为非嵌套 tensor tuple，各路径的 arity 和逐轴形状必须一致。输出位置是稳定身份，不从局部变量猜测公开输出名。共享前端生成输入名、shape、offset、width 和对应输出 schema，再在入口拆包、出口打包为既有单向量 IR；参数不参与打包，执行器不增加第二套模型语义。

将上面的完整模型保存为 `explicit_hidden/model.wgt` 后，可用以下宿主连接两次调用；权重只注册一次，`hidden` 不存入模型状态：

```python
import torch
import weightlang as wgt

model = wgt.load("explicit_hidden/model.wgt", device="cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x1 = torch.tensor([0.2, 0.3], requires_grad=True)
x2 = torch.tensor([0.4, 0.5], requires_grad=True)
hidden = torch.zeros(2, requires_grad=True)
_, hidden1 = model(x1, hidden)
prediction, hidden2 = model(hidden=hidden1, x=x2)
loss = prediction.square().mean()  # 不监督的输出分量不提供 seed。
loss.backward()  # 经过 hidden1 传播到 x1、初始 hidden 和同一组权重。
optimizer.step()
```

`forward(*inputs)` 按源码声明顺序传入，或用 `forward(**named_inputs)` 提供全部输入；不能混用两种方式、遗漏或增加名字。每个张量须精确匹配逻辑 shape，不隐式增加 batch 轴，dtype/device 必须匹配 base。即使仅有一个 tuple 分量，结果也保持 tuple。`parameter_names` 使用 `base.` 前缀；`state_dict` 不重复登记参数。`save(model, path)` / `load(path)` 保留端口包装和键布局；`model.definition` 与 `model.base.definition` 指向同一完整导出。打包向量的研究接口通过 `model.base` 使用，不自动接受逻辑端口参数。

Python 模型接受对应逻辑张量，返回 tuple；宿主可直接用其中一项计算 loss。flatten、concat、拆分和 reshape 保留 autograd 链。没有被监督的输出 seed 为零，同一张量重复返回时梯度累加。Rust `Model::forward_ports(&[...])` 接收各端口的行优先数据，按 schema 返回输出分量；原有 `forward` 仍提供底层打包接口。

旧的一维 typed operation 继续保持其无 AD 合同，不因这项扩展获得可微 state 写入。当前 tuple forward 不接受源级 `fn loss`、持久 state 或参数写入；Python 使用宿主 loss，原生使用显式输出 seeds 或下文的有界序列训练。输入目前都是 f32，不表示任意 int64/bool 子图已经支持。

### 多输入动态形状与 hidden 连接

将下面的完整模型保存为 `dynamic_hidden/model.wgt`。`ROWS` 只控制激活行数；权重始终为 `[2,2]`，hidden 始终为 `[1,2]`，可直接连接不同 `ROWS` 的调用。这些参数有显式初值，示例尚未训练。

```wgt
wgt 0.5
dim ROWS = 2
model DynamicHidden
parameter weight: f32[2,2] = [0.2,-0.1,0.3,0.4]
fn forward(x: f32[ROWS,2], hidden: f32[1,2]) -> (f32[ROWS,2], f32[1,2]) {
    let y = x @ (weight as [2,2]);
    # Sum all rows/features into f32[1], then send it to both hidden channels.
    let contribution = scale([1,1], sum(y)) as [1,2];
    let next_hidden = hidden + contribution;
    return (y, next_hidden);
}
```

```python
import torch
import weightlang as wgt

model = wgt.load_dynamic(
    "dynamic_hidden/model.wgt",
    input_shapes={"hidden": (1, 2), "x": (2, 2)},
    device="cpu", max_specializations=4,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x1 = torch.ones(2, 2, requires_grad=True)
x2 = torch.ones(3, 2, requires_grad=True)
hidden0 = torch.zeros(1, 2, requires_grad=True)
optimizer.zero_grad()
_, hidden1 = model(x1, hidden0)
prediction, hidden2 = model(hidden=hidden1, x=x2)
(prediction.square().mean() + hidden2.square().mean()).backward()
optimizer.step()
```

`input_shapes` 必须是非空字典并覆盖所有根输入；名字与维度的一致性由 Rust 前端检查，不在 Python 解析源码。字典插入顺序不决定调用顺序，位置参数始终按 `forward_interface.inputs` 的声明顺序。每次调用直接读取各张量的逻辑 shape，并把所有输入的形状一起作为特化缓存键；调用者不另传形状字典。输入总宽度和输出总宽度分别不超过 16384。

新形状经 Rust 编译后，先比较 `template_fingerprint` 和完整参数元数据，再构建不运行初始化配方的候选图，并绑定到 base 的原始 `Parameter` 对象。源码或导入指纹变化会提示 `reload required`；参数元数据比较除物化 `data` 外的所有字段，包含初始化配方。失败不会改变已有缓存、参数或优化器。主机显式保留 `hidden1` 的 autograd 链，才产生跨两次调用的梯度；缓存切换本身不会保留隐式 hidden。普通单输入非 tuple API 继续返回展平张量。

## 跨调用 hidden 的原生反向传播

`AdaptiveSession::recurrent_gradients(inputs, targets, StateLink, OutputSlice, remaining, tape_budget)` 保留一段有限序列的 tape，并沿显式 hidden 连接反向传播。

- `StateLink { input_offset, output_offset, width }` 把本次输出分量送到下一次输入。只有第一步所给的 hidden 是自由输入；后续输入中的 hidden 占位内容被覆盖，其对外梯度为零。
- `OutputSlice { offset, width }` 明确指定监督哪些输出，避免把 hidden 自动当成预测目标。
- 每一步 target 为 `Some(vector)` 或 `None`；至少监督一步。MSE 对被监督的特征与时间步取平均。
- `SequenceReport` 含 loss、各步输出、输入梯度和共享参数总梯度；求导本身不更新参数，也不提交持久写入。

`train_recurrent(..., rate, clip, ...)` 对整段序列执行一次 SGD/Adam 更新，保留既有优化器历史。Rust facade 的 `Model::train_recurrent` 和服务 IPC 接通同一实现；当前只接受明确的 MSE 监督和显式 tensor hidden，不支持 workspace memory_* 操作。一次序列为 1..1024 步，输入合同在该序列内固定，跨序列可通过动态实例切换形状。原生预算覆盖整段保留的 primals/adjoints、输出、输入梯度和聚合梯度；更新另预留优化器副本与原子参数替换空间，实际可用量还受实例 memory_bytes 扣除常驻存储和优化器状态后的余额约束，不把每步预算重复使用。与现有训练一致，work 计费指 forward 指令预算，不代表进程内存或所有反向浮点运算的硬上限。

这不是隐式持久状态或跨网络请求保留 tape。宿主显式提供整段输入、初始 hidden、监督位置和目标；失败不会提交参数或优化器更新。Python 可以直接连接多次 tuple forward 的 hidden，由标准 autograd 处理同样的链。

## 运行时索引与张量拼接

| 操作 | 合同 | 梯度 |
| --- | --- | --- |
| `gather(input, indices, axis=0)` | input rank 1..8，indices 为一维，axis 为编译期非负轴；结果把该轴替换为 indices 长度 | 对 input scatter-add，重复地址累加；对 indices 不求导 |
| `dynamic_slice(input, start, length)` | input 为一维，start 为运行时 `f32[1]`，length 为编译期正整数；严格检查边界，不裁剪 | 只回传选中 input 区间；对 start 不求导 |
| `concat(a, b, axis=0)` | rank 相同，非拼接轴严格一致；axis 为编译期非负轴 | 分别拆回两输入，别名梯度累加 |

所有参数除明确的 axis 默认值外均必填。索引值必须有限、非负、精确整数且小于对应轴长度；不截断、不取模、不接受负索引。这些索引是受限的 f32 地址向量，不能据此宣称通用整数张量执行。它们使用 WGT 0.5、Adaptive IR v3 的共享合同。

原生当前为这些新索引指令使用解释执行和原生 tape；`enable_compiled` 选择该明确回退，并报告实际 backend，不能称为新指令已生成 C kernel。Rust 原生并不依赖 Python。Python eager/FX 通过相同 IR 实现这些操作，形状、索引与梯度分别验收。

`embedding(table, indices[, padding_idx])` 是 rank-2 `f32[rows,width]` 表的行索引，indices 为 rank-1 运行时 f32 地址向量，结果为 `f32[count,width]`。地址必须有限、精确非负整数且小于 rows，不截断、不取模，对 indices 不求导。省略 padding_idx 表示 `None`（不屏蔽）；显式 padding_idx 必须为 `[-rows,rows)` 内的静态有符号整数字面量，负值加 rows 归一化，因此 `-1` 指最后一行。前向保留该行原值（包括非零 padding 行）；VJP 对重复非 padding 行 scatter-add，并抑制 padding 行梯度。命名轴保留 table 的特征轴名，以 indices 的轴名替换词表轴名；结果名称重复时拒绝。它使用 WGT 0.5、Adaptive IR v3 的共享前端与原生 tape；CPU compiled 请求明确回退解释器并报告实际 backend，不依赖 Python。Python program/FX 消费同一 IR，省略 padding 不同于显式 `-1`。
