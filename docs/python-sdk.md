# Python SDK：用 PyTorch 运行 WGT 架构

WeightLang 描述参数、张量形状和前向计算，Python 负责数据、训练、推理与可视化。SDK 将现有 Rust 前端编译出的计算图转成标准 `torch.fx.GraphModule`，它继承 `torch.nn.Module`；梯度、优化器和设备执行直接使用 PyTorch。

## 安装

在当前源码仓库根目录执行：

```sh
cargo install --locked --force --path crates/wgt --bin wgt
uv venv
uv pip install -e './python[visualize]'
```

需要 Python 3.10+、PyTorch 2.5+。`visualize` 是可选依赖组，提供 torchinfo、NumPy 和 Graphviz 的 Python 包；仅加载、训练和导出 Mermaid/DOT 时可改为 `uv pip install -e ./python`。SVG/PNG 还需要系统 Graphviz 的 `dot` 程序，例如 Debian/Ubuntu 的 `graphviz` 软件包。

SDK 当前版本为 0.1.0，随本次源码提交提供，尚未由本次开发发布到 PyPI。编译器必须包含 `wgt compile --target torch`；旧发行版需要从当前源码重新安装。已有 Python 项目可直接通过 uv 安装 Git 仓库的 `python` 子目录，或安装上述本地目录。

PyTorch 的 CPU/CUDA 等安装选择继续使用 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)与 [uv 的 PyTorch 集成](https://docs.astral.sh/uv/guides/integration/pytorch/)。SDK 使用已安装的 PyTorch，不下载模型、驱动或运行服务。

## 一个模型、一个普通训练循环

将架构保存为 `model.wgt`：

```wgt
wgt 0.5
model Regression
# 一个样本含一个特征；批次由 Python 提供。
input x: f32[1]
output y: f32[1]
# 八个学习特征映射到一个预测值，初始权重尚未训练。
layer hidden: linear(8)
layer activation: tanh
layer prediction: linear(1)
init seed=42
```

Python 直接加载架构，用 PyTorch 训练 `y = 2x + 1`：

```python
import torch
import weightlang as wgt

model = wgt.load("model.wgt")
device = next(model.parameters()).device
x = torch.linspace(-1, 1, 64, device=device).unsqueeze(-1)
target = 2 * x + 1
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

model.train()
for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    prediction = model(torch.tensor([[0.5]], device=device))
print(prediction)
```

`.wgt` 中的 `forward` 也能直接导入。例如下面的参数乘法将变成 PyTorch 张量运算，`gain` 自动注册为可训练参数：

```python
model = wgt.loads("""wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
""")
```

完整可运行示例位于 [`python/examples/mlp`](../python/examples/mlp)。安装后在仓库根目录执行：

```sh
uv run --no-project python python/examples/mlp/train.py
```

该示例训练后保存到 `weights/sdk-mlp/model.pt`，重新加载并核对预测，同时输出 `weights/sdk-mlp/architecture.mmd`。保留的结果默认禁止覆盖；再次运行请使用新的输出目录，或在自己的宿主程序中显式传入 `overwrite=True`。

## 加载、设备与批处理

| 接口 | 用途 |
| --- | --- |
| `load("model.wgt", ...)` | 通过 Rust 编译器加载源文件并初始化 PyTorch 参数 |
| `loads(source, ...)` | 从 Python 字符串加载完整 WGT 定义 |
| `compile("model.wgt", ...)` | 返回 JSON 可序列化的架构字典，不执行模型 |
| `compile_source(source, ...)` | 从字符串编译架构字典 |
| `from_ir(definition, ...)` | 从架构字典初始化模型，无需编译器 |
| `load("architecture.json", ...)` | 加载预编译架构，无需编译器 |
| `save(model, "weights/demo/model.pt", overwrite=False)` | 保存架构及当前 `state_dict` |
| `load("weights/demo/model.pt", ...)` | 恢复已保存模型，无需编译器 |
| `select_device("auto")` | 返回实际可用的 `torch.device` |

加载函数的 `device` 默认为 `"auto"`，按 CUDA、XPU、MPS、CPU 的顺序选择可用设备。可传 `device="cpu"`、`"cuda:1"` 或 `torch.device` 显式指定。选择可用设备不进行性能基准，也不自动切换 PyTorch 安装版本。

源码模型初始化为 float32；`dtype=torch.float64` 等参数可在加载时转换。SDK 保存并恢复张量的原始精度，传入 `dtype` 才覆盖它。输入必须是与模型设备、精度一致的浮点 `torch.Tensor`，也可按标准 PyTorch 用法对模型调用 `.to(...)`。

WGT 的输入和输出仍是单个固定长度向量。Python 可添加任意前导批次维：输入 `[..., input_width]` 得到 `[..., output_width]`。内部 reshape、permute、归约和 attention 保留批次独立性。`model.input_shape`、`model.output_shape` 给出单个样本的形状；`model.parameter_names` 将 WGT 参数名映射到实际 `state_dict` 键，避免参数名称与 PyTorch 保留属性冲突。

`compile`、`compile_source`、`load` 和 `loads` 可传 `compiler="/path/to/wgt"`、`timeout=30`。编译器选择依次为显式参数、`WGT_COMPILER`、PATH 上的 `wgt`。文件加载遵循模型所在项目的 `.wgt-version`，内联源码遵循调用目录的工具链选择。文件模型的相对纯计算模块导入按源文件位置解析，内联源码需使用标准模块或同文件 helper。

## 保存模型与续训

```python
wgt.save(model, "weights/demo/model.pt")
restored = wgt.load("weights/demo/model.pt", device="cpu").eval()
```

SDK 保存格式包含版本化架构和标准 `state_dict`，加载使用 `torch.load(weights_only=True)`。它可在未安装 WGT 编译器的 Python 环境恢复模型。`torch.compile(model)` 的包装也可保存，保存的是原始模型参数。

加载后默认处于 PyTorch 的训练模式，推理时显式调用 `.eval()`。源码加载会按声明初始化权重；加载 `.pt` 会恢复已保存权重。源码、初始化模型、训练权重与优化器状态有不同用途。

训练续跑时，直接使用 PyTorch 保存优化器和宿主进度：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}, "weights/demo/training.pt")

checkpoint = torch.load("weights/demo/training.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

这类宿主训练断点通过 `torch.load` 恢复；`wgt.load` 接受 SDK 自己保存的模型包。随机数、采样器和数据读取位置仍由宿主按需要保存。SDK 的 `save` 针对保留原有计算图的 WGT 模型；用 Python 替换层或组合更大的模型后，使用标准 PyTorch 的架构与 `state_dict` 管理。

## 查看结构和计算图

```python
print(wgt.summary(model))                    # 参数、模块统计，不执行 forward
print(wgt.summary(model, input_size=(16, 1))) # 显式运行副本，获取形状和估算
print(wgt.visualize(model))                  # Mermaid 文本
wgt.visualize(model, "architecture.svg")    # Graphviz 渲染
wgt.visualize(model, "architecture.png")
wgt.visualize(model, "architecture.dot")    # 无需 Graphviz 程序

graph = wgt.trace(model)                     # 标准 torch.fx.GraphModule
print(graph.code)                           # 查看生成的 PyTorch forward
```

计算图从实际 FX 节点和数据依赖生成，保留残差、共享参数和多分支的数据连线。`summary` 复用 torchinfo；带输入的摘要在模型副本上执行，保留原模型参数、梯度、缓冲区、训练模式、输入和 PyTorch 随机数状态。torchinfo 的运算量是工具估算，并非所有函数式算子的完整成本统计。

`visualize` 不传路径时返回 Mermaid/DOT/SVG 文本或 PNG 字节；传路径时返回 `Path`。输出目录须存在，覆盖需要 `overwrite=True`。已有 FX 模型的可视化无需执行 forward。普通 Python `nn.Module` 也可通过 `trace` 接入 FX 的符号追踪规则。

## PyTorch 生态接口

模型遵循 [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) 的标准行为：`parameters()`、`named_parameters()`、`state_dict()`、训练/评估模式、模块组合和张量反向传播可直接使用。DataLoader、优化器、调度器、混合精度、分布式训练等继续由宿主按 PyTorch 的接口组织；本次验收覆盖 CPU 前向、梯度和优化器更新。

```python
combined = torch.nn.Sequential(model, torch.nn.Sigmoid())
compiled = torch.compile(model, backend="eager", fullgraph=True)
example_input = torch.zeros(1, *model.input_shape, device=next(model.parameters()).device)
exported = torch.export.export(model, (example_input,))
```

验收中的 `torch.compile` 使用 eager 后端，验证完整图捕获；没有据此声称 Inductor 性能提升。`torch.export` 可导出所测模型。其他编译后端、部署格式和加速设备仍受相应 PyTorch 算子的支持范围约束。

## 支持范围

| WGT 结构 | SDK 行为 |
| --- | --- |
| 顺序 Linear、ReLU、Tanh、Sigmoid、Identity | 转成 PyTorch 模块 |
| 普通参数、冻结参数、无约束 `learn float` | 注册为 `nn.Parameter`，冻结项禁用梯度 |
| 可训练 `prob` | 保留 logit 参数和前端生成的一次 sigmoid |
| 算术、激活、exp/log、clip、slice、sum/mean | 转成张量计算；clip 保留端点零梯度规则 |
| reshape、transpose、permute、matmul、batch_matmul | 按静态 WGT 形状解释，并保留 Python 批次维 |
| softmax、log_softmax、按轴 softmax、LayerNorm | 使用 PyTorch 算子 |
| Conv1d/2d/3d、分组卷积、MaxPool2d、AvgPool2d | 使用 PyTorch functional 算子，共享卷积参数 |
| 纯计算 helper 和文件/标准模块导入 | 复用 Rust 前端解析和展开 |
| `fn loss` | 保留在编译元数据中；训练损失由 Python 显式选择 |

当前目标接受无可变状态的直线计算图。typed operations、持久状态、工作内存、参数写入、动态分支/跳转、带短路语义的 gate/route，以及需要优化器投影的有界或离散 `learn` 会明确拒绝。局部静态 shape 与单向量入口仍遵循现有编译器限制。

顺序模型使用声明 seed 对局部 PyTorch generator 做 Xavier 初始化，避免改变宿主随机数状态；其随机序列与 Rust 后端不承诺逐位一致。显式参数初值按编译结果加载。原生 `wgtime` 服务和 Python SDK 使用各自的执行后端，SDK 不向原生模型计算注入 Python 回调。

## 验证

```sh
cargo test --locked -p weightlang-wgt --test torch_export --test no_python_cli
cargo build --locked -p weightlang-wgt --bin wgt
uv pip install -e './python[visualize,test]'
WGT_COMPILER="$PWD/target/debug/wgt" uv run --no-project python -m pytest python/tests -q
```

测试涵盖真实源码编译、Python 模块组合、批次隔离、梯度与优化器更新、卷积和归一化对照、Transformer Encoder 前向及参数梯度对照、权重与精度保存恢复、无编译器加载、FX/导出捕获、图渲染和未支持语义的拒绝。CI 在 Python 3.10、3.12 上执行 CPU 验证。
