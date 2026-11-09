# Python SDK：用 PyTorch 运行 WGT 架构

WeightLang 描述模型内部的参数、张量计算、分支、循环和状态。Python 负责数据、训练循环、推理调用与可视化。SDK 复用 Rust 前端，把模型加载为标准 `torch.nn.Module`：纯张量模型使用 FX 图，含动态控制或状态的模型按原生中间指令执行 PyTorch 张量计算。

当前转换覆盖顺序图、forward 程序和 typed operations 三种模型计划，以及原生指令目录中的全部 44 种指令。转换验收同时检查输出、梯度、初始化、参数共享、惰性执行和事务提交；实际行为由[原生差分测试](../python/tests/test_native_parity.py)与[编译覆盖测试](../crates/wgt/tests/torch_export.rs)约束。

## 安装

在当前源码仓库根目录执行：

```sh
cargo install --locked --force --path crates/wgt --bin wgt
uv venv
uv pip install -e './python[visualize]'
```

需要 Python 3.10+、PyTorch 2.5+。`visualize` 是可选依赖组，提供 torchinfo、NumPy 和 Graphviz 的 Python 包；仅加载、训练和导出 Mermaid/DOT 时可改为 `uv pip install -e ./python`。SVG/PNG 还需要系统 Graphviz 的 `dot` 程序，例如 Debian/Ubuntu 的 `graphviz` 软件包。

SDK 当前版本为 0.2.0，从源码安装，尚未由本次开发发布到 PyPI。编译器与 Python 包应来自同一源码版本，以获得完整指令、原生初始化值和子模型元数据。已有项目可通过 uv 安装 Git 仓库的 `python` 子目录，或安装上述本地目录。

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

该示例训练后保存到 `weights/sdk-mlp/model.pt`，重新加载并核对预测，同时输出 `weights/sdk-mlp/architecture.mmd`。保留的结果默认禁止覆盖；再次运行可从新的工作目录启动该脚本，或在自己的宿主程序中选择新路径、显式传入 `overwrite=True`。

## 训练架构本身

下面的模型把计算深度设为可学习整数。深度决定同一缩放计算重复几次，内部结构仍完整写在 WGT 中：

```wgt
wgt 0.5
model LearnedDepth
input x: f32[1]
output y: f32[1]
# 各轮共享 scale；depth 保存连续 shadow，前向读取为整数。
frozen scale: f32[1] = [2]
learn depth: int in 0..=3 = 1;
fn forward(x) {
    return repeat depth carrying h=x { h * scale };
}
fn loss(prediction, target) {
    return 0.5 * square(prediction-target).mean;
}
```

```python
model = wgt.load("model.wgt", device="cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
wgt.bind_optimizer(model, optimizer)

x, target = torch.tensor([1.0]), torch.tensor([4.0])
optimizer.zero_grad(set_to_none=True)
loss = model.loss(model(x), target)
loss.backward()
optimizer.step()
print(model.evaluate(x))  # tensor([4.])
```

`bind_optimizer(model, optimizer)` 为普通 PyTorch 优化器注册更新后钩子。每次 `step()` 后，连续 shadow 按 WGT 的闭区间、严格下界或 `[0,1]` 规则投影；整数使用半整数远离零的读取规则，布尔使用 `>= 0.5`，两者反向均使用恒等 STE。优化器的动量和调度仍由 PyTorch 管理。钩子同时支持包含 WGT 子模块的 Python 组合模型，返回的 handle 可调用 `remove()`。每次新建优化器或恢复训练时绑定一次；也可显式调用 `model.project_parameters_()`。

`model.loss(prediction, target, reduction="mean")` 执行 WGT 的 `fn loss`，保留对预测与 target 的梯度。每个样本先执行声明的损失，再按 `mean`、`sum` 或 `none` 汇总外部批次。未声明 `fn loss` 的模型使用宿主选择的 PyTorch 损失。

完整示例位于 [`python/examples/adaptive`](../python/examples/adaptive)，执行 `python python/examples/adaptive/train.py`。它会验证深度从 1 更新为 2、损失从 2 降为 0，并保存模型与控制流图。默认输出 `weights/sdk-adaptive/`；再次运行可用 `--output weights/another-run` 选择新目录。

## 控制流、工作内存与写入

forward 模型有三种调用方式：

| 调用 | 梯度与执行 | 存储提交 |
| --- | --- | --- |
| `model(x)` | 遵循 PyTorch grad mode；可参与普通反向传播 | 使用当前参数的独立快照，不提交 WGT `write` |
| `model.evaluate(x)` | 无梯度，启用原生推理短路 | 不提交写入 |
| `model.run(x)` | 无梯度，启用原生推理短路 | 完整成功后提交写入，失败保留原存储 |

`evaluate` 和 `run` 用于包含动态指令的 forward 模块。纯 FX 模型直接使用 `model(x)` 与 `torch.inference_mode()`。动态分支只执行选中的路径；普通循环按指令跳转运行。`gate`、`route` 和学习深度在需要梯度时保留候选分支，以计算门参数梯度；无梯度推理时可在端点跳过候选。`.eval()` 设置 PyTorch 模块模式，关闭梯度使用 `torch.no_grad()`、`torch.inference_mode()` 或上述显式推理方法。

`memory_alloc/read/write/free` 在一次调用内部管理可微工作内存。后续读取能观察到此前写入，反向传播沿各内存版本传递。工作内存随调用结束释放；跨调用保存的内容应使用参数写入或 typed 状态。已有 forward 快照可以在后续 `run()` 提交之后反向传播，其计算值保持不变。

动态模块的 `max_work` 默认 1,000,000，`memory_bytes` 默认 256 MiB；调用时也可传同名参数覆盖。工作量按原生指令成本计数，`resource_state()` 使用相同的逻辑 f32 存储计量。该内存预算约束模型寄存器、候选存储和工作内存，PyTorch 分配器与 autograd 的实际内存开销另由宿主管理。`last_execution` 提供成功调用的工作量与工作内存峰值。

## 多入口与持久状态

typed operations 将入口参数直接写在函数签名中。例如，一个模型可以定义两个输入、读取入口以及显式重置：

```wgt
wgt 0.5
model Counter
state count: f32[1] = [0]
fn advance(step: f32[1], extra: f32[1]) -> f32[1] {
    count <- count + step + extra;
    return count;
}
fn observe() -> f32[1] { return count; }
fn reset() { count <- [0]; }
```

```python
counter = wgt.load("model.wgt", device="cpu")
counter.advance(torch.tensor([1.0]), torch.tensor([2.0]))
print(counter.observe())  # tensor([3.])
counter.call("reset")    # void 返回 None
```

各入口共享同一份参数与状态。一次调用先计算候选，在所有校验通过后统一提交；非有限结果、非法参数候选、运行错误或预算耗尽均回滚。事务内的后续读取看到最新候选。有界和离散参数的更新在最终提交时校验，整数候选必须为整数、布尔候选必须为 0 或 1。`reset` 的重置范围由 WGT 函数体决定。

`model.call(name, *inputs, max_work=None)` 可访问所有入口。安全的入口名也可作为 Python 方法调用；与 `nn.Module` 已有方法冲突的名称通过 `call` 访问。WGT 声明了 typed `forward` 时，`model(*inputs)` 会调用它。`operation_schemas` 返回入口签名和读写 effects，`state_names` 给出状态到 buffer 名称的映射。

typed operations 遵循当前 WGT 的无自动微分合同：参数不参与 autograd，带梯度的输入会报错。需要梯度训练的模型使用普通 forward 与可选 loss 路径。typed 入口接收零个或多个固定宽度的浮点向量，void 返回 `None`；具体类型规则见[操作合同](model-operations.md)。

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
| `bind_optimizer(model, optimizer)` | 绑定 WGT learn 的更新后投影规则 |

加载函数的 `device` 默认为 `"auto"`，按 CUDA、XPU、MPS、CPU 的顺序选择可用设备。可传 `device="cpu"`、`"cuda:1"` 或 `torch.device` 显式指定。选择可用设备不进行性能基准，也不自动切换 PyTorch 安装版本。

源码模型初始化为 float32；`dtype=torch.float64` 等参数可在加载时转换。SDK 保存并恢复张量的原始精度，传入 `dtype` 才覆盖它。输入必须是与模型设备、精度一致的浮点 `torch.Tensor`，也可按标准 PyTorch 用法对模型调用 `.to(...)`。

普通 forward 的输入与输出是固定宽度向量。Python 可添加非空前导批次维：输入 `[..., input_width]` 得到 `[..., output_width]`。内部 reshape、permute、归约、attention 和按输入生成的卷积核保留样本独立性。动态分支可以为不同样本选择不同路径。批次 forward 共享同一份调用前参数快照；显式 `run` 按样本顺序处理候选，整批成功后提交。

`model.input_shape`、`model.output_shape` 给出单样本形状；`parameter_names` 将 WGT 参数名映射到 `state_dict` 键。子模型实例通过编译器提供的身份映射注册成 PyTorch 子模块：重复引用同一实例共享一个参数，不同实例独立。层级可通过 `named_modules()` 查看；执行使用前端展开后的指令，子模块容器承担架构与参数组织。

`compile`、`compile_source`、`load` 和 `loads` 可传 `compiler="/path/to/wgt"`、`timeout=30`。编译器选择依次为显式参数、`WGT_COMPILER`、PATH 上的 `wgt`。文件加载遵循模型所在项目的 `.wgt-version`，内联源码遵循调用目录的工具链选择。文件模型的相对纯计算模块导入按源文件位置解析，内联源码需使用标准模块或同文件 helper。

SDK 使用调用时传入的模型与设备参数，不隐式应用 `wgt.toml` 中的 `train`、`infer`、`runtime` 配置或 `WGT_CONFIG_FILE`。使用统一项目配置的 Python 宿主需读取相关值，再传给 SDK、优化器与推理调用；Python 依赖与 uv 设置继续保存在 `pyproject.toml`。

## 保存模型与续训

```python
wgt.save(model, "weights/demo/model.pt")
restored = wgt.load("weights/demo/model.pt", device="cpu").eval()
```

SDK 保存格式包含版本化架构和标准 `state_dict`，加载使用 `torch.load(weights_only=True)`。它可在未安装 WGT 编译器的 Python 环境恢复模型。typed 模型的持久状态作为 buffers 一并保存，加载后可继续调用入口。`torch.compile(model)` 的包装也可保存，保存的是原始 WGT 模块。该 SDK 模型包与原生仅含参数的权重制品有各自的格式与恢复合同。

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
# 如果恢复时新建了 optimizer，重新绑定 learn 规则。
wgt.bind_optimizer(model, optimizer)
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

graph = wgt.trace(model)                     # 对纯 FX 模型取得静态图
print(graph.code)                           # 查看生成的 PyTorch forward

# 动态或 typed 模型直接绘制 IR，不运行入口。
wgt.visualize(counter, "advance.svg", entry="advance")
wgt.visualize(model, "loss.svg", entry="loss")  # 需要 WGT fn loss
```

纯模型图使用 FX 节点与数据依赖；动态模型图使用原生 IR，显示分支、循环、推理短路和参数／状态读写。typed 图可用 `entry` 选择入口，默认包含所有入口；`entry="loss"` 绘制自定义损失。`trace()` 要求可捕获的静态 FX 图，动态模型直接使用 `visualize()`。

`summary` 复用 torchinfo；带输入的摘要在模型副本上执行，保留原模型参数、梯度、缓冲区、训练模式、输入和 PyTorch 随机数状态。torchinfo 的运算量是工具估算，函数式算子的成本统计受该工具支持范围约束。

`visualize` 不传路径时返回 Mermaid/DOT/SVG 文本或 PNG 字节；传路径时返回 `Path`。输出目录须存在，覆盖需要 `overwrite=True`。已有 FX 模型的可视化无需执行 forward。普通 Python `nn.Module` 也可通过 `trace` 接入 FX 的符号追踪规则。

## PyTorch 生态接口

模型遵循 [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) 的参数注册、`state_dict`、模块组合和设备转换接口。普通 forward 使用 PyTorch autograd，typed 模型遵循其无梯度合同。DataLoader、优化器和调度器继续使用 PyTorch；混合精度、分布式训练及部署格式按具体模型和设备单独验收。

```python
combined = torch.nn.Sequential(model, torch.nn.Sigmoid())
compiled = torch.compile(model, backend="eager", fullgraph=True)
example_input = torch.zeros(1, *model.input_shape, device=next(model.parameters()).device)
exported = torch.export.export(model, (example_input,))
```

上述捕获示例用于纯 FX 模型。验收中的 `torch.compile` 使用 eager 后端验证完整图捕获，`torch.export` 验证所测纯模型；动态程序保留普通 `nn.Module` 执行路径。转换执行、静态图捕获和具体加速后端分别验收。

## 支持范围

| WGT 结构 | SDK 行为 |
| --- | --- |
| 顺序 Linear、ReLU、Tanh、Sigmoid、Identity | 转成 PyTorch 模块 |
| 普通参数、冻结参数 | 注册为 `nn.Parameter`，冻结项禁用梯度 |
| `learn float`、有界／正数 float、int、bool | 保留 shadow、原生读取规则、STE 和更新后投影 |
| 可训练 `prob` | 保留 logit 参数和前端生成的一次 sigmoid |
| 算术、激活、exp/log、clip、slice、sum/mean | 转成张量计算；clip 保留端点零梯度规则 |
| reshape、transpose、permute、matmul、batch_matmul | 按静态 WGT 形状解释，并保留 Python 批次维 |
| softmax、log_softmax、按轴 softmax、LayerNorm | 使用 PyTorch 算子 |
| Conv1d/2d/3d、分组卷积、MaxPool2d、AvgPool2d | 使用 PyTorch 张量运算，支持共享或按输入生成的卷积参数 |
| 纯计算 helper 和文件/标准模块导入 | 复用 Rust 前端解析和展开 |
| `fn loss` | 通过 `model.loss()` 执行，并保留预测与 target 梯度 |
| 动态 if／循环、gate／route、学习深度 | 保留执行路径、梯度候选和无梯度短路 |
| 参数写入、工作内存、资源查询 | 快照前向、可微内存、显式事务提交与预算检查 |
| typed 多入口、持久状态 | 共享参数／buffers，多输入、void、校验与回滚 |
| 子模型数组、共享实例 | 保留实例身份、参数共享和注册层级 |

支持范围跟随当前 WGT 模型语法与 IR。前端仍检查 shape、类型、操作签名与模型组合规则；任意宿主脚本不会作为模型执行。未知指令、未知程序字段和非法模型会报错。指令覆盖测试直接对照原生目录，新增指令时必须同时补充转换与验收。

顺序模型使用原生 SplitMix64／Xavier 初始化，显式参数按编译结果加载，均不改变宿主随机数状态。数值转换保留 WGT 的有限值合同、指定的宽精度累积以及端点梯度规则。不同计算后端仍存在浮点舍入差异；差分测试明确设置前向／梯度容差，初始化与离散状态使用精确断言。为保留数值规则，部分算子采用更宽的累积精度；MPS 上需要 float64 的计算通过可微 CPU 临时计算返回原设备，会增加传输成本。

SDK 对计算节点的激活与反向梯度执行有限值校验，包括冻结参数和常量的内部梯度。原生顺序图的部分推理路径只校验最终输出，因此中间出现 Inf、随后被激活消去的调用可能在原生推理中返回有限值，而 SDK 会拒绝。这是已知的异常处理差异；当前转换不承诺所有异常路径或浮点结果逐位相同。

动态批处理按样本执行，部分算子为保留原生累加顺序和梯度规则使用显式张量循环。吞吐与内存开销需要按模型和设备实测；成功转换为 `nn.Module` 或 FX 图本身不代表已获得加速。

执行器会复用已完成的值校验，并在无梯度推理时省去反向诊断包装。本轮 CPU 微基准、测量波动和复现方法见[运行时性能记录](python-sdk-performance.md)。

原生 `wgtime` 服务和 Python SDK 各自拥有运行实例。SDK 直接执行 PyTorch 张量运算，不调用参考测试工具作为后端，也不向原生模型计算注入 Python 回调。

## 验证

```sh
cargo test --locked -p weightlang-wgt --test torch_export --test no_python_cli
cargo build --locked -p weightlang-wgt --bin wgt --example torch_reference
uv pip install -e './python[visualize,test]'
WGT_COMPILER="$PWD/target/debug/wgt" \
WGT_REFERENCE="$PWD/target/debug/examples/torch_reference" \
uv run --no-project python -m pytest python/tests -q
```

原生差分测试直接复用 Rust `AdaptiveSession`、`OperationSession` 和 `Trainer`，核对前向、输入／参数梯度、自定义损失、SGD 投影、写入和回滚。另有指令全集覆盖、批次隔离、结构共享、保存恢复、FX／导出捕获、IR 图渲染、非法输入和浮点边界检查。CI 在 Python 3.10、3.12 上执行 CPU 验证，并运行 MLP 与可学习深度示例；GPU 硬件与具体编译加速效果按目标设备另行验收。
