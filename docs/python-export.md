# Export standalone Python / 导出独立 Python 模型

`weightc export-python SOURCE -o NEW.py` generates readable PyTorch classes and
explicit `forward()` methods. The generated code never imports WeightLang.
This is implemented in the current source checkout, not a published release claim.

## 从 .wgt 导出结构

```sh
weightc export-python examples/cnn.wgt -o weights/cnn_python/model.py
```

```python
import torch
from model import Model

model = Model().eval()
result = model(torch.randn(2, 1, 4, 4))
```

把 Python 代码放在生成的 model.py 旁运行，或通过你的项目导入该模块。
结构导出只需要 PyTorch；**不执行源文件中的 train、import、inherit 或 init 策略**，
不加载自定义训练器。`Model()` 使用 PyTorch 新初始化的参数，不能当作已有训练权重。
生成文件头部和命令的 `mode: architecture-only` 都标明这一点。
要保留当前权重，导出已经构建好的模型目录。

## 导出已有模型和权重

```sh
# 先按原有构建流程训练；已有模型可以直接执行下一条。
weightc build examples/cnn.wgt --no-store
weightc export-python weights/tiny_cnn -o weights/cnn_export/model.py
```

输出两个文件：`model.py` 和 `model.safetensors`。权重按原文件复制，不转换精度，
需留出额外的权重文件空间。导出前用正常制品加载器验证模型，复制后再次验证权重哈希。
运行环境只需 PyTorch 和 safetensors；不需要源 .wgt、manifest 或 WeightLang 安装。

```python
import torch
from model import load_model

model = load_model()  # 默认从 model.py 所在目录加载 model.safetensors，CPU、eval 模式
prediction = model(torch.randn(2, 1, 4, 4))

# 可指定权重位置和设备；权重仍须匹配导出时记录的 SHA256。
# model = load_model("/path/to/model.safetensors", device="cuda")
```

加载器检查权重 SHA256 并严格核对 state_dict，恢复冻结参数。
若要继续训练，调用 `model.train()` 并自行创建优化器；不会恢复旧优化器或训练进度。
`Model()` 本身始终是新初始化结构；**保留权重和冻结设置应使用 load_model()**。

## Python API

```python
from weightlang import export_python

result = export_python("weights/tiny_cnn", "weights/cnn_export/model.py")
# {"python": ".../model.py", "weights": ".../model.safetensors", "mode": "artifact"}
```

两个参数都是路径；output 必须以 `.py` 结尾。源文件模式的 `weights` 为 None。
命令/API 均拒绝覆盖现有输出，包括旁边同名权重文件；没有隐式 force。
写入异常会清理本次创建的输出，保留原模型和预先存在的文件。

## 执行约定与范围

- 输入为 torch.Tensor，**始终显式提供 batch 维**；dtype 和单样本形状见生成文件的 INPUT，输出契约见 OUTPUT。
  不复制 WeightLang 的列表输入、自动加 batch 或输入诊断包装；按普通 PyTorch 模型使用。
- 支持当前内置全连接、激活、归一化、dropout、卷积/池化、embedding、位置编码、注意力、重排/归约，以及标准库展开后的网络。
- 支持具名分支、add/mul/concat/mix/where、残差、repeat、recur、scan。
  repeat 保持独立参数；recur/scan 保持共享参数；scan 每次调用从零状态开始。
- 保留权重路径和数值，不携带训练数据、训练器、优化器状态、设备分片、调试钩子或完整来源追踪系统。
  导出的模型默认放在单一设备；`.to(device)` 可整体移动它。
- 生成的代码可编辑。编辑结构后，原权重可能不再兼容；加载器不会忽略不匹配。
- 这是 Python 源码导出，不是 ONNX/TorchScript 导出，也不保证任意修改后的 Python 能反向转换为 `.wgt`。

本地验证比较相同参数下的前向结果与参数梯度，并在禁止导入 WeightLang 的独立进程中加载导出的模型。
CPU 上验证通过不等于所有设备、PyTorch 版本和精度下均逐位相同。
