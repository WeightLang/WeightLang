# 自定义和混合权重类型

本地 `wgt 0.3` 支持有限实数取值集合（码本）。二值、三值、非均匀、
非对称、多级整数或浮点类型都使用相同机制。此功能尚未发布。

```wgt
wgt 0.3
model mixed_weights
wtype ternary = [-1, 0, 1]
wtype asymmetric = [-2, -0.25, 0.5, 3]
input x: f32[4]
output y: f32[6]
layer projection: linear(in=4,out=6)
# weight 是 [输出通道, 输入通道]。前两行三值化，中间两行使用自定义码本。
# 最后两行和偏置保持浮点；结构已初始化，尚未训练。
weight projection.weight: ternary axis=0 start=0 stop=2
weight projection.weight: asymmetric axis=0 start=2 stop=4
init seed=7
```

## 完整声明与约束

- `wtype NAME = [VALUE, ...]`：模型文件顶层声明。至少一个有限、可表示为
  f32 的数（允许常量表达式和显式 `+1`）；可以只有一个值，也可以不含零。内部按数值排序。
  类型名不可重复，转换为 f32 后重复的值也拒绝（含正负零）。
- `weight PARAMETER_PATH: TYPE`：整块参数使用指定类型，例如
  `weight projection.weight: ternary`。参数路径必须实际存在，类型必须已声明
  （可在使用语句后声明，最终统一检查）。没有隐式全局默认类型。
- `weight PARAMETER_PATH: TYPE axis=A start=S stop=E`：仅约束沿第 A 轴的
  `[S,E)` 切片，其余轴全选。三个选项必须同时提供；轴从 0 开始，不含 batch。
  `A/S/E` 为整数常量表达式，满足 `0 <= A < rank`、`0 <= S < E <= shape[A]`。
  同一参数的多个切片必须沿同一轴且不重叠；未覆盖元素保持浮点。
  不支持跨轴交叠规则、隐式优先级或负索引。

目标可以是层内任何现有浮点参数，包括 bias、卷积、归一化、Embedding 和注意力
投影。嵌套 block 使用展开后的参数路径。用 Python 的 `model.named_parameters()`
查看准确路径。此声明不改变张量的输入输出 dtype、轴或形状契约，不新增标准库算子。
库文件目前不能声明权重类型；在使用库的模型文件中指定展开后的参数。

同一模型可以有任意多个类型；同一参数可以混合多个类型和浮点区域。
`share` 或 `tie_embedding` 的所有别名自动使用同一约束、同一个主参数；
不同别名上的规则也要满足不重叠要求。共享参数不产生独立可训练副本。

## 前向、梯度与保存

前向把被约束的每个元素投影到最近的码本值，等距时选择较小值。
码本固定、不训练、没有自动缩放因子。反向采用恒等直通估计（STE），即对浮点
主参数的局部导数设为 1；这是训练用的梯度近似，不是离散投影的真实导数。
即便码本只有一个值，主参数仍使用 STE。冻结的参数不会因此恢复梯度。

Python 优化器、`named_parameters()`、`get_parameter()`、层的 `.weight` 和
`state_dict()` 看到的都是浮点主权重。只有模型的前向调用使用投影值；直接调用
子层不属于完整模型前向，不应用根模型的类型规则。浮点主权重可以超出码本范围，
每次前向会重新投影，因此优化器更新不需要额外 clamp 或回调。

```python
import torch
import weightlang

model = weightlang.load("examples/custom_weight_types.wgt")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(8, 4)  # batch=8，每个样本四个输入特征
prediction = model(x)["y"]
loss = prediction.square().mean()  # 仅演示训练连接；真实任务由 Python 提供数据和损失
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

使用类型声明的模型采用具名接口，输出为 `{输出名: Tensor}`；数据准备、损失、
训练循环、采样仍由 Python 负责，不接受 WGT `data/train`。`load` 不训练、不写制品。
类型和切片写入制品格式 v7 的 `interface.weight_types/weight_assignments`，
`safetensors` 保留 f32 主权重和原参数路径。重载、原生 Rust 校验、结构反编译和
独立 Python 导出保留声明；旧格式不允许夹带此元数据。

目前不提供位打包、整数计算内核、任意 Python 权重变换、连续区间约束、
学习型码本或运行时类型切换。不能把逻辑二值/三值化当成磁盘压缩或速度提升。
LoRA 等适配器所在模块暂不接受类型约束，避免合并与量化的次序歧义；模型其他层
仍可使用。接口为 f32，未承诺半精度码本、TorchScript 或 `torch.compile` 支持。

完整可运行示例：[custom_weight_types.wgt](../examples/custom_weight_types.wgt)。
