# LightNest 架构迁移

这是 **WGT 参数计算 + Python 动态宿主** 的可运行项目。迁移的是当前
LightNest 主线全部模型架构，不包含 `archive/` 中已经归档的历史实验。
没有迁移旧训练权重，也没有启动重训。初始化模型不能用于声称 ELF 生成质量。

从 WeightLang 仓库根目录运行，使用当前本地源码（包含尚未发布的 WGT 0.3）：

```bash
# 在已有 WeightLang 环境中安装可选的字节分词器依赖。
uv pip install --python .venv/bin/python tokenizers==0.23.3

# 检查所有架构的输出、参数梯度和保存/重载。
PYTHONPATH=src .venv/bin/python examples/lightnest/verify.py

# 构建默认 ELF 生成器架构，不训练，不覆盖已有模型。
PYTHONPATH=src .venv/bin/python examples/lightnest/project.py elf

# 构建 RGB；可把模型名换成下表中的其他名称。
PYTHONPATH=src .venv/bin/python examples/lightnest/project.py rgb
```

默认模型只写入本项目 `weights/<模型名>/`。要另存版本，使用例如
`--output examples/lightnest/weights/elf-v2`。不会使用 `LightNest-artifacts`，
不会把模型放进 `build/` 或 `target/`，没有默认覆盖开关。

| 模型名 | 架构 |
| --- | --- |
| `elf` | 完整 ELF 字节生成器、两层 GRU、Red 迭代、复制与停止头 |
| `rgb` | Green 观察/路由、Blue/Red 字节编辑、notebook、风险和预算头 |
| `hnet_dense`, `hnet_fixed`, `hnet_dynamic` | 三种分块策略，保留跨调用状态 |
| `any2any`, `any2any_mean` | Transformer 与 GRU/均值请求编码 |
| `any2any_chunks`, `any2any_bpe` | 字节卷积分块、无损 BPE、共享词表语言头 |
| `structure`, `token_structure`, `ast_structure` | 字节卷积、词法 GRU、AST GRU 结构预测 |
| `openat` | 有界路径和 flags 的三分类预测器 |
| `x86_world` | 候选/长度/反馈头，加 ELF 加载、有限 x86 与 PTX 语义 |

`models/index.json` 保存默认配置、WGT 源码入口、宿主模块路径和显式参数映射。
默认构建实际读取 `models/<模型名>/*.wgt`；这些文件是可执行的组件，**每个组件
不是一个完整 LightNest 网络**。整模型的连接和动态行为由 `backend.py` 与
`lightnest_host/` 共同定义，不能单独 `weightc run` 一个组件来代替整模型。

使用 `--config config.json` 可设置原模型构造参数；此时按配置生成新的 WGT 组件。
`--small` 使用验证用的小尺寸，不改变默认架构尺寸。默认 BPE 配置只有完整 256
字节字母表，没有学习过的合并规则；实际语料分词器可通过 `semantic_bpe` 指定。

## 迁移的边界

WGT 执行线性投影、嵌入与 LayerNorm。卷积以宿主展开窗口加 WGT 线性投影执行，
保留 `[out, in, kernel]` 顺序。GRU 使用 WGT 的输入/隐藏投影，宿主按 PyTorch
`reset-after` 公式计算门控；没有使用公式不同的 `std/recurrent.gru`。
无 bias 的投影在 WGT 中具有显式恒零 bias，运行时冻结，不引入可训练自由度。
padding embedding 的行梯度保持为零。

以下仍由 PyTorch/Python 宿主执行，未伪装为 WGT 算子：

- 变长与 packed sequence、跨调用状态、动态分块与停止条件；
- 注意力矩阵乘法、mask、top-k 路由、notebook、复制分布、共享词表输出投影；
- 字节/AST/分词预处理、损失、动态形状和张量索引；
- ELF 加载、有限 x86/PTX 转移。它们不是学出来的完整虚拟机。

`landmark_bias` 和 `slot_keys` 使用 WGT embedding 制品承载参数，但它们在宿主
注意力中直接使用；参数映射明确标记为 `host attention parameter`。
这两项是权重存储迁移，不是完整注意力算法的 WGT 实现。

原仓库的外部 Ollama CLI、隔离执行服务、语料构建和历史训练命令没有复制进来。
它们不是模型架构。`lightnest_host/` 保留当前模型及其必要的确定性宿主，来源和
两边文件哈希见 `provenance.json`，许可边界见 `NOTICE.md`。

## 后续训练与保存

`project.load()` 返回普通 `nn.Module`，可使用现有损失和 PyTorch 优化器。
先构建一次，然后在设置了 `PYTHONPATH=src:examples/lightnest` 的环境中：

```python
from project import load, save, ROOT
import torch

model = load(ROOT / "weights/elf").train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# 用真实训练批次调用 model，计算 loss，再 backward / optimizer.step。
# 完成实际训练后，才填写真实的 training 记录：
# save(model, ROOT / "weights/elf-trained", training={"method": "AdamW", "steps": actual_steps})
```

保存请使用 `project.save()`，它为全部组件调用 WeightLang 的保存接口，保留
WGT 源码、映射和配置；不能直接改 manifest 绑定的 safetensors 文件。
`project.load()` 能重新组装保存后的模型。单个组件制品放在
`weights/<模型名>/components/`，必须保留整个项目模型目录。
优化器状态、训练数据和训练循环不包含在模型快照中。

## 验证证据

`verification.json` 是固定 seed 的 CPU 架构对比，不是训练质量报告。
对照模型和迁移模型使用完全相同的初始化参数；迁移模型经过保存和重载后比较。
检查输出形状/dtype/有限性、离散选择、GRU 流式状态、三种注意力路由、padding、
BPE 共享词表、逐参数梯度、一次匹配 SGD 更新，以及更新后的保存/重载。
训练检查在关闭 dropout 的 eval 模式下启用 autograd；未运行完整训练，也未测 GPU。

ELF/x86/PTX 另有已知 `exit(37)`、坏头拒绝、整数溢出和成功候选接受用例。
这不是新做的原生二进制执行验收，也不增加原项目支持的指令范围。

`models/` 为默认尺寸的源码；行为测试用小尺寸覆盖全部架构配置。默认尺寸已构建、
保存和重载组件，但没有运行大尺寸整模型性能或质量评测。
`export_sources.py` 可重现默认 WGT 源码，已有 `models/` 时拒绝覆盖。
一次性制品在本项目 `.cache/weightlang/` 中创建，正常完成或异常退出均自动清理。
