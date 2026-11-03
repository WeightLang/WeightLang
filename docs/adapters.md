# 原生低秩适配器（本地 WGT 0.3）

支持 `lora`、`dora`、`pissa`、`qlora`。WGT 定义适配器和张量计算，Python
准备数据、计算损失、选择优化器和运行训练。以下是本地实现，不代表发布或任务质量。

## 声明与共同契约

```wgt
wgt 0.3
model adapted
input x: f32[8]
output y: f32[5] from projection
layer projection: linear(in=8,out=5)
# 输入特征 8 -> 低秩轴 3 -> 输出特征 5；仅 A/B 与 DoRA 幅度可训练。
# 随机基础模型只用于演示；正式微调需要显式导入预训练权重。
adapter task: dora target=projection rank=3 alpha=6 freeze=1
init seed=7
```

完整签名：`adapter NAME: METHOD target=PATH rank=R alpha=A [freeze=1]`。
NAME 唯一；METHOD 为上述四种之一；target/rank/alpha 必填，freeze 默认 1。
目标必须为普通、不共享 weight/bias 的 `linear`；不支持卷积、共享基础权重、
fused attention 模块或在同一层叠加多个适配器。rank 为整数且
`1 <= rank <= min(in,out)`；alpha 为有限正数。LoRA 允许 freeze=0/1，
其余三种要求 freeze=1。freeze 只影响目标层基础 weight/bias，其他层要独立冻结。
输入输出沿用 linear 的 f32 张量及 shape/batch 契约；A 为 `[rank,in]`，B 为 `[out,rank]`。
参数身份为 `adapter:NAME:a/b`；DoRA 增加 `adapter:NAME:magnitude`。
QLoRA 的基础 weight 不再是 Parameter，原来的基础 weight 身份不进入优化器参数清单。

## 方法语义

| 方法 | 初始化与执行 | 限制 |
| --- | --- | --- |
| lora | A 为 Kaiming，B 为零；`W + (alpha/rank) BA` | 保持原有语义 |
| dora | 幅度 m 初始化为 W 每行 L2 范数；方向 D=`W+(alpha/rank)BA`，有效矩阵 `m*D/stopgrad(norm(D))` | 范数按输出行计算、下限为 f32 epsilon；bias 不归一化 |
| pissa | 精确 reduced SVD，取前 rank 个奇异分量初始化 A/B；W 保存减去该分量后的冻结残差 | 初始总函数与原 W 在浮点误差内相等；大矩阵 SVD 有计算及内存成本，不提供快速迭代 SVD |
| qlora | 冻结基础权重以 NF4 存储，64 元素一个 absmax；两码一个 uint8；A/B 和计算保持 f32 | 每层前向和输入梯度反向临时反量化，不保留全模型稠密基础权重用于反向 |

QLoRA 基础权重常驻空间约为 `ceil(N/2) + 4*ceil(N/64)` 字节，不含 bias、A/B、
激活和算子临时内存。这是实际 uint8 打包，不是保留整份 F32 权重的假量化。
当前为纯 PyTorch NF4 参考执行，不含 double quantization、paged optimizer、
bitsandbytes CUDA 内核或 BF16 计算；不能宣称论文完整训练栈或特定大模型显存/速度指标。
完整模型文件仍沿用 WGT F32 制品：写入解码后的基础矩阵，加载时重新打包；
**文件没有四位压缩收益**，构建、导出及保存仍可能有稠密临时内存。
NF4 码表和块大小固定，没有未实现的 bits/blocksize 参数。

## 预训练基础模型与训练

先导出普通基础模型，再在适配器源码中显式导入对应 linear：

```wgt
# 将此行加到上方完整模型末尾；路径必须指向实际已有的完整模型目录。
import projection from "../../weights/base/base" layer=projection
```

对普通 linear 的直接导入先复制预训练权重，再做 DoRA/PiSSA 初始化或 NF4 量化，
新适配器不会被冒充为训练过的权重。不要添加 import 的 freeze 标志，除非希望连
适配器一起冻结。对已经适配的相同方法模型，恢复完整残差与适配器状态。
普通模型导入必须逐个指定被适配的 linear；暂不支持通过父 block 导入自动生成其内部适配器。
三种新方法暂不接受 `init file`/`init python`，先生成普通基础模型再显式导入。

```python
import torch
import weightlang

model = weightlang.load("examples/adapters/dora.wgt")
model.train()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)
# x/target 由 Python 提供；本例输入 [batch,8]，目标 [batch,5]。
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(x)["y"], target)
loss.backward()
optimizer.step()
weightlang.save_adapter(model, "task", "weights/my_adapter/task.pt")
```

完整可运行的合成训练示例：
`PYTHONPATH=src python examples/adapters/train.py --method dora`，
也可选 pissa/qlora；默认仅在内存训练，不写制品。显式 `--output weights/my_adapter/task.pt`
才保存独立适配器，已存在时拒绝覆盖。合成损失下降只验证训练路径，不证明实际任务质量。

## 保存、合并与恢复

共同 Python 接口不变：

- `save_adapter(model,name,path,force=False)` / `load_adapter(model,name,path)`：
  LoRA 仍为 `weightlang.lora` v1；其余为 `weightlang.adapter` v2，携带 method。
  DoRA 保存 A/B/magnitude；PiSSA、QLoRA 保存 A/B。加载校验方法、rank/alpha、
  基础矩阵和 bias 摘要、张量 shape/dtype/有限值；不兼容时在复制前拒绝。
  PiSSA 摘要对应冻结残差，QLoRA 对应 NF4 解码矩阵；须配合同一基础状态，不能当作通用 PEFT 文件。
- `merge_adapter(model,name)`：仅 eval，重复调用不重复叠加。
  QLoRA 合并为临时 F32 有效矩阵，会失去该层低位常驻内存优势。
- `unmerge_adapter(model,name)`：精确恢复基础状态；QLoRA 原始码和尺度保持不变。
  合并状态禁止训练和适配器加载。完整模型保存采用未合并的规范状态，重载不重复叠加。
- `save_checkpoint(model,optimizer,path,progress=None,force=False)`、
  `restore_checkpoint(model,optimizer,path,mapping=None,restore_rng=True)`、
  `checkpoint_report(model,path,mapping=None)` 沿用训练断点契约；QLoRA 的 packed/scales
  也随断点保存。先解除合并；支持 SGD/Adam/AdamW，不承诺跨设备或跨版本位级一致。

实现与验证入口：`src/weightlang/adapters.py`、`tests/test_adapter_methods.py`。
独立 Python 导出和反编译保留方法。Rust 校验完整模型的规范 F32 参数布局。
参考：[DoRA](https://arxiv.org/abs/2402.09353)、[PiSSA](https://github.com/MuLabPKU/PiSSA)、
[QLoRA](https://arxiv.org/abs/2305.14314)。
