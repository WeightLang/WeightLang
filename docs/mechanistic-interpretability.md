# 机械可解释性

WGT 提供命名激活、逐次干预、梯度归因、干预验证电路图和稀疏特征字典。
Rust CPU、wgtime 服务和 Python SDK 共用原生编译器产生的观测元数据。
模型定义检查和编译登记边界，研究接口显式执行实验。
[完整研究程序](../examples/native_training/interpretability/research.py)
串联采集、归因、电路发现、WGT 稀疏字典训练和特征消融。

## WGT 表达式与源码定位

`observe(name, value)` 返回与 value 相同形状、轴名称、类别和梯度语义的张量。
name 为长度 1..256 UTF-8 字节的字符串字面量，value 为受支持的张量表达式。
两参数必填，支持当前 `wgt 0.5` 的 forward 模型和纯 helper。
为兼容已有源码，同名用户 helper `fn observe(...)` 继续按原函数调用。

```wgt
wgt 0.5
model CausalExperiment
input x:f32[2]
output y:f32[2]
fn forward(x) {
    # 两个独立特征，平方保持特征轴。
    let hidden = observe("hidden", x * x);
    # 直接输入路径用于测量 hidden 对输出的独立影响。
    return hidden + x;
}
```

共享 IR 的可选 observations 数组记录 name、instruction、register、line、
scope 和 axes。旧 IR 缺省为空，旧观测元数据的新增字段可缺省。
line 为所属编译单元中的语句行号，从 1 开始，旧元数据用 0 表示未知。
root 位置对应已解析定义源码，子模型位置对应拆分后的模型源码；
导入展开与签名归一化后的编译单元由共享前端产生。
scope 保留调用上下文，例如 `forward/helper`、`forward/cells[0]`。
axes 保留命名轴，匿名轴为 null。相同子模型实例的重复调用共享实例路径，
各次静态展开由 instruction 区分，动态循环访问由 occurrence 从 0 编号。

边界采用同形 identity copy，前端和未信任 IR 校验拒绝越界位置、
重复指令位置、非法定位信息及改变形状的边界。

## 激活采集与逐次干预

```python
import torch
import weightlang as wgt

model = wgt.load("examples/native_training/interpretability/model.wgt")
x = torch.tensor([2., 3.])
baseline, events = wgt.inspect(model, x)
ablated, _ = wgt.inspect(model, x, replacements={"hidden": torch.zeros(2)})
patched, _ = wgt.inspect(model, torch.ones(2),
                         replacements={"hidden": events[0].before})
once, _ = wgt.inspect(model, x, interventions=[
    wgt.Intervention(events[0].instruction, 0, torch.ones(2))
])
```

`inspect(model, value, *, replacements=None, interventions=(), max_events=10000,
max_bytes=67108864, max_work=None, memory_bytes=None)` 返回 output、events。
value 为单个无 batch 输入。model 为 WGT ProgramModule。
replacements 按标签替换所有同名位置，interventions 按 instruction、occurrence
选择一次动态事件。两者同时匹配时逐次干预优先。重复逐次目标、未知静态位置、
未执行的动态目标均返回错误。替换值须与完整 shape、dtype、device 一致且有限。

事件包含 name、instruction、occurrence、shape、before、after 和 site 定位元数据。
替换发生在观测表达式求值后、下游消费前。置零实现消融，复用基准 before 实现
激活移植，宿主也可以构造单神经元、注意力头切片或方向干预后的完整替换值。
采集值为 detached 副本，模型执行锁保护实验快照。实验结束或失败后丢弃候选状态。

## 梯度归因

`attribute(model, value, seed, *, max_events=10000, max_bytes=67108864,
max_work=None, memory_bytes=None)` 返回 AttributionReport。
seed 与输出向量的 shape、dtype、device 相同且有限，定义标量目标
`sum(output * seed)`。选择一个输出坐标时使用相应 one-hot 向量。

报告包含 output、input_gradient、events、context。每个事件含 site、occurrence、
value、gradient 和 contribution，其中 contribution 为逐元素 value × gradient。
同一循环位置的不同访问使用独立的值版本。detach 截断输入梯度，
其下游观测激活仍可作为独立干预边界计算局部敏感度。
常量或未使用的路径返回零输入梯度。typed 操作保留既有无自动求导合同。

```python
report = wgt.attribute(model, x, torch.ones(2))
print(report.input_gradient)
print(report.events[0].gradient)
print(report.events[0].contribution)
```

Python 使用 torch.autograd.grad，保留调用方已有参数 .grad。
正常 forward 的梯度语义继续由 SDK 执行器和 PyTorch 提供。
原生实现复用已有不可变值版本 tape 和 VJP，编译前向 tape 也保留观测身份。

## 干预验证电路图

`circuit(model, value, seed, *, max_nodes=32, max_work=None, max_bytes=67108864,
memory_bytes=None, epsilon=0.000001)` 返回 Circuit。

它先记录基准路径，然后对每个动态观测事件单独置零并从相同初始状态重跑。
max_nodes 为 1..128，超过基准事件上限即失败。max_work 覆盖基准和全部实验，
epsilon 为有限非负阈值。图中节点对应基准动态事件，边记录实测总因果影响。

- 指向输出的边使用有符号目标变化 `sum((baseline - patched) * seed)`。
- 指向下游观测的边使用原激活变化的 L2 范数。
- 干预导致下游事件消失时，该边 effect 为 null。
- changed_paths 记录发生执行路径变化的实验源节点。

同一对节点的影响可能经过中间节点，图的解释范围覆盖当前输入、
选定边界和置零条件。梯度归因与这些有限干预的结果可以一并比较。
研究中也通过干预检验归因图提出的机制假设，参见
[Circuit Tracing](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html)。

```python
graph = wgt.circuit(model, x, torch.ones(2))
print(graph.edges)
print(graph.mermaid())
data = graph.to_dict()
```

to_dict 返回可保存为 JSON 的节点、边、路径变化、总 work 和 context。
mermaid 返回可渲染的流程图文本。

## 稀疏特征字典

[WGT 稀疏自编码器](../examples/native_training/interpretability/sparse_autoencoder/model.wgt)
实现 2 维激活到 4 维非负特征的编码和解码。输出前 2 项为重建，
后 4 项为特征强度。fn loss 使用重建 MSE 加 0.001 × 平均特征强度，
可以通过 Rust 的 Loss::Custom 或 Python model.loss 训练。
输入数据和优化器循环由宿主提供。

```python
sae = wgt.load("examples/native_training/interpretability/sparse_autoencoder/model.wgt")
optimizer = torch.optim.Adam(sae.parameters(), lr=0.01)
activations = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
for _ in range(80):
    optimizer.zero_grad()
    loss = sae.loss(sae(activations), activations)
    loss.backward()
    optimizer.step()

dictionary = wgt.SparseAutoencoder.from_wgt(sae)
effects = wgt.feature_effects(model, x, dictionary, events[0].instruction, torch.ones(2))
```

SparseAutoencoder(width, features, *, device=None, dtype=None) 也支持普通
PyTorch 字典训练。两个维度必须为正整数，encode、decode、forward
使用 ReLU 编码与线性解码。
loss(activations, *, sparsity=0.001) 为重建 MSE 加稀疏项，
sparsity 须有限非负。from_wgt 复制 encoder、encoder_bias、decoder、
decoder_bias 四个同名已训练参数，保留调用方 CPU RNG。

intervene(activation, feature, value) 通过
`activation + decoder_column * (value - encoded_feature)` 修改一个特征，
保留原激活的重建残差。索引须合法，value 和结果须有限。
export 返回原生 FeatureDictionary 的 row-major JSON 表示。
权重和语义标签由宿主实验维护。

`feature_effects(model, value, dictionary, instruction, seed, *, occurrence=0,
max_features=128, max_work=None, max_bytes=67108864, memory_bytes=None)`
逐个消融字典特征，返回 output、features、有符号 effects、总 work、context、
instruction、occurrence 和 dictionary_fingerprint。字典权重在扫描开始时建立快照。
特征数须为 1..max_features 且 max_features<=128。
激活为字典 width 的向量。整个扫描保持同一模型快照，工作预算覆盖全部实验。

## Rust 与 wgtime 服务

从 `wgtime_core::adaptive::research` 导入 Research、Intervention、
FeatureDictionary 和报告类型。原生值均为 f32，完整张量采用 row-major。

| 进程内接口 | 返回值 |
| --- | --- |
| AdaptiveSession::inspect(input, max_steps, research) | (Vec<f32>, Research) |
| AdaptiveSession::attribute(input, seed, max_steps, tape_bytes) | AttributionReport |
| AdaptiveSession::circuit(input, seed, max_nodes, max_steps, max_bytes, epsilon) | Circuit |
| AdaptiveSession::feature_effects(input, seed, dictionary, instruction, occurrence, max_steps, max_bytes) | FeatureEffects |

以上接口均返回 Result。Research::new(max_events, max_bytes) 的预算必填，
replacements 为 BTreeMap<String, Vec<f32>>，interventions 为 Vec<Intervention>。
低层 ForwardPass::attribute(seed) 可以对已有 tape 求归因，
其 context 为 None，高层 AdaptiveSession::attribute 包含快照标识。

Rust facade 通过唯一 wgtime 服务执行对应接口，均返回 Result<serde_json::Value>。

| facade 接口 | 必填参数顺序 |
| --- | --- |
| Model::inspect | input, replacements, max_events, max_bytes, max_steps |
| Model::intervene | input, Vec<ActivationIntervention>, max_events, max_bytes, max_steps |
| Model::attribute | input, seed, max_steps, max_bytes |
| Model::circuit | input, seed, max_nodes, max_steps, max_bytes, epsilon |
| Model::feature_effects | input, seed, dictionary_json, instruction, occurrence, max_steps, max_bytes |

服务检查模型所有权。IPC 允许最多 1024 个捕获事件、32 个电路节点、
262144 字节捕获/tape 预算和 1000000 步。
step 预算须为正，电路和特征扫描的 step 预算覆盖全部实验。
FeatureDictionary JSON 包含 width、features、encoder、encoder_bias、decoder。
编码器形状为 [features,width]，解码器为 [width,features]。
两矩阵元素数各不超过 1000000，维度为 1..16384，
特征扫描最多 128 个特征。解码 bias 在残差保留干预中抵消。

## 快照、预算与后端合同

归因、电路和特征效果报告中的 context 保存输入、目标 seed、
模型结构 SHA-256 和初始参数/状态 SHA-256。
权重发生变化会产生不同的状态标识。指纹帮助核对实验来源，
模型定义、权重及报告的持久保存由宿主显式完成，遵循[存储约定](storage.md)。

原生研究接口使用 CPU 解释执行，结束或失败后恢复先前的编译执行配置。
既有 ForwardPass 的编译 tape 也可直接归因。Python 使用 eager ProgramModule，
forward 内部状态候选和实验性写入在成功或失败后均丢弃。
同一报告中全部实验拥有相同的模型快照。
Python 在 autograd 支持的 dtype/device 上遵循 SDK 现有合同，
本次差分验收采用 CPU/f32。typed operations 和顺序 layer 入口遵循各自执行合同，
研究接口要求 forward program。

max_bytes 对捕获计入 before、after 的 payload。
Python 归因捕获额外预留 gradient 和 contribution 的 payload，
PyTorch autograd 自身内存由 PyTorch 管理。
原生 tape 字节预算覆盖保存的 primals 和 adjoints，报告副本另计。
原生归因报告最多 10000 个事件，IPC 归因报告最多 1024 个事件。
图的 max_bytes 为每次运行的捕获上限，基准节点与边的容器另计。
结构/状态指纹计算也需要序列化和暂存内存。
max_events、max_nodes 和总工作预算独立限制实验规模。
