# 组合、参数身份与训练状态（本地实现）

以下语法均要求 `wgt 0.3`。它们定义模型结构与张量计算；数据、损失、优化器、
采样和分块调度继续由 Python 管理。示例只初始化，未训练。

## block 的具名返回值

block 可以在最后一行写 `return output=<引用> <字段>=<引用> ...`。
`output` 必须出现，每个字段唯一；引用只能指向输入、已经计算的层或已声明的结果字段。
普通 block 的第一个实参名为 `input`，后续实参依次是 `arg1`、`arg2`。
调用使用 `layer pair: pair() from x,condition`，结果通过 `pair.output`、
`pair.product` 等引用；省略字段引用 `pair` 等价于主结果 `pair.output`。
没有 `return` 的旧 block 保持单输入、最后层输出语义。
字段是扁平的张量记录，不是任意 Python 对象或嵌套容器。

```wgt
wgt 0.3
model pair_demo
input x: f32[4]
input condition: f32[4]
output total: f32[4] from pair.output
output gated: f32[4] from pair.product
block pair():
  # 两个同形向量；乘法门不隐式广播。
  layer sum: add(input,arg1)
  layer gate: mul(input,arg1)
  return output=sum product=gate
layer pair: pair() from x,condition
init seed=7
```

## 显式多状态递推

`scan_state(cell()) from sequence,state0,state1,...` 使用同一组 cell 权重逐步计算。
sequence 的第一条模型轴是 T，cell 的 `input` 是移除 T 后的单步样本。
cell 必须返回 `output` 和全部 `state0`、`state1` 等字段，不能多出或遗漏状态。
每个状态保持传入时的 shape/dtype，输出样本形状必须在所有合法符号取值下保持不变。
状态数量由实参数量决定；状态可为任意支持的张量形状和 dtype，不是模型参数。

`scan.output` 沿 T 堆叠每步 output，`scan.stateN` 是最后状态。T=0 时产生空输出、
原样返回初始状态，不执行 cell。可选 batch 轴不属于模型 shape，所有输入一致带 batch
或一致不带 batch；分块的状态张量不能 detach，才能保持跨块梯度。
完整例子：[multi_state.wgt](../examples/multi_state.wgt)。
旧 `scan(hidden=H,cell())` 和 `recur` 保留原有契约；一般具名状态目前使用 `scan_state`。

## KV cache 和辅助损失作为端口

`rope_attention(...,ports=1) from x,mask,old_key,old_value`：

- x 为 `f32[T,dim]`，T>0；mask 为 `bool[P+T]`，true 表示可见。
- old_key/old_value 为相同的 `f32[P,kv_heads,head_dim]`；P 可以为 0。
- 结果字段为 output（`f32[T,dim]`）、key/value（`f32[P+T,kv_heads,head_dim]`）。
- `P+T <= max_len`；同时遵守 causal/window。全屏蔽行的注意力概率及梯度为零。
- 不支持此入口的加性浮点 mask、自定义 position_ids 或隐式缓存对象。

`ports` 默认 0，保持旧单张量执行。其余参数和默认值见中文参考的完整算子签名。
`sparse_moe(...,ports=1)` 返回 output 和 aux（`f32[]`）；带 batch 时 aux 是每样本一个值。
Python 明确选择 aux 的 batch 归约、系数及是否加入训练损失。默认 ports=0 保持旧调用。
[cache_ports.wgt](../examples/cache_ports.wgt) 同时演示有界 P/T、mask、cache 和 MoE。
`run_sequence` 仍作为旧图兼容入口；这些新图直接调用模型，不通过按类名分派的序列入口。

## 稳定参数身份与共享

`parameter <身份名> = <层路径>.<参数名>` 赋予参数独立于模块路径的名字。
`share <目标参数路径> = <身份名>` 令目标与源指向同一 Parameter，要求 shape/dtype 相同。
身份、所有者路径和共享目标不能冲突；不复制参数来模拟共享。
例如 `parameter vocabulary = tokens.weight`、`share head.weight = vocabulary`。
旧 `tie_embedding` 兼容保留，新模型优先使用名字。

- `parameter_inventory(model)` 列出 `id:<名字>`，未命名参数回退到 `path:<路径>`，包含别名。
- `migrate_parameters(source,target,mapping=None,apply=False)` 返回 reusable/conflicts/new/unused。
  mapping 是目标身份到来源身份。只有显式 `apply=True` 且没有冲突才复制；不猜测参数顺序，
  不自动扩展或缩小张量。未命名参数重命名后仍需要显式 mapping。

## 带类型的配置与固定源码打包

在端口、层之前声明：

```wgt
config WIDTH: int default=4 min=2 max=16
config SCALE: float default=0.5 min=0.0 max=1.0
config BIAS: bool default=true
require WIDTH % 2 == 0
```

int 严格接受整数（不把 bool 当整数），float 接受有限 int/float，bool 严格接受 bool。
default 必填；数值 min/max 可选且包含端点；bool 不接受数值范围。
配置与 const/dim 不得同名，未知覆盖项报错。`require` 使用已有的有限整数表达式。

- `model_config(source)` 返回 schema 和默认 values。
- `load(source,config={...})` 在内存构建；`parse`、`parse_file`、`compile_file` 同样接受 config。
- `package_model(source,output,config=None,force=False)` 解析依赖、展开 block、固定配置并打包
  一份独立架构源码；不创建参数、不读取训练数据、不训练。已有文件默认不覆盖。
- 已固定制品不允许再用 config 覆盖。带显式权重导入、继承、派生、冻结规则或自定义初始化的
  包当前使用 `build_bundle`；`package_model` 会明确拒绝，不能静默丢弃这些声明。

完整例子：[configured_adapter.wgt](../examples/configured_adapter.wgt)。

## 声明式适配器

`adapter <名字>: lora target=<linear层路径> rank=<正整数> alpha=<正数> freeze=1`
将普通、不共享的 linear 变成 LoRA linear；rank 不超过 min(in,out)，alpha 必须有限且为正。
freeze 默认 1，只冻结此 linear 的基础 weight/bias；其他冻结范围用已有 freeze/unfreeze。
A 使用 Kaiming 初始化，B 为零；增量为 `(alpha/rank) * B @ A`。
当前不支持共享基础参数、卷积或 fused 模块直接作为适配器目标。

- `merge_adapter(model,name)`：仅 eval 模式可合并，重复调用不重复叠加；合并时保存基础权重。
- `unmerge_adapter(model,name)`：精确恢复基础权重；合并状态下不允许训练。
- `save_adapter(model,name,path,force=False)` / `load_adapter(model,name,path)`：独立适配器文件，
  仅含 A/B、rank/alpha 和基础权重摘要；不同基础权重不能直接加载，加载前必须解除合并。
- 完整模型保存保留未合并的基础权重和适配器声明，重载不重复叠加。
  适配器参数身份由适配器名稳定标识；不是临时替换 forward。

DoRA、PiSSA、QLoRA 现已提供相同声明入口；把方法名换成 dora/pissa/qlora，
后三种要求 freeze=1。QLoRA 为 NF4 运行时打包，完整模型文件仍 F32。
新增方法的完整签名、公式、导入/精度限制、制品格式及例子见 [适配器](adapters.md)。

## 训练断点与模型制品

`save_checkpoint(model,optimizer,path,progress=None,force=False)` 保存参数身份、参数/缓冲区、
SGD/Adam/AdamW 的优化器状态及参数组、可训练性、模块模式、Python/NumPy/Torch CPU/CUDA RNG，
以及 JSON 训练进度；不保存架构，不代替完整模型制品。合并后的 LoRA 必须先解除合并。
`checkpoint_report(model,path,mapping=None)` 返回 reusable/conflicts/missing/unused。
`restore_checkpoint(model,optimizer,path,mapping=None,restore_rng=True)` 要求严格兼容，恢复前验证
参数及优化器状态，返回训练进度。mapping 同样是目标身份到来源身份；不按参数顺序恢复。
参数组内部可以重新排序；优化器种类及分组身份必须相同。缓冲区路径必须一致；重命名的模块
保留调用方当前 train/eval 模式。没有承诺跨 Torch 版本或跨设备位级重现。

带参数身份/共享/适配器的完整模型使用制品格式 6（可同时携带符号维度）。
具名端口使用格式 4，符号维度使用格式 5；记录结果字段保存在层的 results/returns 中。
训练断点为独立 `weightlang.training-checkpoint` v1，适配器为 `weightlang.lora` v1。
所有这些是本地实现与本地验收，不表示已发布。

## 尚未覆盖的路线

原子算子的签名/default 已由 Python 和 Rust 共享同一目录，旧算子的形状推导仍有重复实现。
尚未完成统一算子 IR 与四部分精度契约（存储、计算、累加、输出）；当前完整模型权重仍为 F32。
现有 cast_after/norm_cast_after/rope_fp32 仍按原语义工作。不要把配置支持或动态前端 dtype
能力等同于已完成精度策略。split 可用多个 slice 组合；通用多输出 split/top-k、arange/cast
仍需要独立设计与验证。

## 条件与自适应计算增量（0.3，本地未发布）

`route(branch0,branch1,...) from selector,x` 支持按样本真正选择执行路径；
`recur(N,body,until=stop,compact=1)` 压缩活动样本；
`ponder(N,body,until=stop)` 返回可微停止分布、每步状态和期望计算量，训练留在 Python。
`resize2d`、`pad2d`、`avgpool2d` 支持多尺度空间组合。
完整签名、默认值、约束、限制和可运行例子见 [adaptive-compute.md](adaptive-compute.md)。
