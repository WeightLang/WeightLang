# 权重快照、引用与显式迁移

权重操作使用三个对象：`WeightSet` 保存一组不可变参数值，`WeightRef` 指向某一版本中的一个完整参数名，`Plan` 为目标模型的每个参数指定来源。构造计划与执行模型分开；合并或迁移先得到新的快照，再由宿主明确恢复为实例。本文覆盖当前仓库的 Rust/Python 实现，尚未发布。

## 快照合同

`WeightSet` 只保存参数，不保存架构、运行状态、优化器历史或 RNG 进度。`revision` 是 `weights.safetensors` 原始字节的完整 SHA-256；同一文件被读取、复制或验证时保留该 revision。重新编码同一组数值可能产生不同字节，因此不要把 revision 当作跨编码器的数值相等判断。`manifest.json` 记录格式、revision、固定 payload 名、张量表和 provenance，见 [Rust 实现](../crates/wgt-model/src/weights.rs) 与 [Python 实现](../python/weightlang/weights.py)。

| 项目 | 当前边界 |
| --- | --- |
| 格式 | `weightlang.weights.v1`；payload 为 `weights.safetensors` |
| 值 | 有限 F32、稠密、行优先；不隐式转 dtype、转置、裁剪或补值 |
| 形状 | rank 1..8，每轴 1..16384；元素数须与存储长度一致 |
| 文件上限 | payload 整体最多 64 MiB，包含 safetensors 头；manifest 最多 1 MiB |
| 保存 | 目标目录必须不存在，manifest 最后发布；已有快照不覆盖 |
| 读取 | 检查格式、摘要、张量表和 payload；可额外要求指定完整 revision |
| 绑定 | 完整目标参数名集合、形状和存储值域均匹配；未知项和缺项都报错 |

Python 返回的张量是副本，Rust 返回只读借用；修改宿主张量不会修改原快照。参数全名沿用共享前端给出的名称，例如 `projection.weight` 或子模型展开后的名称，不做后缀猜测、通配匹配或自动重命名。当前快照没有独立的持久参数 ID 或架构语义摘要；同名同形是否仍表示同一种特征，由迁移代码明确负责。

learn 参数按实际存储导出。`learn bool` 与 `learn int` 保留连续 shadow，不在快照、复制或恢复时先取整；可训练概率参数保存 logit，固定概率保存概率值。它们不同于 `update_parameters` 的外部离散候选合同。完整恢复会依照目标声明验证存储域，并保留合法的连续值。

## 固定引用与运行中共享

Python 的 `snapshot.ref(name)` 和 Rust 的 `snapshot.reference(name)` 生成固定 `WeightRef`，携带 revision、完整参数名和对应值。即使磁盘路径后来发生变化，已经读入的引用也不跟随它改变。读取时可先指定 revision，再取得引用；Rust getter 为 `revision()`、`name()`、`tensor()`，Python 为 `.revision`、`.name`、`.tensor()`。

引用用于复制或计算新的快照值，不建立两个运行实例之间的联动。一个实例中的参数共享仍由 WGT 子模型/参数所有权定义；历史快照引用不是 live tie。权重快照的 revision 也不是服务实例用于更新的 revision，不能当作并发修改的 CAS 令牌。

## 计划覆盖规则

无 base 的 `Plan` 初始没有任何绑定，每个目标必须通过完整张量、固定引用或初始化器得到值。`set` / `copy` 重复写同一目标报错。`replace` 必须指向已经存在的绑定，而且每个目标最多替换一次；它把覆盖变成显式操作。当前不提供隐式优先级、通配覆盖或目标切片赋值；切片、拼接和矩阵运算先生成完整目标张量，再赋值。

提供 `base` 是明确请求继承整组参数。base 必须满足目标完整名称、形状和存储域合同，然后只为选中的目标调用 `replace`。架构改名、扩宽或增加参数时使用没有 base 的迁移计划，明确枚举复制、变换与新增初始化。名称和形状验证不能替代对架构含义的判断。

`preview` 只报告覆盖情况、形状与来源，绝不调用初始化回调。缺项会在 `build` 执行任何回调前失败；通过完整性检查后，每个选中的回调在该次 build 中执行一次，其结果仍须通过值域和形状验证。失败不会产生新的快照或写入运行实例，但宿主回调自身的副作用无法回滚，应使用纯算法。Python 的计划可再次 build，回调也会再次执行；Rust `build` 消耗计划值。

## Python：只合并指定两组参数

假定 `model.wgt` 编译出的参数包含 `encoder.weight` 和 `head.weight`，两个已保存版本都符合该定义。下例明确继承版本 A，只替换这两个完整参数：A 占 0.75，B 占 0.25。`sources` 把参与计算的固定引用记录进 provenance；它不会检查任意 Python 表达式是否真的使用了这些来源。

```python
import weightlang as wgt

definition = wgt.compile("model.wgt")
version_a = wgt.WeightSet.load("weights/model/version-a")
version_b = wgt.WeightSet.load("weights/model/version-b")
plan = wgt.Plan(definition, base=version_a)
for name in ("encoder.weight", "head.weight"):
    left, right = version_a.ref(name), version_b.ref(name)
    assert left.tensor().shape == right.tensor().shape  # No implicit broadcasting.
    blended = 0.75 * left.tensor() + 0.25 * right.tensor()
    plan.replace(name, blended, sources=[left, right])

print(plan.preview())
merged = plan.build()
merged.save("weights/model/merged-v1")
model = wgt.restore(definition, merged, device="cpu")
```

若两层来自同一快照，取两次不同参数名即可；形状不同应先明确设计 reshape、转置或选择区域，而不是依靠广播。需要固定输入文件版本时用 `WeightSet.load(path, revision=expected_sha256)`。`wgt.WeightSet.from_model(model)` 从现有模块捕获参数；模块先显式转换为 F32，再捕获非 F32 训练结果。

## Python：扩宽模型并只初始化新增区域

以下两个定义以 `[输出特征, 输入特征]` 保存矩阵。旧模型有两个输出，新模型有三个输出；新增一行和一个偏置的生成算法由迁移宿主明确指定。

旧 `model.wgt`：

```wgt
wgt 0.5
model ProjectionV1
input x: f32[2]
output y: f32[2]
parameter weight: f32[2,2] = eye()
parameter bias: f32[2] = zeros()
fn forward(x) {
    return linear(x, weight) + bias;
}
```

新 `model.wgt`：

```wgt
wgt 0.5
model ProjectionV2
input x: f32[2]
output y: f32[3]
# Migration must provide both complete target tensors.
parameter weight: f32[3,2] = external
parameter bias: f32[3] = external
fn forward(x) {
    return linear(x, weight) + bias;
}
```

```python
import torch
import weightlang as wgt

target = wgt.compile("model.wgt")  # ProjectionV2
old = wgt.WeightSet.load("weights/projection/v1")
weight, bias = old.ref("weight"), old.ref("bias")
plan = wgt.Plan(target)
plan.set("weight", torch.cat([weight.tensor(), torch.zeros(1, 2, dtype=torch.float32)], dim=0), sources=[weight])
plan.set("bias", torch.cat([bias.tensor(), torch.zeros(1, dtype=torch.float32)], dim=0), sources=[bias])
print(plan.preview())
next_weights = plan.build()
next_weights.save("weights/projection/v2")
model = wgt.restore(target, next_weights, device="cpu")
```

只增加一个独立参数时，可 `plan.copy` / `plan.reuse` 现有目标，再 `plan.initialize(["new_parameter"])` 明确选择新增参数的 WGT 配方；自定义算法使用 `plan.set("new_parameter", lambda shape: ...)`。若参数改名，用 `plan.copy("new_name", old.ref("old_name"))` 明确建立映射。只要有任何目标没有来源，build 就失败，不会自动选择同名参数或源码配方。可执行流程见 [`python/examples/weight_lifecycle`](../python/examples/weight_lifecycle)。

## Rust：同一组合同

Rust facade 的库名为 `wgt`，`wgt::weights` 导出 `Tensor`、`WeightSet`、`WeightRef`、`Plan`。常用签名如下；完整可编译宿主示例见 [`weights_roundtrip.rs`](../crates/wgt-model/examples/weights_roundtrip.rs)。

| 操作 | Rust API |
| --- | --- |
| 新空计划 | `Plan::new(&definition)` |
| 完整继承 | `Plan::from_base(&definition, &base)?` |
| 固定来源复制 | `plan.copy(target, snapshot.reference(source)?)?` |
| 显式张量 | `plan.set(name, tensor)?` / `plan.replace(name, tensor)?` |
| 记录计算输入 | `set_with_sources(name, tensor, &[refs])` / `replace_with_sources(...)` |
| 声明配方 | `plan.initialize_declared(name)?` |
| 自定义算法 | `plan.initialize(name, move |shape| -> wgt::Result<Tensor> { ... })?` |
| 审查与构造 | `plan.preview()` / `plan.build()?` |
| 保存与固定读取 | `snapshot.save(directory)?` / `WeightSet::open(directory, Some(revision))?` |
| 捕获运行实例 | `model.snapshot()?` |

`Tensor::new(shape, values)` 验证有限 F32 与精确长度，`scaled`、`add`、`lerp`、`reshape`、`slice(axis,start,end)`、`Tensor::concat(axis,&[...])` 和 rank-2 `transpose` 返回新张量。`lerp(other, alpha)` 使用 `(1-alpha)*self + alpha*other`，alpha 必须在 `[0,1]`；`add` 和 `lerp` 要求形状完全一致。迁移需要的多轴 slice/concat 是宿主权重 API，不改变 WGT 初始化表达式的一维 slice/concat 限制。

```rust,ignore
let left = version_a.reference("head.weight")?;
let right = version_b.reference("head.weight")?;
let mut plan = wgt::weights::Plan::from_base(&definition, &version_a)?;
let blended = left.tensor().lerp(right.tensor(), 0.25)?;
plan.replace_with_sources("head.weight", blended, &[left, right])?;
let report = plan.preview();
assert!(report.complete);
let merged = plan.build()?;
merged.save("weights/model/merged-v1")?;
```

## 恢复新实例与更新现有实例

Rust `Client::load_snapshot(&definition, directory, revision, options)` 检查 manifest 和可选 revision，再由服务校验同一 payload 的摘要并直接构造实例。`Client::load_weights(&definition, path, options)` 直接读取完整 safetensors，不要求 manifest。两者都跳过参数初始化器，允许绑定源码中未初始化的参数；新实例的状态仍按定义初始化，优化器按 options 明确选择。

现有 `Model::load_weights(path, optimizer)` 的对象是已经创建的实例，保留句柄和操作状态，要求 frozen 参数与当前值相同，成功时重置优化器历史；失败保持原实例。新实例的首次完整绑定允许为 frozen 提供外部值。要避开源码参数配方，直接使用 Client 的构造入口，不先 `initialize` 再覆盖。`Model::save_weights` 只输出参数 safetensors；`snapshot().save(...)` 另保存版本 manifest。

Python `wgt.restore(definition, snapshot, device=..., dtype=...)` 直接绑定完整权重并创建新模块；`wgt.from_ir(definition)` 则明确执行声明配方。SDK 的 `.pt` / `.pth` 保存的是架构与 `state_dict`，包含已注册状态 buffer，其 `wgt.load` 恢复路径跳过参数和状态配方。快照不包含优化器；恢复后由宿主创建所需优化器。具体 SDK 用法见 [Python SDK](python-sdk.md)。

## 验证与边界

定义/配方的行为回归在 [`initializers.rs`](../crates/wgt-language/tests/initializers.rs)，Rust 快照/计划在 [`weights.rs`](../crates/wgt-model/tests/weights.rs)，Python 在 [`test_weights.py`](../python/tests/test_weights.py)。服务恢复另外需要真实 IPC 测试 [`weights_service.rs`](../crates/wgt-model/tests/weights_service.rs)；通过静态解析不代表服务运行已经验证。

当前没有自动架构匹配、稳定逻辑 ID 重命名协议、分布式分片、部分加载、自动 dtype 转换或权重专用 CLI。provenance 是可审查的来源记录，不是可自动执行的迁移程序或可信签名。任意宿主算法需要可复现时，应额外固定算法代码、seed 与依赖版本；精确恢复以保存并校验 payload 为准。
