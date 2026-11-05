# Weight ABI v1（设计草案）

> **状态：未实现的格式提案。** 当前原生模型制品使用 model.json、checkpoint.json 与 manifest.json；本文定义未来可互操作的权重交换边界，不改变现有文件格式，也不表示当前加载器接受此 ABI。

## 1. 目标与非目标

Weight ABI 定义一个可验证、可绑定、可迁移的参数集合。它不定义模型计算图、训练循环、优化器算法或服务实例协议。

目标：

- 权重身份独立于 WGT 模块路径和 Tensor 临时值。
- 多个参数名可以指向同一份 canonical storage。
- 允许权重与模型定义独立发布、加载和绑定。
- 通过显式元数据支持 dtype、shape、layout、量化和约束。
- 大型 payload 使用二进制 shard，避免 JSON 数值数组的膨胀和解析成本。
- 校验失败时拒绝加载，不进行部分绑定或静默 cast。

非目标：

- 不是 optimizer checkpoint；优化器状态应由独立 checkpoint contract 管理。
- 不是训练数据、RNG 状态、服务运行状态或执行图格式。
- 不承诺任意 dtype、设备或量化方案都可由每个后端执行。

## 2. 文件布局

建议目录：

~~~text
weights/my-model/
├── weight-manifest.json
└── shards/
    ├── 00000.wgtb
    └── 00001.wgtb
~~~

manifest 是 UTF-8 JSON；payload 是二进制。一个 manifest 可以引用多个 shard。写入时先写临时文件并完成 fsync/hash 校验，最后原子发布 manifest 或完成标记。读取端只把具有完整且有效 manifest 的目录视为可用权重集。

## 3. Manifest schema

v1 使用严格 schema：未知的顶层字段应拒绝，或通过显式 extension namespace 承载；不能静默解释未知字段。以下为概念示例，具体 JSON Schema 应在实现前冻结：

~~~json
{
  "format": "weightlang.weight-abi",
  "format_version": 1,
  "weight_set_id": "urn:uuid:...",
  "model": {
    "name": "Transformer",
    "revision": "model-revision-id",
    "interface_hash": "sha256:..."
  },
  "weights": [
    {
      "id": "urn:uuid:...",
      "bindings": ["encoder.0.attention.wq"],
      "shape": [768, 768],
      "dtype": "f32",
      "layout": {
        "kind": "dense",
        "order": "row_major"
      },
      "trainability": "trainable",
      "constraints": [],
      "initializer": {
        "kind": "kaiming_uniform",
        "arguments": {}
      },
      "storage": {
        "shard": "shards/00000.wgtb",
        "offset": 64,
        "length": 2359296,
        "alignment": 64,
        "sha256": "..."
      }
    }
  ],
  "aliases": [
    {
      "binding": "decoder.0.attention.wq",
      "weight_id": "urn:uuid:..."
    }
  ],
  "extensions": {}
}
~~~

字段含义：

- weight_set_id：该权重快照的稳定身份；每次不可变快照有独立 ID。
- model.revision：产生或验证该权重集的模型定义版本。
- interface_hash：参数接口摘要，用于防止只凭同名绑定错误加载。
- weights[].id：参数身份；共享绑定必须引用同一 ID，不得复制一份相等但独立的参数。
- bindings：一个 canonical Weight 可暴露的路径。绑定名不参与身份计算。
- shape/dtype/layout：逻辑张量合同。v1 的 layout 只允许明确声明的 dense row-major；扩展 layout 必须带版本化定义。
- trainability：trainable、frozen 或 inference_only。后两者不自动意味着文件不可读。
- constraints：参数值域约束的机器可校验表达；未知约束类型必须拒绝绑定。
- initializer：可选的初始化来源记录，不要求恢复时重新初始化。
- storage：payload 的精确字节范围、对齐和 SHA-256。offset/length 必须在文件边界内且不得非法重叠。
- aliases：额外绑定路径到 canonical Weight ID 的映射。alias 不能形成链或环；必须直接指向存在的 ID。
- extensions：命名空间扩展；核心加载器不解释未知扩展，但必须保留或明确拒绝，不能悄悄丢弃后重新签发。

## 4. Payload contract

v1 只定义 dense contiguous row-major payload。元素字节序固定为 little-endian；每个 shard 的范围按声明对齐。payload 不包含隐式 header；dtype 和元素数由 manifest 决定。

v1 最小 dtype 集合建议为 f32、f16、bf16、i8、u8、i32。不支持的 dtype 必须明确报错，不得静默转成 f32。元素数量为 shape 维度乘积，乘法必须进行溢出检查；计算得到的字节数必须与 storage.length 完全一致。

量化数据需要独立描述逻辑 dtype 与物理 payload dtype，以及 scale/zero-point、量化轴或 block、舍入和饱和规则。v1 可先拒绝未标准化的量化 metadata；不要把存成 i8 误认为完整的 INT8 语义。

## 5. Binding 与共享语义

绑定流程分为两个阶段：

1. **Validate**：校验格式版本、manifest schema、hash、payload 范围、dtype、shape、layout、约束、ID 唯一性和 alias 完整性。
2. **Bind**：将模型参数路径映射到 canonical Weight ID；检查模型接口 hash 或显式兼容规则。只有全部参数校验成功后才提交绑定。

同一个 Weight ID 只能创建一份 canonical storage 和一份参数更新状态。多个模块路径引用该 ID 时，读取共享数值；梯度按 ID 累积；optimizer/search state 按 ID 归属。View/transpose/slice 是 Tensor 的存储视图，不创建新的 Weight identity。

默认要求精确 dtype 和 shape 匹配。任何 cast、transpose、reshape 或量化转换都必须由调用者显式选择并记录转换结果，不由加载器猜测。

## 6. 版本、完整性与安全

- format_version 是 ABI 主版本；不兼容变化必须升级主版本。
- model.revision 与 weight_set_id 分别版本化；不要把 Git commit SHA 当作唯一参数身份。
- SHA-256 用于检测损坏，不是来源签名。需要供应链真实性时，应在 ABI 外层添加签名 envelope。
- manifest 引用的路径必须是相对路径，拒绝绝对路径、..、符号链接逃逸和重复规范化路径。
- 拒绝整数溢出、越界、重叠 payload 区间、重复 Weight ID、悬空 alias、alias 链/环、未知核心字段和不受支持的 dtype/layout。
- 加载不得执行 manifest 中的代码、脚本、插件或任意 initializer。
- 失败时不得留下部分绑定的模型实例；写入失败不得产生可误认的完整制品。

## 7. 与现有制品的关系

当前 model.json 是可加载的推理模型制品，checkpoint.json 保存模型和优化器状态，manifest.json 描述原生构建产物。Weight ABI v1 未来应作为独立的权重交换文件格式：

| 格式 | 负责内容 | 不负责 |
| --- | --- | --- |
| WGT source | 模型定义与计算语义 | 参数值快照 |
| Weight ABI | 参数值、身份、绑定和存储元数据 | 优化器状态、服务运行状态 |
| Model artifact | 当前原生推理模型打包格式 | 通用跨框架 ABI 保证 |
| Training checkpoint | 模型与训练恢复所需状态 | 通用权重交换兼容保证 |

迁移必须提供显式导入/导出工具和 roundtrip 测试。不能直接重命名现有 model.json，也不能在缺少转换验证时宣称二者兼容。

## 8. 最低一致性测试

- 同一 manifest 多次读取结果一致，hash 对每个 payload 范围匹配。
- 共享绑定加载后 canonical storage 数量等于唯一 Weight ID 数量。
- 相同 shape 但不同 dtype 的权重默认拒绝。
- interface hash 不匹配时拒绝，显式兼容规则可覆盖且留下记录。
- 缺失 shard、损坏 hash、越界 offset、长度不匹配、整数溢出、路径穿越、重叠区间全部拒绝。
- 未知 ABI 主版本、核心字段和 layout 拒绝；扩展字段遵循 namespace policy。
- 加载失败无部分实例；保存中断后没有完整标记。
- 导入 -> 导出 -> 导入后，前向输出、共享身份和参数值一致。
- 若支持转换，必须单独测试量化误差、dtype cast 和 layout materialization；转换不能混入普通 roundtrip 测试。

## 9. 实施顺序

1. 先冻结 JSON Schema、二进制 shard payload contract 和测试向量。
2. 实现只读 validator，不接入运行时绑定。
3. 实现 exporter 与 deterministic roundtrip。
4. 接入 canonical Weight identity 与原子 binding。
5. 再加入量化格式与 PyTorch/ONNX 转换工具。

在第 4 步完成前，此规范仍是设计草案，不得对外宣称 Weight ABI 已支持。
