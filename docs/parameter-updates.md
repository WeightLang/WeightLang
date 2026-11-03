# 可编程参数更新（本地 0.3，尚未发布）

更新算法由普通张量计算组成，不内置特定优化器、更新模型或接受策略。
WGT 既有 input/output、matmul/add/mul、block 和显式状态端口可以描述更新规则；
Python 负责数据、反馈、迭代与是否写回。这里新增宿主参数操作接口，不增加 WGT
关键字，不修改 load 的构造副作用，不在 forward 中注入任意 Python。

## Torch 模块：读取、临时绑定、写入

从 `weightlang` 导入以下接口；用 `load(path)` 获取模块。
也可用于 `load_artifact(...).module` 与独立导出的 Torch 模块。

| 完整签名 | 行为 |
|---|---|
| `read_parameters(module, names=None) -> dict[str, Tensor]` | names 是参数路径序列；默认按 named_parameters 顺序每个共享对象只读取一次。返回 detach 后的独立副本，保留 dtype/device，不会修改源参数。重复或未知路径拒绝。 |
| `call_with_parameters(module, parameters, *args, **kwargs)` | parameters 是部分参数路径到张量的映射；未指定参数保留原值，返回模型原有张量/具名结果。暂时绑定参数并保持共享别名，保留到传入张量的梯度；不要求反向传播。 |
| `write_parameters(module, parameters) -> None` | 校验完整部分映射，再将快照复制进原有 Parameter；共享关系、对象身份、freeze 和优化器引用不变。空映射无操作。 |

映射值必须为稠密张量，形状、dtype、device 精确匹配且有限；无隐式广播、转换或设备搬运。
一批绑定/写入中同一共享对象只能指定一个路径，可以选其任一别名；不能同时指定两个别名，
即使数值相同。读取显式别名得到独立快照。`requires_grad=False` 的参数禁止写回；
临时绑定可以预览冻结参数的替代值，但不改变冻结状态或持久权重。

临时绑定当前使用 Torch functional_call，并复制参与执行的全部参数和注册 buffer，
避免 forward 的原地张量修改影响原模型或调用方张量；它不是零复制接口。
不隔离任意 Python 属性、hook、随机数或外部副作用，同一模块的访问必须串行化。
绑定在异常后恢复；自定义权重类型仍对传入主权重执行原有投影，不绕过域约束。

写入先快照全部来源，因此交换两组参数或传入当前参数的视图不会依赖写入顺序。
校验失败时所有参数保持不变；不承诺设备故障的事务回滚。
写入不建立梯度连接，不清空已有 .grad、不重置优化器状态；不要在尚待 backward 的计算
中途写入。纯无反向传播流程可使用 `torch.no_grad()`。要保留更新器的元梯度时，使用临时绑定。
这里的“共享”指同一 Parameter 对象；不支持用户另建、存储相交的多个 Parameter 对象。

更新不会自动保存文件。已有 Artifact 可用 `artifact.save(new_directory)` 保存到新目录，
无训练记录的改动仍标为 edited；不把直接赋值称为已训练。架构、模型制品、训练断点仍分开。

## 原生 runtime：存储读写

`NativeSession` 支持下列 Python 接口。Rust `Session` 和 `ExecutionSession` 提供对应操作，
不需要 Torch/自动求导。Python 包本身仍依赖 Torch，不等于原生执行依赖 Torch。

| 完整签名 | 行为 |
|---|---|
| `session.parameter_info() -> list[dict]` | 按声明顺序返回 name、shape、writable。 |
| `session.read_parameter(name, *, start=0, length=None) -> list[float]` | 读取有效参数的连续扁平区间，默认至结尾；start/length 是元素数。resident/file/view/add 都可读取。 |
| `session.write_parameters(writes) -> int` | writes 为 `{name, values, start=0}` 字典序列；values 是扁平 f32 替换值。返回会话 revision。 |

写入是替换，不是累加；加法、裁剪、门控和低秩更新由调用方计算。
只允许 resident 与最终指向 resident 的连续 view；修改 view 会改变其共享 owner，
所有引用在下一次执行观察到相同新值。add 是只读表达式，应更新明确的 base/delta 来源；
file 仍只读，绝不通过改文件暗中更新会话。不能改变 shape、dtype、绑定关系或文件源。

写入先校验全部名字、边界、有限值、冻结状态以及来源存储上的区间冲突。
同一 owner 上不相交的片段可同时写入；重叠别名拒绝，无先后覆盖语义。
校验失败不修改任何参数且 revision 不变。revision 初始为 0，每批非空写入加 1；
空批不变。单条零长度写入仍需有效名字和边界。Rust 可变借用串行化读写；
Python 会话不提供线程安全保证，需要宿主串行调用。

native IR v1 参数新增可选 `frozen: bool=false`。冻结 view/add 会保守冻结其全部底层
owner，避免绕过别名修改冻结表达式（不是仅冻结其某个切片）。WGT freeze/unfreeze
按既有规则顺序降低到参数属性，import freeze 作为默认；共享目的路径不是独立 owner。
底层冻结传播完成后，parameter_info 报告实际可写性。

GPU 保留 CPU 权重为来源；写入后下一次前向刷新缓存的有效权重，流式参数也重新物化。
不重新选择设备，不重建会话，不增加常驻张量预算；GPU 上传和 JSON 传输仍可能分配。

Rust 签名（两种 session 一致）：

```rust
fn parameter_info(&self) -> Vec<ParameterInfo<'_>>;
fn revision(&self) -> u64;
fn read_parameter(&mut self, name: &str, start: usize, out: &mut [f32]) -> anyhow::Result<()>;
fn write_parameters(&mut self, writes: &[ParameterWrite<'_>]) -> anyhow::Result<()>;
// ParameterWrite { name: &str, start: usize, values: &[f32] }
// ParameterInfo { name: &str, shape: &[usize], writable: bool }
```

Rust CPU read/write 成功路径无堆分配，读取使用调用方缓冲区；读取 I/O 失败可能部分填充 out，
但不修改参数。元数据列表、错误对象、JSON、Python 对象、调用方缓冲区不在零分配或张量预算保证内。

JSON-lines 保留原向量前向协议，新增对象命令：

```json
{"op":"parameters"}
{"op":"read","name":"projection.weight","start":0,"len":2}
{"op":"write","writes":[{"name":"projection.weight","start":0,"values":[2,-1]}]}
```

对象命令返回 `{"ok":...}` 或 `{"error":"..."}`，拒绝命令后会话可继续使用；
len 可省略/null。原向量前向的致命 I/O/协议错误行为不变。

## 可运行示例与范围

- [目标模型](../examples/parameter_updates/model.wgt)、[具名更新程序](../examples/parameter_updates/updater.wgt)、[Torch 运行脚本](../examples/parameter_updates/run.py)：显式权重/反馈/记忆输入，输出候选权重与下一状态；先临时执行再写回，全程 no_grad。
- [原生控制器](../examples/parameter_updates/controller.wgt)、[纯原生前向更新脚本](../examples/parameter_updates/native.py)：两个 Rust 会话，控制器输出决定目标参数变化。

```sh
PYTHONPATH=src python examples/parameter_updates/run.py
cargo build -p weightlang-runtime
WEIGHTLANG_NATIVE=target/debug/wgt-native PYTHONPATH=src python examples/parameter_updates/native.py
```

示例控制器系数与可观测目标反馈由脚本指定，未训练，不能据此声称通用学习或优于反向传播。
原生后端仍仅支持其已有静态向量顺序图；完整多输入更新图用 Torch 模块执行。
没有新增 native 多端口、通用图、动态重新绑定、磁盘写回或原生模型保存。
测试覆盖无自动求导更新、绑定梯度、共享与失败不写入、freeze、权重域、制品/导出往返、
原生 CPU/GPU 缓存一致性及 Rust 分配计数。
