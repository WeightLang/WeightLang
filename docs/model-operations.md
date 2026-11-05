# 多范式模型操作

> **状态：本地指定范围已验收，尚未发布。** 操作语法和 Rust API 合同已冻结；以下验收记录仅适用于明确列出的本地范围。

## 目标

在现有模型 forward 与梯度训练路径之外，允许一个模型定义包含多个具名、typed 的操作。操作可读取模型状态和参数，并通过显式事务请求更新状态。操作的 effects 从编译后的读写指令静态推导，包含各分支可能访问的状态和写入的参数；它不是单次执行实际走过路径的日志，也不依赖调用者自报。

Rust 应用负责调度操作、外部搜索循环、数据准备与 I/O。机器级唯一 `wgtime` 服务是可变模型状态的唯一 owner；Rust 端持有服务句柄并发起调用。该模型可以没有 loss，也可以不执行 backprop。首期不支持跨状态求导；遇到这类求导需求应明确拒绝。

本设计不引入通用调度平台或新的算法内建操作。

## 需保持的架构边界

以下对象有不同生命周期和含义，不能混为一谈：

| 对象 | 含义 | 不隐式发生的事 |
| --- | --- | --- |
| 模型定义 | 可复用的静态结构、操作与参数 schema、初始化说明 | 不持有活跃实例状态，不初始化、不执行、不训练、不创建制品 |
| 初始化实例 | 服务中按定义显式创建的运行对象 | 定义解析或加载本身不创建实例 |
| 参数 | 模型可训练或可显式绑定的数值；可从权重制品加载或保存 | 参数不是运行状态；权重文件不保存操作状态 |
| 状态 | 服务实例持有、可由操作显式更新的可变运行数据 | 不因保存或加载普通权重而隐式重置或恢复 |
| 优化器 | 可选的梯度训练状态 | 无 loss 或不调用梯度训练的模型不要求拥有优化器 |

普通权重与含优化器状态的 checkpoint 仍是不同制品。状态的持久化格式、保存与恢复流程尚未确定；不得借用权重格式暗示状态已被保存。

## 冻结的 WGT 操作语法

以下语法与接口已实现并通过下述本地指定范围验证，不构成已发布能力承诺。

- 模型可声明多个具名操作。每个入口参数都必须写明类型；支持多个参数和输入。
- 操作参数与返回值仅支持 `f32[N]`，其中 `1 <= N <= 16384`，各输入宽度总和不超过 16384。不支持其他 dtype 或 rank 合同。
- typed operations 不能与根级 input/output/target、子模型声明或 loss 混用，不支持操作互调。
- 返回类型省略表示 void；无参数、无返回值的 `fn reset() {}` 是合法形态。
- 状态使用 `state name: f32[N] = [finite, ...]` 声明，初始化值必须为有限数值；声明末尾分号可选。
- `<-` 左侧只能是裸参数名或状态名，右侧为计算新值的张量表达式；更新语句必须以分号结尾。写入后读取状态应观察到本次事务中的候选值。
- 参数名与状态名冲突时拒绝定义。
- 所有操作共享同一份实例参数与状态存储；操作调用的可变状态由 `wgtime` 唯一持有。
- `ModelDefinition.interface()` 返回 `Option<&ModelInterface>`；`operations()` 返回 `&[OperationSchema]`；`states()` 返回 `&[StateSchema]`。
- 实例 `Model.interface()` 同样返回 `Option<&ModelInterface>`，`operations()` 返回 `&[OperationSchema]`。`call(&self, name: &str, inputs: &[Vec<f32>]) -> Result<Option<Vec<f32>>>` 按操作名调用，`None` 表示 void。
- `parameters()` 与 `state()` 均返回 `Result<Vec<ParameterCandidate>>`。`ParameterCandidate` 是 `(String, Vec<usize>, Vec<f32>)`，分别表示名称、shape 和值。参数与状态使用不同名字空间，定义时禁止两者重名。
- `update_parameters(&self, candidates: &[ParameterCandidate]) -> Result<()>` 提交参数候选。候选更新成功会清空既有 forward optimizer；后续训练须通过 `load_weights` 显式选择 optimizer。typed operations 不创建或使用 optimizer。
- `train_with_policy(&self, inputs: &[Vec<f32>], targets: Targets<'_>, options: TrainOptions, policy: DifferentiationPolicy) -> Result<f32>` 是既有 forward 训练路径的一次服务请求；策略只对该请求生效，不作为持久模型设置。策略检查与训练在同一个服务请求中执行。相关类型见 [Rust 后端](native-backend.md)。
- 服务响应可带 revision；revision 仅是服务响应信息。接口没有 CAS，也不提供状态 snapshot。Rust 调度应避免“读取、外部评估、写入”之间并发竞争；不能依赖过期候选检测来解决竞争。

## 操作与更新合同

- 更新形成事务候选；操作整体成功且候选通过校验后才提交。校验失败、预算超限或执行错误必须使该次调用无更新：参数、状态及相关计数均保持调用前值。
- 一次调用在服务中完成候选计算、校验和提交，对外呈现原子结果。这不使跨多次请求的外部搜索循环成为事务，也不提供 revision 冲突检测。
- 操作 schema 列出可能读取、写入的状态和可能写入的参数。服务验证调用所属实例；更新目标受冻结、形状与取值域约束。
- `reset` 是显式、可调用的操作。其具体重置范围由模型的 `fn reset() {}` 实现决定；服务应提供原子调用语义。
- 不同实例的可变状态相互隔离。定义可复用不意味着实例共享状态；不提供状态克隆或会话恢复 API。
- 参数候选可由外部搜索产生，并由 `update_parameters` 显式提交；候选包含名称、shape 与值。服务校验候选并原子更新参数。接口不提供 CAS 或状态 snapshot，也不承诺拒绝过期候选；Rust 调度负责避免读取参数、外部评估和提交写入之间的竞争。
- 参数候选拒绝 frozen/probability 参数；bool/int 候选必须严格离散。候选更新失败保持参数和优化器原状态。
- 从普通权重加载参数应保持实例状态不变。权重不包含状态，参数名与状态名分离。成功提交参数候选会清空既有 forward optimizer；重新训练须经 `load_weights` 显式选择 optimizer。typed operations 不使用 optimizer。
- 不需要 loss 或 backprop 的模型可以通过操作演化状态，或接收并提交外部搜索得到的参数候选。typed operations 不支持 AD；跨状态求导明确拒绝。

## 无 loss 模型示例

以下完整独立示例已由本地 checker 抽取并通过定义解析检查；解析检查本身不代表执行验证。

```wgt
wgt 0.5
model Counter
state count: f32[1] = [0]

fn advance(step: f32[1]) {
    count <- count + step;
}

fn observe() -> f32[1] {
    return count;
}

fn reset() {
    count <- [0];
}
```

该模型无 loss、无优化器、无 backprop。Rust 应用调度 `advance`、`observe` 与 `reset`；服务持有实例状态，并保证每次操作原子提交。状态演化、reset 与实例隔离另由下述 core 和真实服务测试覆盖。


## 梯度策略兼容边界

自动微分（AD）首期仅适用于既有 `forward` 训练路径；typed operations 全部不支持 AD，即使操作只读参数或不写状态也不例外。`train_with_policy` 只为既有 forward 训练路径选择 `DifferentiationPolicy::{ContinuousOnly, StraightThrough}`：`ContinuousOnly` 保守检查整个模型，只要存在任意 `learn bool` 或 `learn int` 参数就拒绝，即使该参数未出现在当前执行路径；`StraightThrough` 显式允许这些参数并采用 STE。旧 `train` 保留兼容行为，默认使用 StraightThrough。跨状态求导拒绝。这些策略与兼容行为已纳入下述本地定向测试，不扩展 typed operations 的训练能力。

## 验收状态

**本地已通过的指定范围**

- wgtime-core：7 suites，31 tests 通过。
- operations 覆盖写入后无限循环触发预算失败并回滚；两次 state 写后读及参数混合更新成功；零 storage 纯操作；void-only 操作；混合 typed 与 local `<-` 被拒绝。
- typed import roundtrip：1/1 通过。
- 常规 language 与 model：16 tests 通过、8 ignored；service 覆盖由单独的 service 测试执行。
- 最终 daemon 重建后真实 IPC：5/5 通过，包括 operations 1、既有 service 1、CNN 1D/2D/3D 共 3 项。
- `cargo check --workspace --all-targets` 通过，保留 4 条既有 warning；typed `model!` 宏 fixture 编译通过。
- 完整示例定义解析：活跃语法导览 T01、抽取的 T02/T03/T04/T05/T09/T10/T11/T12/T13、本文 Counter 及编辑器 model_operations fixture 通过。这些是定义解析检查，不是新增执行测试。
- 编辑器：20 corpus、57 conformance、16 WGT fixture 及 query/XML/Vim/Neovim 检查通过；VSCode 测试 11 文件通过，生成文件一致性检查通过。

这些是本地指定范围的验收结果，不表示全仓所有能力已验收或已发布。可重复的行为断言见 [core operations 测试](../crates/wgtime-core/tests/operations.rs) 与 [真实服务 operations 测试](../crates/wgt-model/tests/operations.rs)；后者需显式连接测试服务。

## 行为边界与尚未承诺事项

- typed operations 不支持 AD；AD 仅适用于既有 forward 训练路径。
- 没有状态 checkpoint、状态持久化或服务重启后的状态恢复保证；普通 safetensors 仅保存参数。
- 不提供 CAS 或状态 snapshot；revision 仅为 service 响应信息。Rust 调度应避免读取、外部评估与写入之间的竞争。
- 不承诺服务失联后的 exactly-once 执行语义。
