# Rust 主程序与 WGT 模型契约

架构类别覆盖及逐项验收计划见[模型架构覆盖矩阵](architecture-coverage.md)；该矩阵是设计清单，不构成当前语法或能力支持承诺。

Rust 控制应用、数据准备、I/O、调度和训练循环；WGT 只描述模型结构、模型内部计算及可选的可微 loss。Rust 库提供内嵌源码、编译期文件和运行期文件三个入口，共用一个模型定义类型和解析器。模型执行、参数与优化器状态由机器级唯一 `wgtime` 服务持有；应用只持 IPC 连接和实例句柄，连接失败明确报错，不回退到进程内核心。

## 当前实现与本轮目标

当前源码已拆成 `wgt-language` 静态解析库、`wgtime-core` 执行库、`wgtime` IPC 协议及客户端、`wgtime-control` 脚本客户端和 `wgtime-service` 服务。生产 CLI 不依赖 core，旧脚本控制路径仍保留在源码中，不能将其视为新模型 API。下文区分已实现的静态定义/宏与正在接入验收的 Rust 客户端接口；均不是已发布能力，验证状态见下文。

可复用入口是 `wgt_language::source::{split,graph}`、`adaptive::{parse,parse_with_functions}` 和共用函数 AST。静态 lowering 的参数保存初始化 recipe 或未绑定标记，`data` 在需要物化时才填充；兼容 IR 仍可能带显式数据。`ModelDefinition` 把结构、参数 schema 和初始化说明，与服务中的已初始化/已加载实例区分开。检查、编译及导出不抽样，不执行 recipe；完整权重绑定跳过参数配方。

## 服务端 compiled forward

新 Rust 模型入口在服务初始化时把固定 shape 模型 IR 生成为 C，再调用本机 `cc` 编译共享库。推理通过 `wgt_forward_v2` 单一原生入口执行固定控制流，不逐条分派 IR；这是服务侧即时 AOT，不是 Rust 宏直接生成机器码。初始化需要 Linux、可用 C 编译器和可写磁盘缓存；缺少工具、编译失败或超时明确报错，不回退 VM。

入口 ABI 接收输入、当前参数指针表、工作区、输出、预算与可选 trace hook；参数独立于机器码，每次调用提供当前权重，因此训练或重新绑定权重不会复用旧值。代码仅由校验后的 IR、shape、有限常量生成，不拼接用户标识符或任意源码。编译使用独立私有临时目录，30 秒超时终止并回收子进程；已加载库以完整生成源码/ABI为进程内缓存键，最后实例释放后卸载并清理临时文件。默认缓存根为 `.cache/weightlang/compiled-forward`，可通过 `WGTIME_COMPILE_CACHE` 指定磁盘路径。

编译入口覆盖当前 [tensor_contracts.json](../language/tensor_contracts.json) 的完整 44 指令集合，包括普通 `Write`、工作内存分配/读写/释放与 `ResourceState`。推理保留 gate/route 端点惰性求值；训练前向使用相同原生入口并启用 trace hook，记录实际张量、参数重绑定与工作内存版本供既有反向 tape 使用。反向传播仍遍历梯度 tape，不等同于 compiled backward。普通 Rust `Model::forward` 成功时提交显式写入，失败整体回滚；训练前向只使用快照，不提交临时写入。typed 操作继续经过无 AD 的事务执行路径，见下文。完整清单、使用方式和验收命令见 [Rust SDK](rust-sdk.md)。

固定 shape 张量几何合同包含局部 `f32[axis=size,...]` 类型注解、命名轴、`reshape`、rank-2 `transpose` / `.T`、rank-2 `matmul` / `@`、rank-3 等 batch 的 `batch_matmul`、通用 `permute`、多轴 `softmax` 和有限有符号字面量 `scale`。注解可把相同元素数的一维向量直接绑定为多轴 Shape；命名轴在编译期检查收缩/归约/置换关系。实际使用这些几何算子时使用 adaptive IR v3；旧 adaptive IR v1/v2 不因 AST 中出现新 intrinsic 而全定义升级。参数 `read` 与裸参数引用始终为 flat vector；局部类型注解或 reshape 显式确定矩阵几何。原生 CPU interpreter 使用 base/offset/shape/strides TensorView；Matmul 与 BatchMatmul 直接读取 strided 输入，不复制数据。其他算子按需物化；AOT compiled-forward 当前仍采用物化回退。既有 Rust forward 服务入口只接受单个 f32 向量；forward 的矩阵结果须 reshape 或以同元素数向量注解满足返回接口。typed 操作多输入的本地验收范围见下文。完整签名、轴范围与 attention fixture 见[固定 shape 张量几何](tensor-geometry.md)。指定范围的本地验证已通过：core 7 suites / 30 tests、真实 service 集成一项及 macro fixture 编译一项。service 的 native-cpu attention 路径训练 100 步，MSE 从 0.046146184 降至 0.00000076705595，并覆盖权重 roundtrip、同 SGD 续训及资源释放。debug 单机 1000 次 evaluate 为 2.033213 ms（约 2.03 μs/次，含返回 Vec）；managed tensor 计量 1128 B（参数 96 B、tape 1056 B），charged work 98。这些是单次运行记录，不是 benchmark 优势、进程 RSS 或真实峰值内存。kernel 使用朴素 f64 累加而非 BLAS，并复用 AxisGroups；let k=q 已验证输出和梯度相同且 work 更少。当前无全局 CSE 或活跃区间复用，compiled workspace 各 register 常驻且 session 保留 reference buffers，不宣称内存最优。核心补充 IR 安全校验后仍需回归；未验证完整 Transformer、多头 attention、动态 batch 或 GPU 路径。

`compiled_instruction_coverage` 将实际编译执行的指令与目录集合对应；`compiled_semantics` 检查状态、工作内存、梯度和预算。既有 `compiled_forward`、`compiled_training`、`compiled_control_flow` 继续检查编译加载、活参数和控制流。真实服务用例与生成 Rust 消费者另行执行，具体命令见 [Rust SDK 验证](rust-sdk.md#验证与边界)。

## 共享定义与 crate 边界

| crate | 职责与依赖 |
|---|---|
| `wgt-language` | 新增 `model` 模块；拥有 `ModelDefinition`、接口和参数 schema、初始化说明、模型专用解析与 lowering；不依赖服务或 core，不创建会话 |
| `wgt-model`（新增） | Rust 公共库；重导出共享定义，提供 `Client`、`Model`、加载/初始化/forward/训练 API；仅依赖 language、轻 IPC 与文件传输，不依赖 core |
| `wgt-model-macros`（新增） | proc-macro；仅依赖 language 和宏处理库，复用模型解析器并产生诊断与定义表达式；不连服务、不读权重、不初始化模型 |
| `wgtime` / `wgtime-control` | wire、鉴权连接、有界 FD 与模型 lease 传输；复用底层设施，不要求公共模型 API 构造脚本 `Runtime`、`Env` 或动态 `Value` |
| `wgtime-core` / `wgtime-service` | 仅服务生产路径持有核心、执行计划、参数、优化器和设备资源；core 单向依赖 language |

`ModelDefinition` 是可重复使用的静态值，包含模型名、参数 schema 和初始化说明等；旧模型具有单输入/输出合同和可选 loss target，typed 操作模型则具有操作与状态 schema。字段通过受校验构造器建立；定义不含已训练状态、优化器状态或活跃设备资源，复制定义不复制服务模型。旧 forward 只接受单个 f32 向量，保留 logits/probabilities 输出类型；typed 操作的多输入合同已通过下述本地指定范围验收，不表示支持 batch forward、cache 或任意 dtype。

`ModelDefinition::interface()` 现返回 `Option<&ModelInterface>`：旧 forward/顺序模型为 `Some`，typed 操作模型为 `None`；旧调用方须显式处理缺失接口。typed 操作的执行及服务 API 已通过下述本地定向测试，不能以定义解析推断任意模型均可执行。

`ModelDefinition::parse(source: &str) -> Result<Self>` 与 `ModelDefinition::from_file(path: impl AsRef<Path>) -> Result<Self>` 已在 language 库实现并通过定点测试。二者只解析和检查；字符串入口支持 `std/activation`、`std/loss` 纯张量逻辑模块，文件入口另支持相对文件的 `./` 计算模块及嵌套导入。解析将依赖闭合为自包含源码；服务端可重新验证而不读取客户端目录。不提供宿主模块或包注册表查找。新增导入行为的定点验收状态单独记录，不能由原有无 import 测试推断。源码上限 1 MiB，单个参数和一般模型总参数上限 16,777,216 个元素（顺序 layer 源码另限 4,000,000），shape 为 1..8 个正整数轴；乘法溢出、长度不匹配和非有限初始化值均拒绝。初始化表达式通过共享函数 parser 转成纯配方；定义检查不执行张量生成器。源码可带显式列表或无初值声明，尚未绑定的参数不会阻止编译，但会阻止未提供完整权重的实例化。模型专用入口复用现有 parser，但拒绝 `fn main`、agent/environment、应用 I/O、tasks、训练/保存语句和宿主控制函数；允许模型 forward/loss 及现 lowering 能接受的纯张量辅助函数。不得仅解析成功就声称 forward 已执行。

## 多入口操作

本地指定范围已验收，尚未发布：core 7 个 suite 共 31 项通过，含预算写后回滚、两次 state 写后读与参数混合更新、零存储及仅 void 入口、非法混合参数和局部变量 `<-` 的拒绝；typed import roundtrip 1/1。最终 daemon 重建后真实 IPC 5/5 通过（operations 1、既有 service 1、CNN 1D/2D/3D 共 3 项），覆盖状态演化、实例隔离、reset、混合更新回滚、外部搜索、权重与状态分离、策略及优化器处理。language 与 model 常规测试 16 项通过、8 项 ignored，相关服务测试单独执行。workspace all-targets 检查、typed `model!` 宏编译和完整示例解析均通过；解析本身不证明执行。完整记录见[模型操作验收状态](model-operations.md#验收状态)。

一份实例的 typed 入口共享参数和独立的状态存储；每个参数为固定 `f32[N]`（N=1..16384），单次入口总输入宽度不超过 16384，返回 `f32[N]` 或 void。入口可无参数，例如 `fn reset() {}`。操作模式不使用根级 input/output/target，不能声明 loss；支持具名操作互调、有状态子模型及实例数组。`state name: f32[N] = RECIPE` 的分号可选，未初始化 state 只可检查定义，实例化时仍须有配方；`name <- expression;` 的目标须是裸状态或参数名且分号必填。

模型定义存在 typed 函数签名或 `state` 声明，或没有 `forward` 但存在无参函数时，进入操作模式。无参且无返回类型标注的普通操作视为 void 入口，不能从该入口返回张量。有返回值的无参入口须显式写返回类型，例如 `fn observe() -> f32[1] { ... }`。纯 tensor helper 须至少有一个无类型标注参数，并遵守纯张量限制；被初始化配方引用的纯 helper 可以无参，不作为 typed 入口。

调用内写后读看到候选值，执行及校验成功后原子提交，失败保持原参数和状态；普通权重只含参数，不保存状态。原生执行不使用 VM，typed 操作不支持 AD。参数候选拒绝 frozen/probability 参数，bool/int 候选必须严格离散。候选更新不提供 CAS（比较版本后提交）或状态 snapshot，revision 仅在服务响应中返回，不能据此承诺拒绝过期候选。`update_parameters` 成功后清空旧 forward 优化器，失败保持原状态；需要继续训练时须通过 `load_weights` 显式选回优化器。typed 操作不支持优化器或 AD，加载权重保留 state。`train_with_policy` 以单个 `TrainStepWithPolicy` 请求原子完成策略检查与训练，不持久设置策略。旧 forward 的训练策略边界见[模型操作](model-operations.md)。

当前 Rust facade 提供以下接口，已通过上述本地指定范围验证，尚未发布；`Result` 为 `anyhow::Result`：

```rust,ignore
pub type ParameterCandidate = (String, Vec<usize>, Vec<f32>);
Model::operations(&self) -> &[OperationSchema]
Model::call(&self, name: &str, inputs: &[Vec<f32>]) -> Result<Option<Vec<f32>>>
Model::parameters(&self) -> Result<Vec<ParameterCandidate>>
Model::state(&self) -> Result<Vec<ParameterCandidate>>
Model::update_parameters(&self, candidates: &[ParameterCandidate]) -> Result<()>
Model::train_with_policy(&self, inputs: &[Vec<f32>], targets: Targets<'_>, options: TrainOptions, policy: DifferentiationPolicy) -> Result<f32>
```

候选元组依次为名称、shape 和扁平 f32 值。`call` 按入口 schema 接收多个向量，`None` 表示 void；`parameters` 和 `state` 分别读取参数与状态。`train_with_policy` 仅用于旧 forward 训练路径，inputs 是单向量样本的批次，并非多输入操作训练。

静态 schema 的公开字段：`ArgumentSchema { name: String, width: usize }`；`OperationSchema { name: String, inputs: Vec<ArgumentSchema>, output: Option<usize>, effects: Effects }`；`StateSchema { name: String, shape: Vec<usize> }`。`Effects` 的 `reads_state`、`writes_state`、`writes_parameters` 均为 `Vec<String>`，表示静态推导的可能访问集合，不是调用日志。`ModelDefinition::states()` 返回 `&[StateSchema]`。旧接口 `ModelInterface` 的 `input`、`output` 为 `usize`，`target` 为 `Option<usize>`，`output_type` 为 `wgt_language::categorical::VectorType`；typed 定义没有此接口，不应从其缺失推断模型无法执行。

## 三种 Rust 入口

以下使用当前公共接口；三入口真实服务测试及宏 fixture 编译已有本地验证，范围见上文及[模型操作验收状态](model-operations.md#验收状态)，不表示所有示意路径均已执行；不恢复已删除的 examples：

```rust,ignore
use wgt::{Client, InstanceOptions, ModelDefinition, model, include_model};

let inline = model!(r#"wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
"#);
let compiled_file = include_model!("fixtures/scale/model.wgt");
let runtime_file = ModelDefinition::from_file("models/scale/model.wgt")?;
let client = Client::connect()?;
let model = client.initialize(&inline, InstanceOptions { memory_bytes: 1024 * 1024, optimizer: None })?;
let output = model.forward(&[3.0])?;
assert_eq!(output, vec![6.0]);
```

`model!` 首期接受 Rust 字符串字面量，避免 Rust token 重排改变 WGT 换行/注释语义；token 形式不是首批承诺。`include_model!` 接受相对消费 crate `CARGO_MANIFEST_DIR` 的路径，嵌入完整闭合源码，并对根文件和每个传递依赖生成 `include_str!` 引用以跟踪重编译；产物运行不再查找模型文件。两个宏编译时复用同一解析器验证，展开式首次构造时仍调用纯 `ModelDefinition::parse`，不是预生成静态 IR；均不在编译期间连接服务、执行 forward 或创建制品。宏默认使用 `::wgt` 公共库路径，依赖改名可用 `crate = alias` 显式指定。运行期文件入口允许部署后更换定义，解析完成后不再依赖原文件。模块只提供不可变 let/return 的纯张量计算，不拥有权重或应用状态；循环、路径越界、重复别名和宿主模块明确拒绝，完整闭包限额与标准函数合同见[模型内部标准库](standard-library.md)。

## 显式生命周期与训练

CLI 还提供 `wgt compile model.wgt --target rust -o src/model.rs`，输出自包含架构模块供 Rust `include!` 使用。该入口与上述定义入口共享语义，公开 `SOURCE` 和返回 `wgt::Result<ModelDefinition>` 的 `definition()`；完整 imports 随源码嵌入。它不创建运行实例，详见 [Rust 导出入口](rust-sdk.md#导出入口)。

| 当前接口 | 必须满足的行为 |
|---|---|
| `Client::connect() -> Result<Client>`；`connect_to(&Path)` | 连接已有服务并验证当前 UID/协议；不隐式启动另一个执行核心 |
| `Client::initialize(&ModelDefinition, InstanceOptions) -> Result<Model>` | 执行声明的参数与 state 配方；未绑定项报错；options 含字节预算和可选 SGD/Adam，缺省预算 64 MiB、无优化器 |
| `Client::load_weights(&ModelDefinition, &Path, InstanceOptions) -> Result<Model>` | 完整 safetensors 直接构造新实例，跳过所有参数配方；首次绑定允许外部 frozen，state 仍按配方初始化 |
| `Client::load_snapshot(&ModelDefinition, &Path, Option<&str>, InstanceOptions) -> Result<Model>` | 读取 manifest 快照，可固定 revision；服务校验同一 payload 摘要后直接绑定，参数规则同上 |
| `Model::load_weights(&Path, Option<Optimizer>) -> Result<()>` | 向既有实例绑定完整 F32 权重，检查名称、shape、值域及预算，frozen 必须等于当前值；保留句柄与 state，成功重置优化器，失败保持原实例 |
| `Model::snapshot() -> Result<weights::WeightSet>` | 捕获不可变参数值与原始 learn shadow；不含 state 或优化器 |
| `Model::forward(&self, input: &[f32]) -> Result<Vec<f32>>` | 服务端执行并校验输入合同；成功提交显式 forward 写入，失败回滚；返回拥有数据的结果，不承诺跨 IPC 的零分配或借用 core 缓冲 |
| 优化器选择 | 当前通过初始化或权重绑定显式选择 SGD/Adam，未提供单独 `configure_optimizer`；不得把加载权重当作恢复优化器 |
| `Model::train(&self, inputs, Targets, TrainOptions) -> Result<f32>` | Rust 发起单步批训练，返回 loss；options 显式选择 loss、正有限学习率和可选正有限裁剪；服务执行原子更新，Rust 控制循环；不自动训练 |
| `Model::save_weights(&Path) -> Result<()>` | 显式写入 safetensors，默认拒绝覆盖；本轮公共库未提供 checkpoint 保存/恢复 API，不把权重互通称为 PyTorch 完整续训兼容 |

`Model` 是绑定连接的 lease，不提供无说明的深拷贝；显式关闭和连接断开释放实例，服务重启使旧句柄失效。首期不自动重连并重建有状态实例。训练输入首期对应已有 `Vec<Vec<f32>>` 批，目标通过 typed enum 区分 MSE 向量、交叉熵类别和声明 loss target，公共 API 不要求用户手填 JSON。无优化器实例调用训练必须报错；失败更新不能改变参数、优化器或步数。是否共享/克隆模型须后续单独设计，不能把定义复用等同于可变状态共享。

服务使用 `InitializeDefinition` 进行配方初始化，使用 `InitializeWeightsFd` 直接从定义与完整权重创建实例；`BindWeightsFd` 只更新既有实例。参数与 state 均先校验，再注册新实例，失败不留下可用句柄。所有者校验、预算与失联清理仍适用；服务重新解析定义并检查 payload，不能信任宏已检查过来源。独立 optimizer 配置仍未提供。

## 权重快照与离线计划

`wgt::weights::{Tensor, WeightSet, WeightRef, Plan}` 在宿主中操作完整参数值，不创建服务实例。`WeightSet` 是有限 F32 的不可变快照，revision 固定 safetensors 字节；`WeightRef` 引用 revision 与完整参数名。`Plan::from_base` 显式继承完整匹配快照，再用 `replace` 选择例外；新架构用 `Plan::new` 显式复制、变换和初始化每个目标。

`preview` 不执行回调，缺项在 build 调用任何初始化器前失败；`initialize_declared` 只选择指定参数的 WGT 配方。快照保存 learn 的连续 shadow，payload 上限 64 MiB、manifest 上限 1 MiB；新增快照不覆盖原目录。Rust tensor 操作、来源记录、迁移与 restore 完整示例见[权重快照与迁移](weights.md)。当前实现不提供自动重命名、部分加载或稳定参数身份协议。

## 实现与验收分工

1. 前端 owner：新增 `crates/wgt-language/src/model.rs`，修改其 `lib.rs`，必要时只抽取 `source.rs`、`adaptive/source.rs` 的静态验证；新增 `tests/model_definition.rs`。覆盖拒绝应用逻辑、三入口共用语义、路径诊断及定义检查无执行。
2. 服务 owner：修改 `crates/wgtime/src/lib.rs`、`crates/wgtime-service/src/main.rs` 与必要 core 初始化/optimizer API；新增服务合同测试，覆盖无优化器推理、原子权重加载、身份/预算、错误状态不变和断连释放。保留已有未提交实现。
3. Rust API owner：新增 `crates/wgt-model/{Cargo.toml,src/lib.rs,tests/model_api.rs}`，公共类型/lease/typed train options；必要时抽取 control 的纯 FD 传输，不能通过构造脚本 Runtime 模拟 API。模型与应用循环测试使用最小 fixture。
4. 宏 owner：新增 `crates/wgt-model-macros/{Cargo.toml,src/lib.rs}` 及编译正负 fixture；使用 language API，禁止复制 parser。Rust API 通过可选 re-export 暴露宏，宏展开引用消费方的公共库路径并处理依赖改名。
5. 集成 owner：最后统一 workspace/lock、本文/中文参考/技能边界及本地定点验证；分别证明模型库和生成宏消费者的 normal/build 依赖树不含 core。三种入口均须验证真实 forward；加载检查无训练、显式训练与 checkpoint 分别验证，不以旧通用脚本测试代替。

以下章节描述既有 core 的 IR 与内存实现，供服务实现复用，不是让 Rust 应用直接链接执行核心的公共入口。

```sh
cargo test -p wgtime-core
cargo test -p wgt-language
```

## 原生 IR v1

顶层字段：`version=1`、正整数 `input`、`parameters`、非空 `layers`。
每个参数有唯一 `name`、非空正整数 `shape` 和下列一种定义；逻辑数值为 f32：

| kind | 字段 | 契约 |
|---|---|---|
| resident | data | 扁平有限数值，长度等于 shape 元素数；会话拥有数据 |
| file | path, offset | offset 为 4 字节对齐的字节偏移；文件保存 little-endian f32 |
| resident_file | path, offset | 对齐的 little-endian f32 文件范围，准备阶段一次加载至常驻存储 |
| packed | storage, group, data, scales | storage 为 int4/int8/f16/bf16，group 为 1..4096；data 是编码字节，scales 是 F32 列表（浮点存储为空）；仅二维矩阵 |
| packed_file | storage, group, path, offset, scales_offset | 准备阶段按 shape 推导编码/scale 字节数，从文件加载紧凑常驻权重，偏移单位字节 |
| view | source, start | 引用先前参数的连续扁平区间；start 是元素偏移，区间长度由本参数 shape 决定 |
| add | base, delta | 两者都引用先前参数且完整 shape 相同；结果是表达式，不是独立参数副本 |

引用必须指向此前声明，因此自引用、循环和前向引用在 prepare 被拒绝。
表达式深度上限 32。view 可以改变形状，但只支持连续扁平片段，不支持任意步长切片。
add 不支持广播。参数加法和 view 是 native IR 能力，**尚未增加 WGT 的 delta/bind 语法**。
WGT 0.5 模型构造器当前不支持 share 声明。

linear 节点字段为 `op=linear, weight=NAME, bias=NAME|null`；bias 可以省略，表示无偏置。
权重必须为 `[out,current_width]`，bias 必须为 `[out]`。
其余节点只有 op。严格拒绝未知算子和字段。
Rust 宿主的参数保存与恢复见[权重快照](weights.md)；模型定义不隐式创建制品。这里的 IR 存储形式不增加源码语法或 checkpoint 合同。

## 内存与生命周期契约

`Session::prepare(model, root, budget)` 检查契约、文件长度和预算，生成顺序执行计划，
打开参数文件并分配两个激活缓冲区和一个参数工作区。常驻参数独立保存；view/add
没有永久数据副本。文件参数每次用到时同步读取，复用工作区，不把整个文件读进堆。
组合权重临时物化在工作区中，计算结束后该空间可复用于下一层。

预算单位为字节，公式为：

```text
managed_bytes = resident_bytes + activation_bytes + workspace_bytes
activation_bytes = 2 * 最大层宽度 * sizeof(f32)
workspace_bytes = 最大单层所需工作区；直接 packed linear 只需 bias 元素数 * 4，其他为 (weight + bias) 元素数 * 4
```

预算是受管理张量存储上限，**不是进程 RSS 上限**：不包含解析期间的 JSON、IR 元数据、
OS 文件页缓存、进程/管道开销，以及 I/O 的 4 KiB 栈缓冲区。大模型控制文件应使用 file
源，避免在 JSON 中重复嵌入大量参数。会话不承诺锁定物理内存或没有缺页。
文件必须在会话期间保持不变；当前没有文件摘要锁定或磁盘写回。常驻参数的显式读写与会话 revision 见下节。
prepare 检查长度；执行时检测短读和文件中非有限值，不能静态保证磁盘永远可读。

`Session::run(&mut self, input: &[f32]) -> Result<&[f32], &'static str>` 的成功执行
不进行堆分配，也不依赖垃圾回收。结果借用会话存储；Rust 借用规则阻止仍使用旧结果时
再次可变借用会话。drop 确定性释放权重、缓冲区和文件。
测试通过全局分配计数器验证内存和文件参数路径，并非仅观察 RSS 或运行一次 warmup。

## GPU 存储和能力边界

CPU 的预算公式保持不变。GPU 总预算为 `host_managed_bytes + gpu_buffer_bytes`，
后者包含不同 weight/bias 组合的常驻有效权重、文件表达式共享暂存区（至少 4 字节）、
两个激活缓冲、回读缓冲和每算子 32 字节控制参数。纯常驻 view/add 上传一次，
显式修改参数后刷新缓存；依赖文件的表达式每次读取并上传，提交顺序防止暂存覆盖。
GPU 分配失败、超出设备 buffer/dispatch 限制均拒绝候选。

核显/统一内存设备可执行，但当前仍创建独立主机和 GPU 缓冲，预算分别计费；
设备 memory_model 只是报告信息，不代表已实现共享物理分配或零拷贝。
预算不包含驱动、编译、校准输入、上传暂存和系统开销，不能作为硬性 RSS/显存峰值上限。

本文描述的 NativeSession 推理入口不提供自动求导、优化器、KV cache、通用图、GPU batch 或异步文件预取。
CPU 常驻 linear 使用 AVX2/FMA（不支持时标量回退），直接读取常驻权重免去逐次矩阵拷贝。
GPU linear 使用工作组并行归约，文件 I/O 尚未与计算重叠。
不能把此阶段描述为完整 PyTorch 替代。所有能力为本地源码实现，尚未发布。

验证执行核心：`cargo test -p wgtime-core`。GPU 能力须在目标设备单独验证。
