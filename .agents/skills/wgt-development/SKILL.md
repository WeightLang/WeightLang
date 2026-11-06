---
name: wgt-development
description: 编写和验证本仓库当前 WGT 模型定义与 Rust 主程序边界。
---

模型源码入口必须命名为 `model.wgt`。每个独立模型/训练示例一个目录，使用 `<模型或示例名>/model.wgt`，不要创建 `main.wgt`、`train.wgt` 或功能命名的模型入口。`fn main()` 是旧普通脚本的执行入口，纯模型定义不要求它。迁移或移动入口时同步命令、文档链接、测试引用及相对数据/保存路径；标准库模块和制品的 `source.wgt` 快照不改名。

以 docs/reference.zh-CN.md、docs/differentiable-programs.md 和 Rust 实现为准；不是已发布功能承诺。Rust host 集成目标使用 edition 2024；WGT 语言版本仍以实际 parser 声明为准，不能因 Rust edition 将其改写为 WGT 版本。

## Tensor 合同维护

固定 Shape/DType/命名轴和 stride 元数据共用 `wgt_language::tensor`，详见 [Tensor Contract](../../../docs/tensor-contract.md)。源定义和未信任 IR 都经过 `Program::validate_tensor_contracts`；保留 IR v1/v2/v3 的线格式和 f32 执行边界。新增 Instruction 必须同步 `language/tensor_contracts.json`，运行 `node scripts/check_tensor_contracts.cjs --write` 后再运行不带参数的检查，补齐源级正反例、CPU 数值和 VJP 测试。不能把 metadata dtype、可读重叠 view 或制品 graph 原语当作通用执行能力。

源级变换、矩阵乘法、归一化和逐元素规则直接复用 `TensorType`；Instruction 最低版本从合同目录取得，不另建版本名单。`@` 和 `matmul` 的命名收缩轴检查相同，结果保留未收缩轴名，输出名称必须唯一；命名 attention 为查询和键使用不同轴名。batch 轴只在一侧命名时也保留该名称。复制和保形单输入算子保留轴名，同形逐元素运算按位置细化匿名轴并拒绝冲突或重复；同一寄存器的赋值和条件分支必须兼容已有名称。显式 reshape 清除结果名称，新绑定的类型注解重新声明。LayerNorm 仿射向量的已命名轴须匹配归一化轴。维护这些规则时运行 `cargo test -p wgt-language --test tensor_source_contracts`，并保留既有 CPU/VJP 回归。

gate/route 的同形概率张量也参与轴名合并与冲突检查；route 两支必须兼容。只保留既有 `[1]` 标量门例外，不把 `[1,1]` 或其他单元素 shape 扩展成标量广播。

## 架构迁移方向

Rust 应用负责数据准备、I/O、调度、服务连接和训练循环；WGT 与 `std/` 聚焦可复用模型架构及模型内部参数/张量流、子模块、分支、循环和状态，并可定义可微 loss。旧通用 WGT 宿主脚本架构正在退出，不新建宿主控制脚本，也不将其描述为持续支持方向；这不表示现有代码已经删除。

当前 `crates/wgt/src/main.rs::Cmd` 没有内置 run、script 或 test，不能生成 `wgt run` / `wgt test` 操作指引。源码静态检查用 `wgt check`；模型行为通过 Rust 测试与真实服务验证。`weightc compile` 和 `compile-client` 当前共用 `client_compiler`，只接受已迁移的构造；core 解释器和旧 codegen 的支持不能证明该客户端编译器支持。后文普通脚本内置函数仅描述兼容层，不能代替 Rust facade 的 API 契约。

`wgt-model` facade 的 Rust 库名为 `wgt`，当前提供 `model!`、`include_model!` 字符串宏以及 `ModelDefinition::parse`、`ModelDefinition::from_file`。内嵌源码、编译期文件和运行期文件入口共享模型定义解析语义。上述能力已在本地实现并通过定点验证，尚未发布。不要把 Rust 宏形式加入 WGT 语法说明，也不要用解析成功推断 forward 已执行；宏和定义解析只构造/检查定义，不连接服务、不执行模型。

既有 forward 模型入口支持单个 f32 向量；typed 操作多输入已通过下述本地指定范围验收，尚未发布。源码中实现了标准纯计算模块导入及文件入口相对路径、嵌套导入，但模块导入与编译 forward 尚待完整行为验收。不得据此宣称 batch forward、cache 或任意 dtype 支持。模型定义与服务中初始化/加载的实例、训练权重和优化器检查点是不同对象；加载定义不得隐式执行训练或创建运行状态。准确接口及实现状态以 [docs/native-backend.md](../../../docs/native-backend.md) 为权威。

### 多入口操作

本地指定范围已验收：core 7 suite 共 31 项、typed import roundtrip 1/1；覆盖状态演化与隔离、reset、预算写后回滚、两次状态写后读及参数混合更新、零存储及仅 void 操作、混合参数与局部变量 `<-` 拒绝、外部搜索、权重与状态分离、策略及优化器处理。最终 daemon 重建后真实 IPC 5/5（operations 1、既有 service 1、CNN 3）；language 与 model 常规测试 16 项通过、8 项 ignored，相关服务测试单独执行。workspace all-targets、typed `model!` 宏编译和完整示例解析均通过；解析不证明执行，不能说全仓验收或已发布。合同见 [docs/model-operations.md](../../../docs/model-operations.md)。每个 typed 参数为 `f32[N]`，N=1..16384，各参数宽度总和不超过 16384；返回 `f32[N]` 或省略表示 void，`fn reset() {}` 合法。状态声明 `state name: f32[N] = [有限数值, ...]` 可省末尾分号；`name <- expression;` 必须以裸状态或参数名为目标且分号必填。typed 模式不用根级 input/output/target，不支持 loss、子模型或操作互调；typed forward 禁止写入。

模型定义出现 typed 签名或 `state`，或没有 `forward` 但有无参函数时进入操作模式。无参且无返回类型的函数是 void 入口，不是 helper，不能返回张量；无参返回值入口须显式标注如 `fn observe() -> f32[1] { ... }`。纯 helper 须至少有一个无类型标注参数。

同实例各入口共享参数及独立状态，调用内写后读看到候选，成功原子提交、失败保持原值；普通权重不保存状态。服务侧原生执行、无 VM，typed 操作无 AD、无跨状态求导。旧 forward 的 `train_with_policy` 中 ContinuousOnly 遇到模型任意 learn bool/int 即拒绝，StraightThrough 使用 STE，旧 train 保留 STE。`ModelDefinition::interface()` 现为 Option，旧调用须显式处理缺失值；`call`、`operations`、`parameters`、`state`、`update_parameters`、`train_with_policy` 签名见 [Rust 后端](../../../docs/native-backend.md#多入口操作)；策略检查与训练由单个 `TrainStepWithPolicy` 请求原子处理，不持久设置策略。参数候选拒绝 frozen/probability 参数，bool/int 候选必须严格离散。候选更新无 CAS、无状态 snapshot/checkpoint，revision 仅在服务响应中返回，不承诺拒绝过期候选或断线 exactly-once；`update_parameters` 成功清空旧 forward optimizer，失败保持原状态，重新训练需通过 `load_weights` 显式选回。typed 操作无 optimizer，权重加载保留 state。

### 固定 shape 多轴张量几何（行为待验证）

核心合同为 WGT 0.5，相关 adaptive IR 为 v3；旧 IR v1/v2 仅在显式提供一维 `shape=[len]` 时接受，shape 缺失拒绝。局部绑定支持 `let q: f32[sequence=8, hidden=32] = x;`，注解检查 rank/维度并允许从同元素数的一维向量直接重解释；命名轴参与编译期收缩和归约检查。`reshape(x,[d1,...,dn])` 仍可显式重解释，维度为 1..8 个正整数常量，每轴 1..16384，元素总数不超过 16777216，且总元素数须保持不变。`transpose(x)` / `x.T` 只接受 rank-2；`matmul(a,b)` / `a @ b` 为 rank-2 `[M,K] @ [K,N] -> [M,N]`，不广播。`permute(x,[axes])` 支持 rank 1..8 的完整轴排列，可使用整数索引或已声明的轴名；interpreter 以 base/offset/shape/strides 实现零拷贝 view，Matmul/BatchMatmul 直接消费；其他算子按需物化，AOT compiled-forward 当前物化回退。`batch_matmul(a,b)` 支持等 batch 维的 rank-3 `[B,M,K] @ [B,K,N] -> [B,M,N]`，不广播。`softmax(x,axis)` 的 axis 可为非负整数字面量或已声明的轴名，沿整轴独立稳定归一化并使用 f64 累加；原 `softmax(x)` 与 `log_softmax(x)` 仍为 rank-1 形式。`scale(x,literal)` 接受有限有符号数值字面量，不广播并保 shape。既有 forward 服务入口为单个 f32 向量，矩阵结果需 reshape 为向量；typed 操作多输入验收范围见上节。指定范围已本地验证：core 7 suites / 30 tests、真实 service 集成 1 项和 macro fixture 编译 1 项。native-cpu attention 100 步 SGD 的 MSE 从 0.046146184 降至 0.00000076705595，并覆盖权重 roundtrip、同 SGD 续训及资源释放。核心补充 IR 安全校验后须回归；不能据此声称完整 Transformer、多头 attention、动态 batch 或 GPU 路径已验证。完整 attention 模型见 [docs/tensor-geometry.md](../../../docs/tensor-geometry.md) 和 `language/syntax-tour/model.wgt` 的注释 T10。

### 可学习架构参数

顶层模型定义支持标量及固定长度数组 `learn`。core 定点五组共 33/33 通过（adaptive、compiled_forward、control 10、std 5、learn 6），另有 architecture 6/6 和 conv 3/3 通过；真实 service 的 conv 3 项和 architecture 1 项通过，覆盖训练、权重恢复续训和释放；真实 control facade 1/1 已覆盖 native-cpu 子模型、三种 learn 数组、indexed repeat、if、blend 与半平方 loss 的训练、权重回载、续训一致和释放归零；矩阵分支正反例及 5→5.5 已通过。此前 daemon 的标量路径验证不自动扩展到新语法；这是局部验证，不等于全仓验收或已发布。语法和边界见 [docs/learnable-architecture-parameters.md](../../../docs/learnable-architecture-parameters.md)。

`learn` 每项独占一行且以分号结束：float 无界、float 闭区间、float 严格下界、bool 字面量及有界 int；类型名后紧跟 `[N]`（1..16384），例如 `int[2] in 0..=3`；不能写在范围末尾，此语法及错误形式的拒绝均已实现并通过测试，syntax-tour T12 已解析通过。标量初值填充或等长列表逐项初始化。float 在 SGD/Adam 更新后投影；int/bool 使用连续 shadow，硬读取并以 identity STE 反传。读取形状为 f32[1] 或 f32[N]，索引须编译期确定；不能直接作为 linear 原始参数或被 write，`choice` 不支持。

`repeat depth [indexed i] carrying h=initial { body }` 以常量或范围落在 0..=256 的 learn int 作深度，逐轮传递同形状态；indexed i 为零基编译期索引，必须与 carrying 状态名不同。学习深度训练时全部候选计算并以 STE 传梯度，推理跳过未启用的内联候选；一般动态循环不因此可微。`model A { ... }` 定义子模型，根模型仍写 `model Root`；`blocks: A[2];` 中不同索引参数独立、同一索引共享，实例索引只接受编译期常量或 repeat 索引。子模型调用的输入和结果须符合其声明的输入、输出合同；子模型 forward 仅不可变 let 和最终 return，不能嵌套子模型定义或自带 loss。

`blend { p => a; (1-p) => b; }` 权重各在 [0,1]，逐项求和不归一化；gate/route 的门为 [1] 或与值完全同 shape，route 两支也须完全同 shape，仅等长不足。表达式 if 的 learn bool 或 `learn_int >= 整数字面量` 条件使用 STE，其余支持的比较及语句 if 为硬分支。有限字面量乘张量是缩放，`**` 仅支持指数 2；不提供一般张量广播。完整签名及局部验证范围见主参考。

权重互通：`save weights net to "weights.safetensors";` 与 `reload weights net from "weights.safetensors" using "sgd";`；绑定显式构造的源模型，标准safetensors仅有限F32精确name/shape/C-row-major，不猜图/改名/转置/下载。成功保留句柄别名、重置优化器/步数/自定义状态，失败原子保留。view/add只存独立参数；weights不含图或checkpoint训练状态，不宣称torch.load完整兼容。learn 参数权重保存其 shadow，重新加载必须绑定声明相同 learn 元数据的模型定义。示例与限制见docs/file-io.md。
只接受 wgt 0.5，模型、forward、loss 和 main 可在同文件。顶层 `learn` 是模型架构参数声明，不是 `main` 中的宿主训练控制流；训练循环与应用控制由 Rust 主程序负责。执行不得依赖 Python。
旧通用 WGT 宿主脚本的 `agent Actor {policy; [critic;] learner:update;}` 与 `learn actor {...yield record;}` 属于不同的整批采集/learner 回调机制；不将它们与顶层模型定义中的标量 learn 参数混为一谈，也不把旧脚本机制作为新模型 API 的发展方向。保留中的现有实现契约见 docs/agents-environments.md。
原生environment声明observation/action/feedback固定f32宽度及reset/step；Ctor(state,seed)构造reset一次，reset->{state,observation}，step->{state,observation,feedback,terminated,truncated}；同步纯WGT，state固定schema，有界有限记录字面量，全部验证后state/RNG原子提交。结束后拒step至reset，实例RNG隔离，reset重放seed，不影响host RNG。actor.act仅类别sample，logprob先取actor.policy局部变量。资源递归拒save json/async，无agent/env整实例checkpoint，仅显式policy checkpoint。完整签名/限制docs/agents-environments.md，真实例子crates/wgtime-core/tests/agent_environment.rs。
语境能力见 language/source_capabilities.json；共同正负语料在 language/conformance.json。
连续声明 `let a=expr, b=expr;` 在共用 parser 展开为原有 Stmt::Let 序列；从左到右各求值一次，后项可引用前项，无整组回滚。host/async/forward/loss 继承原有作用域和重名规则；纯 operator 在支持它的工作区同样展开。嵌套调用/列表/record 的逗号不分隔声明；无元组/新赋值语法或尾逗号，sample 解构及 collect/learn 须独立声明。完整例子 crates/wgtime-core/tests/continuous_let.rs；验证 tests/continuous_let.rs 与真实 AOT。
函数共用 parser 不表示普通 fn 与 tensor forward/loss 语义相同；模型声明独立解析。
language/primitive_specs.json 是制品 graph 原语域，不是源码内置表。编辑器容错不证明源码有效。
在 language/prelude.json 维护原生函数、张量 intrinsic、语法、类型、常量与内部接口的统一内部目录；同步纯 helper 的 operator_arity 和需要保留的名称标记，不在 parser 或生成器另写名单。
修改后运行 node scripts/check_prelude.cjs、node scripts/test_prelude.cjs 和 node scripts/generate_editor_support.cjs，再用 --check 验证编辑器资产同步。
构建检查目录结构，Node 检查实现集合、签名及默认值；原生声明不能写在普通源码中。旧 std/prelude.wgt 已移除，不恢复其生成链。完整契约见 docs/prelude.md。
普通函数与 tensor 上下文分开登记，不自动引入 std 模块。
原生异步第一阶段见 docs/async-tasks.md 与 crates/wgtime-core/tests/async_tasks.rs：async fn调用冷future，tasks作用域spawn、await只等目标并轮询兄弟任务，作用域join/取消并等待在途IO清理。
仅纯数据参数/返回值，嵌套async await可用；拒绝嵌套tasks、async内spawn/tasks、模型等资源参数/返回、host/compute活跃上下文。await net(x)没有异步模型执行路径，不能宣称支持。
资源预算见 docs/compute-budget.md：WGT_HOST_POLICY 单次解析，权限/HTTP/状态共用资源初始化前固定的绝对截止时间；子scope只收紧，错误退出恢复原父deadline与上限但不退已消费额度。HTTP仅固定授权端点GET，不跟重定向、不带环境代理/认证，剩余不足1ms拒绝；不提供POST/SSE/共享异步网络账本或完整沙箱。真实AOT验证用 crates/wgt/tests/resource_execution.rs，不能用compiled::run替代生成二进制。
await sleep(ms)与await load text from path通过固定两线程/64队列阻塞池执行；文本最多64MiB，metadata后按大小预留共享操作预算，worker按实际字节结算，未用预留恰一次归还。解释/AOT均须真实运行，不把解析通过当并发支持；AOT生成原生future，不回放AST。
`train model on inputs against targets using strategy` 是旧通用 WGT 脚本训练接口，静态调用同单元同步用户fn strategy(model,inputs,targets)，
三参各求值一次且左至右、策略调用一次并返回实际结果；局部同名值不覆盖静态目标，拒绝builtin/构造器/async/错误形参。新 Rust 主程序模型 API 不通过扩展此脚本入口实现。
为轻客户端驻留策略使用单一源码单元：源码包含 import 且所选入口可达静态策略调用时，编译会在调用 Cargo 前拒绝。服务注册不解析导入文件；普通客户端与策略可共享同单元辅助函数。保留普通调用的局部绑定检查、语法内置的直接绑定和严格短路 bool 规则；修改时同时验证同步客户端、异步客户端和驻留策略。
结果限纯数据，允许null，最大深度32、计量1MiB，拒绝资源句柄。超参写在策略体内，与配置块互斥；策略决定零/多次显式commit，无隐式提交或重试；失败不撤销已成功commit。
例crates/wgtime-core/tests/train_strategies.rs；完整契约docs/native-training.md。
CPU forward 自定义算法：以 none 构造，differentiate(model,inputs,targets) 对受限 fn loss 求批次平均梯度；
gradient_snapshot(handle) 提供loss/parameters/gradients/names/shapes/state/steps，普通WGT循环自行计算候选参数和状态，
commit_gradients(handle,candidates,state) 原子提交并释放；release_gradient(handle) 放弃快照。
最多16存活句柄、64MiB计量预算，状态最多32层/1MiB且不含资源句柄；普通host程序不自动可微。
forward-training.v2 checkpoint保存自定义状态和步数，仍读旧v1；不自动执行算法，不支持该路径的顺序/GPU模型。
完整契约 docs/native-training.md；真实例子 crates/wgtime-core/tests/custom_optimization.rs，beta=0为SGD。
differentiate_at(gradient,candidates) 在临时参数上复用首次句柄保留的同一f32批次求一阶梯度，
新快照parameters/state/steps仍是原提交基准，loss/gradients来自临时点；不写live模型，保留批次与绑定计入64MiB。
crates/wgtime-core/tests/custom_optimization.rs 用两次一阶求导、一次原子提交定义可修改SAM+SGD/动量，无高阶AD或SAM专用对象。
旧通用 WGT 脚本普通函数中保存推理模型用 `save model VALUE to PATH;`，读取用 `load model from PATH using OPTIMIZER` 创建新优化器；Rust 主程序模型 API 的生命周期与权重入口见 [docs/native-backend.md](../../../docs/native-backend.md)。
保存训练断点用 `save checkpoint VALUE to PATH;`，读取用 `load checkpoint from PATH` 恢复原优化器状态。
旧 model/save/restore 加载与保存调用已移除，不能作为兼容别名使用；`model` 仍可作为普通函数或模型句柄名称，未解析到用户定义/绑定的旧 loader 调用在运行期拒绝。所有参数必填，保存拒绝已有目标。
本地运行 `cargo test -p wgtime-core --test source_conformance` 与 `node scripts/test_source_conformance.cjs`。

普通脚本模块与纯张量算子是两种独立导入机制。core 兼容层保留本地 `module NAME` 及 `./module.wgt` 导入，普通及 async 函数的解析边界见 `crates/wgt-language/src/script/modules.rs`，不能用于 forward/loss。旧 `std/openresponses/harness` 与 OpenResponses 宿主脚本示例已移除；工具映射、网络交互和控制循环由 Rust 应用负责，不恢复旧模块或引用其历史测试来证明当前客户端编译支持。

纯张量自定义算子：forward/loss 可调用只含不可重复 let 和最终 return 的 fn；参数全部必填且显式传入，
不可捕获模型参数/局部变量，不允许递归、状态写入、I/O 或宿主回调；纯 helper 条件允许比较、`positive` 和布尔字面量。`operators NAME` 声明独立库，
`import alias from "./ops.wgt"` 或 `"MODULE/LibraryName"` 导入，仅供张量调用 `alias.fn(...)`；这不是普通脚本 module loader。
算子展开为原生 IR，继承梯度与保存/AOT；release-assets 自动打包库，fetch/锁文件校验源。
完整签名、原语集合、版本和限制见 docs/custom-operators.md；真实例子 crates/wgtime-core/tests/custom_operators.rs。
实现与验证入口见 references/custom-operators.md。

- parameter p: prob = 0.7 使用可训练 logit，初值必须在 (0,1)；frozen p: prob = 1 支持精确端点。
  读取为 f32[1] 概率，底层参数/梯度暴露 logit；不得 write(prob) 或直接作为 linear 的参数。
- 互斥类别输出可声明 `output scores: logits[N]` 或 `output choices: probabilities[N]`（N=1..16384）；只按真实 forward 语义声明，raw linear 输出使用 logits。模型返回、局部变量/函数参数、制品/训练断点和 AOT 保留类型。
  `scores.probabilities` 对 rank-1 普通 f32 分数及 logits 做减最大值稳定 softmax，可写 `linear(x,w,b).probabilities`；对 probabilities 验证并保持原值，每项有限且严格 [0,1]，f64 累加总和绝对容差 1e-5，不静默归一化。
  f32 即使数值为 [0.2,0.8] 仍按分数转换；结果在局部变量、纯 helper 和连续后缀中保留概率类型，不二次 softmax。拒绝非向量/空向量/非有限值；输出声明本身不转换。显式 `softmax(x)` 保留；独立二分类 logit 未实现，标量 prob 不变。f32/logits 复用 softmax 梯度，概率验证用恒等梯度；loss 的 log_softmax/CE 仍直接消费 logits，GPU 新类型路径未验证。
  模型类别采样新语法以 `decode policy(x) {method:sample;}` 为目标；等待 parser/runtime 实现后核实返回类型、RNG 与不提交 forward 写入的完整契约。
  `crates/wgtime-core/tests/sample_record.rs` 描述旧实现的同次动作/概率绑定；新 decode 构造语法尚未完成接线，勿复制旧形式。
  完整例子 crates/wgtime-core/tests/fixtures/cnn_probabilities/model.wgt；契约 docs/reference.zh-CN.md、docs/native-training.md。
- 记录批次：`let batch=collect {for i in range(0,35){yield {observation:[i],target:[i*2]};}};`，声明形式非任意表达式；字段左到右一次求值、同字段/种类/固定同质shape，整条追加。`batch.observation`投影，`batch[i]`取记录，保留.mean/.probabilities后缀，同名字段用`batch[i]["mean"]`。`for part in batch size 16`顺序对齐切分，保留尾批/空批零次/size正整数；yield归同函数最近collect，return穿透并清理。100k行/256字段/64层/1m节点及1m文本字节；不接受句柄，不自动可微。完整运行例子crates/wgtime-core/tests/ppo_records.rs与docs/reference.zh-CN.md记录批次节。
- 普通函数文件IO用 `save KIND VALUE to PATH [overwrite];`、`load KIND from PATH`，KIND为text/json/model/checkpoint；load model必填using优化器，checkpoint恢复记录状态。父目录须存在，默认拒绝覆盖，显式overwrite只原子替换普通可写文件；64MiB/文件、按编码字节消耗operation budget，host policy下拒绝直接IO。JSON拒绝句柄/非有限数，UTF-8文本，JSONL继续流式batch/close。断言用 `assert CONDITION;` 或 `assert CONDITION, MESSAGE;`，严格bool且仅失败求消息。详见docs/file-io.md。
- 普通 f32 权重不默认限制到 [0,1]；`learn bool` 和有界 `learn float` 是独立架构参数声明。区间含零不能防止除零，普通脚本的除零和非有限结果报错。
- mix p { then { a } else { b } } 用于 forward/loss 张量混合，每支一个表达式。
  与 route 相同，推理在精确端点短路内联分支；训练/VJP 保留完整梯度。提前 let 的计算不会省略。
- fn loss(prediction,target) 是无模型参数的张量损失，返回 f32[1]；配置 `loss: loss` 的声明式 train 才执行该自定义训练。
  支持 mean、square、减法与已有张量算子；不是任意宿主函数自动微分。
  平均可写 `x.mean`、`square(prediction-target).mean`，等价 `mean(x)`；后缀优先于一元/二元运算。
  普通函数接受非空一维有限 f64 列表并返回标量；forward/loss 的 f32[N] 归约为可微 f32[1]。
  不支持其他归约后缀或 axis 参数；完整例子 crates/wgtime-core/tests/postfix_mean.rs。
- `x[start:end]` 是零基左闭右开一维切片，省略端点默认 0/长度，不裁剪越界值。
  普通列表允许运行时整数端点和空结果；forward/loss 仅接受编译期数字常量表达式且拒绝空结果，梯度只进入选中区域。
  可接调用/括号/索引/切片及 `.mean`，不支持负索引、步长、多轴或下标赋值。
  `slice(x,start,length)` 保持旧张量长度语义：`slice(x,1,1)` 等价 `x[1:2]`。
  契约见 docs/reference.zh-CN.md；可运行例子 crates/wgtime-core/tests/slicing.rs。
- sample p { then { a } else { b } } 仅用于普通函数，概率为 [0,1] 标量，仅执行一边。
  seed 控制 RNG，精确端点不抽随机数；forward/loss 中拒绝；模型类别采样改用待实现的 decode 语法。
- mix/sample 分支可嵌套、可选尾随分号，不支持块内 let/return；普通 if 仍要求 bool。
- forward 工作内存及分段执行的完整签名、工作量计费和例子见 docs/working-memory.md 与 crates/wgtime-core/tests/workspace.rs；它们不提供跨请求缓存。

兼容层行为回归：crates/wgtime-core/tests/differentiable_syntax.rs 与 crates/wgtime-core/tests/lazy_routing.rs。
新模型例子解释轴、共享参数与初始化/训练区别。只对当前受支持的路径验证解释、客户端编译或真实服务，并检查梯度、回滚与保存续训；未迁移的客户端构造应明确拒绝。
公共语言变更同步主参考、专题文档和此技能；运行 git diff --check。
模型构造/加载与训练分离；保留模型放 weights/，不覆盖已有产物。遵循 docs/storage.md。

标准库完整契约见 docs/standard-library.md。`std/` 的旧用户原生声明与宿主模块已移除；当前模型纯张量库还包括 `std/attention` 和 `std/numerics`，算法均须显式导入，模块导入及编译 forward 尚待完整验收；不要恢复旧声明或宿主模块来证明模型执行支持。纯张量逻辑模块 std/activation 导出 swish(x)、swiglu(gate,value)、squared_relu(x)，非空同形 f32[N] 到 f32[N]；std/loss 导出 squared_error(prediction,target)、half_squared_error(prediction,target)、categorical_logits(logits,target)，同形 f32[N] 到 f32[1]，前两项特征均值、最后一项类别加权负对数似然之和。全部参数必填，无默认值/广播/隐含batch；不创建权重或状态，target分布由调用方保证，CPU梯度继承展开原语。使用逻辑名导入，不暴露内部 /operators 路径。普通列表原生函数无需import，不能据此用于forward/loss。
数据接口：zeros(count)、ones(count)、full(count,value)、linspace(start,end,count)、
one_hot(index,classes)、concat(x,y)、vector_slice(x,start,end)、gather(x,indices)、
compress(x,mask)、reverse(x)。参数全部必填；有限 f64 一维列表，允许空列表；
mask 必须 bool，索引为合法零起始整数，切片左闭右开；输出上限 100 万项。
linspace 含端点，count=1 只给起点；独热 classes>=1。预算在输出分配前检查。
这些是普通脚本原生函数；完整签名以 language/prelude.json 和 docs/reference.zh-CN.md 为准，std/data 下旧演示已删除。

独热编码使用 `onehot[N] index`（N 为 1..1000000 字面量），例如 `onehot[4] 2`。
复杂索引用括号；等价 `one_hot(index,classes)` 支持动态类别数。只限普通函数，不对索引求导。
输出长度按类别数扣预算，索引必须为 [0,N) 整数；参考 crates/wgtime-core/tests/differentiable_syntax.rs。

概率分析标准库：normalize_probabilities(weights)、entropy(p)、kl_divergence(p,q)、
js_divergence(p,q)、cross_entropy_probs(p,q)、temperature_softmax(logits,temperature)。
全部参数必填，非空有限 f64 一维列表，双分布等长；分布项 [0,1]、总和绝对误差<=1e-9，
通过后修正总和；权重归一化要求非负且有正质量，温度有限且>0。单位 nat。
KL/交叉熵 p>0 且 q=0 报错，JS 支持不相交分布。无张量梯度或新增 fit 目标。
这些是无需导入的普通原生函数；旧 std/probability 演示已删除，完整签名与限制见 language/prelude.json 和 docs/reference.zh-CN.md。

更新既有代码时，route 可等价改写为 mix；独立标量 logit=0 的门可用 prob=0.5 并去掉读取处的 sigmoid。输入相关的 sigmoid(linear(...)) 保留。固定类别独热使用 onehot[N]；模型类别采样目标语法为 decode，待实现后迁移示例。

原生 RL：`train model on inputs against targets {rate:lr; loss:loss; clip:max_norm;}` 使用内置配置块，表达式返回更新前 f32 批次平均损失；rate必填，loss默认mse，可选mse/cross_entropy/loss，clip可选且仅CPU支持。仅 forward + fn loss 使用loss目标，sgd/adam；全批次平均梯度做全局 L2 裁剪后原子更新。范数只计实际可更新的参数所有者；View/Add 梯度先合并到 resident 所有者，排除表达式槽位和 frozen 参数，避免重复计量。旧fit/train调用和tuple形式拒绝。`target NAME: f32[N]` 指定 loss 第二形参宽度/名称，默认同输出，不增加 forward 输入。forward/loss 支持有限数字向量常量、detach、exp、log、log_softmax、sum、minimum（相等各半梯度）、clip（有符号字面量边界、端点零梯度）。全部无广播/默认值。完整契约 docs/reinforcement-learning.md，运行例子 crates/wgtime-core/tests/ppo_records.rs；不支持单独 backward/step、跨调用梯度对象、环境/RNG 全进程续训。

GPU/WGSL（本地 WGT 0.5）：devices() 枚举 CPU 与硬件 GPU；training_device(model,id,memory_bytes,max_batch) 为顺序模型选择训练后端，training_status(model) 返回实际 device/steps/inference_device/state。声明式 train/reinforce 支持 GPU MSE/CE/REINFORCE + SGD/Adam；forward/loss 仍只在 CPU，推理仍 CPU。wgsl_kernel(source,entry,id,memory_bytes) 编译字符串，wgsl_run(kernel,input,output_count,[x,y,z]) 调度固定 group0/binding0 只读输入、binding1 读写输出的 f32 缓冲；不参与自动求导、不支持 NPU。启用宿主策略或 compute_budget 时拒绝直接 GPU 训练/WGSL。检查和编译不执行 GPU；GPU 验证须使用启用 gpu feature 的实际运行时与硬件，不能由客户端编译成功推断。完整契约 docs/wgsl.md，真实例子 crates/wgtime-core/tests/script_gpu.rs 与 crates/wgtime-core/tests/gpu_training.rs。训练事务合并一次提交并复用同尺寸设备缓冲，每步仍上传和回读；不要声称跨步权重常驻。

## 原生测试与评估

新增测试通过 Cargo 验证 Rust 模型定义与真实服务。`wgt-model/tests/service.rs` 的执行用例需要预构建 `WGTIME_TEST_DAEMON` 并显式包含 ignored 测试，默认解析测试通过不能计作服务执行通过。
旧 `wgtime_core::script::testing::Suite` 只从传入源码发现 `test_*` 函数并独立运行，不扫描目录或提供 CLI filter/JSON 汇总。维护入口、覆盖映射和编译拒绝边界见 references/native-testing.md 与 docs/native-testing.md。

WGSL 工作组数量受普通脚本整数范围限制，每项为 1..1000000000 且不超过所选设备限制；超范围值报错，不截断。


<!-- permissions-foundation -->
宿主权限底座在解释执行与 AOT 执行入口读取一次 `WGT_HOST_POLICY`（JSON，最多 1 MiB），
拒绝未知字段、无效预算及提供者校验失败；失败不会退回无权限限制模式。
启用策略即禁止直接路径的模型/断点/数据 I/O；权限检查不能由脚本绕过。
`work`、`workspace_bytes`、`calls`、`deadline_ms` 必填，前三者是预算上限信息，
权限底座自身只执行直接 I/O 拒绝和协作式 deadline；计算计费需要完整资源提供者。
`work`、`workspace_bytes`、`deadline_ms` 最大 1000000000，`calls` 最大 10000；0 允许，
0 毫秒在执行检查点立即超时。deadline 继承同一绝对截止时间，复制子运行时不续期。
最小权限运行时拒绝非空设备/文件/输出/网络授权、非零 I/O 额度及 CPU/地址空间硬限制；
完整资源运行时将同一份已解析策略交给提供者，继续执行设备、文件和进程隔离检查。
协作式 deadline 不会中断正在进行的阻塞系统调用，也不等同于 OS 沙箱。
最小有效策略：`{"work":100,"workspace_bytes":4096,"calls":2,"deadline_ms":10000}`；
用 `wgt 0.5
fn main() { return 7; }` 验证基本执行，直接 `jsonl("data.jsonl")` 在策略下应拒绝。
此处描述本地实现与分阶段接线，不表示设备/隔离扩展已经随权限底座发布。


### Transformer Encoder acceptance

The fixed-shape two-head encoder acceptance model is `crates/wgt-model/tests/fixtures/transformer_encoder/model.wgt`; its Rust definition/service checks are in `crates/wgt-model/tests/transformer_encoder.rs`. The fixture uses distinct `query_sequence` and `key_sequence` axis names: attention contracts the `head_dim` axis and softmax reduces `key_sequence`. Heads are represented as the batch dimension of rank-3 `batch_matmul`. Keep the single-vector facade boundary explicit. Do not claim dynamic batch, masking, dropout, positional embeddings, arbitrary dtype or GPU fusion from this fixture. See `docs/transformer-encoder.md`; record only tests that were actually run.
