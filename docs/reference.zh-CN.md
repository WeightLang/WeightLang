# WeightLang 0.5 语言参考

当前源码只接受 **`wgt 0.5`**。这是破坏性语言更新，未发布；
`wgt 0.1/0.2/0.3`、`wgt script 0.1`、`wgt adaptive 0.1` 全部拒绝。
模型声明、训练函数和自更新前向函数属于同一个编译单元，不能通过选择命令改变语法版本。
工具包版本、模型 JSON 的 `version` 和训练断点版本是独立概念，不随语言版本替换。

## Rust 主程序与 WGT 模型目标

目标工作流由 Rust 主程序负责数据准备、I/O、调度与训练循环，WGT 定义模型结构、模型内部计算及可选的可微 loss。Rust 库包名为 `wgt-model`，crate 名为 `wgt`。`ModelDefinition::parse`、`ModelDefinition::from_file`、`model!` 与 `include_model!` 共用模型解析与定义；两个宏仅接受 Rust 字符串字面量，编译时复用同一 parser 检查并嵌入源码，运行时仍解析定义。模型模块导入在本地实现中包含 `std/activation`、`std/loss`、`std/attention` 和 `std/numerics`；文件入口还实现了相对路径纯计算模块及嵌套导入。模块导入和编译 forward 尚待完整行为验收，不能视为已支持或已发布能力。既有 forward 模型入口只接受单个 `f32` 向量；多输入 typed 操作已通过本地指定范围验收，尚未发布，见[模型操作](model-operations.md)。

模型执行、参数和优化器状态由唯一的机器级 `wgtime` 服务持有。定义解析、编译期检查和源码嵌入不连接服务或初始化模型；Rust 程序须先显式调用 `Client::connect()`，再请求初始化、训练或权重保存。当前 facade 提供 `Model::save_weights`，没有公共训练 checkpoint API。Rust facade 通过 IPC 使用服务端原生编译 forward；这不意味着旧脚本执行路径已全部删除。API 已在本地实现并通过定点验证，尚未发布；不代表更广泛的能力支持。完整签名与契约见 [Rust 后端契约](native-backend.md)。

```rust,ignore
use wgt::{include_model, model, Client, InstanceOptions, ModelDefinition};

let inline: ModelDefinition = model!(r#"wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
"#);
let compiled = include_model!("models/scale/model.wgt");
let runtime = ModelDefinition::from_file("models/scale/model.wgt")?;

// 显式连接并初始化后才执行模型。
let client = Client::connect()?;
let model = client.initialize(&runtime, InstanceOptions::default())?;
let output = model.forward(&[3.0])?;
assert_eq!(output, vec![6.0]);
# let _ = (inline, compiled);
```

宏和定义解析只检查/构造定义，不执行模型；应用先调用 `Client::connect()`，再显式初始化并执行实例。训练与保存也由 Rust 代码显式请求，不会随定义加载自动发生。

当前模型后端由 Rust 实现。模型纯张量逻辑模块还包括 `std/attention` 与 `std/numerics`，均须显式导入，导入路径仍待完整验收。完整图算子、参数化层工厂与多轴神经网络结构尚未全部迁入原生构造器；不支持的声明显式报错，不回退 Python。

## wgtime 服务边界

Rust facade 通过 IPC 连接机器级 `wgtime` 服务，由服务持有模型执行、参数和优化器状态；模型 forward 使用服务端原生编译实现。旧脚本控制与执行路径仍在源码中，正在逐步退出，不能据此认为已全部删除或等同于新模型 API。`wgtime` 服务专题见 [wgtime](wgtime.md)。服务连接失败明确报错，不回退到应用进程内执行。模型定义解析、检查和编译不隐式初始化或训练。

## 语法与能力边界的机器可读规格

[上下文能力规格](../language/source_capabilities.json) 记录当前源码 edition、解析入口及普通函数、
张量 forward/loss、模型声明和编辑器的边界；[共同正负语料](../language/conformance.json)
由 Rust 与编辑器本地测试共同读取。这是已测试边界的有限规格，不是完整内置函数表。
普通函数与 forward/loss 共用函数 parser，但张量 lowering（转换为张量指令）另行限制表达式、
控制流与形状；顶层模型声明由独立解析入口处理。函数解析成功不代表运行时调用有效。
[primitive_specs.json](../language/primitive_specs.json) 描述制品 graph 原语，供制品验证使用，
不能当作当前源码全部可调用内置函数的清单，也不改变旧制品的读取兼容性。

## 无需 import 的内置入口

旧 `std/prelude.wgt` 用户声明源已移除。当前 [language/prelude.json](../language/prelude.json) 是运行时 registry 与编辑器使用的过渡内部目录；不要将其视为新的用户声明入口或稳定公开 API。`language/prelude_metadata.json` 保留语法、类型、常量及内部 lowering 元数据。模型纯张量模块包括 `std/activation`、`std/loss`、`std/attention` 和 `std/numerics`；详细边界见[模型内部标准库](standard-library.md)。
`ordinary_function` 供解释与 AOT 共用；`tensor_forward` / `tensor_loss` 是独立张量转换契约，
不能因名字出现在目录中就用于普通函数。`write` 等标注为语句，`positive` 只用于张量条件。
目录也列出 `true`、`false`、`null` 常量及 `f32`、`prob`、`logits`、`probabilities` 声明类型；这些不是可调用函数。
`std/` 模块不自动导入，仍须按各模块的使用契约显式引入。
原生 WGT 计算库使用逻辑模块名 `import activation from "std/activation"` 与
`import loss_ops from "std/loss"`，仅在 forward/loss 中调用，输入为非空同形 f32[N]。编译器随附模块与当前目录无关；开发版也允许相对导入文件的显式源码路径，本地路径不能覆盖 std 逻辑名。AOT 打包实际依赖，不写入开发机绝对源码路径。
无默认参数、隐含 batch 或标量广播。前者导出 `swish(x)`、`swiglu(gate,value)`、
`squared_relu(x)`（同形返回）；后者导出 `squared_error(prediction,target)`、
`half_squared_error(prediction,target)`（特征均值）及 `categorical_logits(logits,target)`
（类别加权负对数似然之和），均返回 f32[1]。target 分布条件由调用方保证，不动态验证。
库不创建权重或训练状态，支持 CPU 梯度及 AOT；标准源内嵌、未知 std 路径报错。
模型纯计算只允许已提供并通过定义解析的标准逻辑模块；普通宿主 harness 模块正在退出，不属于新的模型导入 API。
完整数学定义、形状与状态边界见[模型内部标准库](standard-library.md)。

## 编译单元与入口

**模型源码入口固定为 `model.wgt`。** 每个独立模型或训练示例使用独立目录，
例如 `examples/native_training/working_memory/model.wgt`；模型/功能名放在目录名中，
不能用 `main.wgt`、`train.wgt` 或 `working_memory.wgt` 作为模型入口。
`model.wgt` 是文件入口，`fn main()` 是显式执行时调用的函数，两者不是同一概念。
这是项目、示例与技能的组织规范；底层 CLI 接受显式源码路径不代表推荐其他入口名称。
标准库模块文件和制品中的 `source.wgt` 快照不属于模型源码入口，不随此规范改名。

首个非空、非注释行必须恰好是 `wgt 0.5`，版本头只能出现一次。
`#` 到行尾为注释，函数字符串使用双引号和 JSON 转义。顶层模型声明一行一条；
函数使用 `fn name(args) { ... }`，语句以分号结束，控制流块无需分号。
名称使用 ASCII 字母或下划线开头、后接字母/数字/下划线；`__` 前缀供编译器使用。

一个文件可声明一个根模型、多个具名子模型定义和多个函数。`fn main()` 无参数，执行程序时才调用。
模型名成为构造函数，调用才初始化会话；只解析/编译不访问训练数据、不训练、不保存。
只有模型声明的文件可检查，但执行或编译可执行文件时必须显式定义 main。

```sh
weightc check examples/native_training/supervised/model.wgt
wgt run examples/native_training/supervised/model.wgt
wgt run examples/native_training/supervised/model.wgt --compiled
weightc compile examples/native_training/supervised/model.wgt -o .cache/weightlang/regression
```

`wgt script` 是 `wgt run` 的命令别名，不是另一套语言。
解释和编译模式使用同一前端；编译模式生成 Rust 控制流和静态模型计划，运行时不解析 WGT。
`weightc check` 检查语法及受支持的模型声明，不证明所有函数调用、路径初始化或循环终止。

## 多入口模型操作

WGT 0.5 的 typed 操作已通过本地指定范围验收，尚未发布。core 7 个 suite 共 31 项、typed import roundtrip 1/1，最终 daemon 重建后真实 IPC 5/5（operations 1、既有 service 1、CNN 3）；language 与 model 常规测试 16 项通过、8 项 ignored，相关服务测试单独执行。workspace all-targets 检查、typed `model!` 宏编译和完整示例解析通过；解析不证明执行，完整覆盖范围见[模型操作验收状态](model-operations.md#验收状态)。每个入口的每个参数必须标注 `f32[N]`，N 为 1..16384，各参数宽度之和不超过 16384；可有多个参数。返回类型为 `-> f32[N]`，省略表示无返回值；无参无返回的 `fn reset() {}` 合法。typed 模式使用函数签名声明端口，不使用根级 `input`、`output` 或 `target`，不支持 `loss`、子模型或操作互调；纯张量辅助函数仍受各自限制。

模型定义中出现 typed 函数签名或 `state` 声明，或者没有 `forward` 但存在无参函数时，会进入操作模式。该模式中无参且无返回类型标注的函数按 void 入口处理，不是纯 helper；例如 `fn constant() { return [1]; }` 会被拒绝。有返回值的无参入口须写成 `fn observe() -> f32[1] { ... }`。纯 helper 必须至少有一个未标注类型的参数，仍须遵守纯张量 helper 限制。

`state name: f32[N] = [有限数值, ...]` 声明固定向量状态，初值长度须匹配，末尾分号可选。`name <- expression;` 显式更新裸状态或参数名，分号必需；冻结目标及形状不符的更新拒绝。状态名与参数名不能冲突，入口参数也不能遮蔽存储名。typed `forward` 不允许写状态或参数。表达式遵守当前张量 lowering 限制，不支持任意宿主计算。

同一服务实例的入口共享参数与状态；调用内写后读看到候选值，成功并通过校验才原子提交，失败保持原值。状态与权重分离，普通权重不保存操作状态；不提供状态快照、状态 checkpoint 或重启恢复保证。操作由服务侧原生代码执行而非 VM，typed 操作不支持自动微分或跨状态求导。旧 forward 路径的 `train_with_policy` 提供 `ContinuousOnly`，遇到模型中任意 `learn bool` 或 `learn int` 即拒绝；`StraightThrough` 允许离散参数使用 STE，旧 `train` 保持 STE 兼容行为。策略检查与训练由单个 `TrainStepWithPolicy` 请求原子处理，不持久设置策略。参数候选拒绝 frozen/probability 参数，bool/int 候选必须严格离散；没有 CAS，revision 仅在服务响应中返回，不承诺拒绝过期候选。`update_parameters` 成功清空旧 forward 优化器，失败保持原状态，重新训练须通过 `load_weights` 显式选回。typed 操作无优化器，加载权重保留 state。当前 Rust facade 签名见[Rust 后端契约](native-backend.md#多入口操作)。详见[模型操作](model-operations.md)。

## 声明式模型

以下为当前原生构造支持的完整声明集；端口不含 batch 轴。

模型定义中的 `learn` 支持标量及固定长度数组。本地 core 定点五组共 33/33 通过（adaptive、compiled_forward、control 10、std 5、learn 6），另有 architecture 6/6 和 conv 3/3 通过；真实 control facade 1/1 已覆盖 native-cpu 组合模型的训练、权重回载、续训一致和释放。矩阵分支正反例与 5→5.5 已通过；这些结果不代表全量验收或发布。语法、float 投影、离散 shadow/STE 与 `depth_gate` 的具体限制见[可学习的模型架构参数](learnable-architecture-parameters.md)。它是顶层模型架构声明，不是旧宿主脚本中采集记录并调用 learner 的 `learn actor { ... }` 形式；旧形式不代表新模型定义 API。

| 声明 | 参数、默认值与约束 |
| --- | --- |
| `model Name` | 一个模型名称，不得与函数或内置函数重名 |
| `input x: f32[N]` | 一个输入，固定 N 为 1..16384，先于 layer |
| `output y: f32[M]` | 一个输出，M 必须等于最后一层宽度；互斥分类也可声明 `logits[M]` 或 `probabilities[M]`，见下文分类向量 |
| `layer id: linear(in=N,out=M,bias=1)` | out 必填；in 默认前层宽度；bias 仅 0/1，默认 1；可用 `linear(M)` |
| `layer id: relu` / `tanh` / `sigmoid` / `identity` | 不改变向量宽度，无参数 |
| `init seed=N` | 必填、不可重复；N 为非负 u64；调用构造函数时进行均匀 Xavier 初始化，bias 为零 |

层按源码顺序执行，层名唯一。每个模型最多 400 万参数；参数名为 `层名.weight/bias`。
模型构造函数 `Name(optimizer)` 的 optimizer 必填，仅 `"sgd"` 或 `"adam"`。
调用两次产生独立参数；复用同一个句柄则共享参数及优化器状态。初始化不等于训练。

```wgt
wgt 0.5
model Regression
# 每个样本一个输入特征和一个回归目标；不含 batch。
input x: f32[1]
output y: f32[1]
layer hidden: linear(out=8)
layer activation: tanh
layer result: linear(out=1)
init seed=42

fn main() {
    # 构造只初始化；以下所有步共享同一组参数。
    let net = Regression("adam");
    for step in range(0, 100) {
        # 输入按 [样本, 特征] 排列；MSE 对样本与输出元素取平均。
        train net on [[0], [1]] against [[1], [3]] { rate: 0.03; loss: mse; };
    }
    return net([1]);
}
```

## 原生异步任务

普通脚本支持 `async fn name(args) { ... }`、`tasks { ... }`、`spawn future` 与 `await future_or_task`。
调用 async fn 创建冷 future；spawn 仅用于同步 tasks 作用域，await 驱动/等待，正常退出 join，失败取消并等待在途 I/O 清理。
第一阶段使用本地 future executor 和固定两线程阻塞 I/O 池，支持 async 用户函数、嵌套 await、`await sleep(ms)`、`await load text from path`；
不支持嵌套 tasks、async fn 内 tasks/spawn、资源参数/返回值或异步模型执行。活跃 host/compute 限制下显式拒绝。
完整签名、预算、RNG、错误/释放契约和限制见[原生异步任务](async-tasks.md)，例子见[async_tasks/model.wgt](../examples/native_training/async_tasks/model.wgt)。

## 普通函数、训练与数据

值包括有限数值、bool、字符串、null、列表、JSON 对象和不透明资源句柄。
`let name = expr;` 绑定变量，`name = expr;` 修改已有绑定。

`let a = expr, b = expr, c = expr;` 一次声明多项，可在逗号前后换行。每项从左到右求值一次并立即绑定，后项可引用前项；等价于同一块内连续的普通 `let`，不新增作用域或事务，后项失败不回滚前项副作用。普通函数保留原有重绑定规则，forward/loss 保留原有声明与形状约束。函数实参、列表和记录字面量内部的逗号由各自括号消费；最外层逗号才分隔声明，不提供元组或多目标赋值。`collect` 和绑定结果的 `learn` 必须独立书写，不能混入连续声明。组末不能有尾随逗号。
完整示例见 [continuous_let/model.wgt](../examples/native_training/continuous_let/model.wgt)。

支持函数调用、列表/对象索引、一元 `- !`、算术 `+ - * / % @ **`、比较及短路 `&& ||`。
矩阵乘法可写为 `A @ B`，等价于 `matmul(A,B)`；rank-2 转置可写为 `A.T`，等价于 `transpose(A)`。
`@` 与 `* / %` 同优先级并左结合；`.T` 是高优先级后缀，仅支持 rank-2 张量。
`A:[M,K]` 与 `B:[K,N]` 的 `A @ B` 得到 `[M,N]`，不做 batch broadcasting。局部绑定支持 `let q: f32[sequence=8, hidden=32] = value;` 形式的固定 Shape 注解；命名轴可用于矩阵收缩、Softmax 轴和 `permute` 的编译期一致性检查。
二元优先级从低到高为：`||`、`&&`、`== !=`、`< <= > >=`、`+ -`、`* / % @`、`**`；
幂运算优先于一元负号，即 `-2 ** 2` 等价于 `-(2 ** 2)`；一元 `!` 仍按一元运算解析。
幂运算右结合（`2 ** 3 ** 2` 等价于 `2 ** (3 ** 2)`），其他二元运算左结合。
张量 lowering 仍只允许当前明确支持的 `x ** 2`，不因此开放任意指数。
数值运算产生无穷大或 NaN 时报错，包括数字除以零及对零取余；不会隐式替换除数。
源码入口限制为 4 MiB，函数词法分析最多生成 100000 个 Token；超限在扫描期间拒绝。
普通脚本中的 `sample p { then { a } else { b } }` 按标量概率 p 在 [0,1] 抽样，只执行选中表达式；
精确端点不消耗随机数。它不建立张量梯度，示例见 [sample_branch/model.wgt](../examples/native_training/sample_branch/model.wgt)。

类别采样目标语法为 `decode policy(input) {method:sample;}`。parser/runtime 实现前，此形式为规划契约，不可执行；目标模型/输入只求值一次，采样不提交 forward 写入，并沿用模型声明的 logits 或 probabilities 分布语义。返回字段、RNG 消耗和预算限制待实现后补齐。
条件必须 bool，无隐式数值或张量真假转换；函数参数按位置传递，参数数目严格匹配。

## 原生 agent、environment 与 learn

`agent NAME { policy; [critic;] [learner: FUNCTION;] }` 声明专用构造器，显式传入已有模型句柄，
例如 `let actor=Predictor(policy);`。policy/critic 保留别名，不隐式训练；learner 为同步二参具名用户函数。
`let report=learn actor { ... yield {field:value}; ... };` 整批完成后先释放目标 policy 写保护，
再调用一次 learner(actor,batch)，返回其报告；`using other` 可覆盖，无默认时报错，普通模型必须显式 using。
learner 自行决定训练/提交次数。采集失败或 return 不调用 learner；同 identity 所有别名写入均拒绝，
包括训练、自定义提交、优化器重置、reload、weightfs 和含 Write IR 的 forward；只读求值/采样/保存允许。

`environment NAME { observation:f32[N]; action:f32[A]; feedback:f32[F]; reset:RESET; step:STEP; }`
构造签名 NAME(initial_state,seed)，宽度1..16384，seed整数0..1000000000。构造调用一次reset。
RESET(state,seed) 返回 `{state,observation}`；STEP(state,action) 返回
`{state,observation,feedback,terminated,truncated}`，均为同步二参函数；feedback不限定为reward。
`env.observation` 读最近观测，`env.reset()` 返回初观测，`env.step(action)` 返回完整反馈记录。
真正终止与截断标志独立保留，任一结束后 step 必须等 reset；state 首次reset后固定schema/shape。
state/观测/反馈/标志全部验证后原子提交，失败保留实例state/observation/done/RNG；外部IO不能回滚。
各实例seed/RNG隔离，不改变host RNG，reset重放原seed。`actor.act(obs)` 只读类别采样，
同次log_prob需先取 `let policy=actor.policy;` 再结构化sample；不支持sample限定callee。
普通记录字面量 `{field:expression,...}` 左到右一次求值、字段唯一、有限有界普通数据，不自动可微。
agent/env资源及嵌套容器不能save json或传入异步；无整实例checkpoint，仅显式保存actor.policy模型状态。
完整签名、预算、保存边界及有效例子见[专题](agents-environments.md)与
[agent_environment/model.wgt](../examples/native_training/agent_environment/model.wgt)。

## 记录批次与对齐小批次

普通函数支持声明 `let batch = collect { ... };`。块内 `yield {name: expression, ...};` 每次追加一条记录，
字段表达式从左到右各求值一次，完整校验通过后才追加。首条记录确定字段名称、值种类及递归固定列表形状；
之后字段集合、种类和形状必须相同，字段书写顺序可不同。列表必须同质，不自动补字段、广播或填充。
字段允许有限数字、bool、字符串、null、列表和已有普通对象，拒绝资源句柄、类别类型向量和嵌套记录批次。
最多 100000 行、每记录 256 字段、64 层值嵌套、1000000 个值节点，以及字段名/字符串合计 1000000 UTF-8 字节。
构造、列投影和分块复制按访问节点消耗操作预算，失败不退还预算。

`batch.observation` 返回按行对齐的普通列列表，`batch[i]` 返回一条普通对象，`batch[i].observation` 取字段。
普通索引列表不自动变为记录批次，缺失字段报错。无行批次尚无 schema，列投影返回空列表。
`.mean` / `.probabilities` 仍是已有后缀操作；同名记录字段用 `batch[i]["mean"]` 等字符串索引访问。
记录批次可作为函数参数/返回值；输出 JSON 为记录数组，JSON 重载仍是普通列表，不恢复批次类型。

`for minibatch in batch size 16 { ... }` 只接受记录批次，batch 和 size 各求值一次；size 必须为正整数。
所有字段按同一行范围切分，默认保持原顺序、不洗牌；最后不足 size 的批次保留，空批次零迭代。
它与 `batch(reader,n)` 数据读取接口独立，普通 `for item in list` 不变。
`yield` 属于同一函数中词法最近的 collect，不能从被调用函数向调用者 yield；嵌套 collect 完成后恢复外层。
`break` / `continue` 作用于最近循环；`return` 穿透 collect 返回当前函数，未完成批次不绑定到变量。
错误同样丢弃未完成批次并恢复外层收集状态；已经发生的 RNG、文件或环境副作用不回滚。
collect/yield/分块不支持 tensor forward/loss，不使记录自动可微；PPO 示例仍显式组装既有 f32[4] target。
完整例子见 [ppo/model.wgt](../examples/native_training/ppo/model.wgt)。

普通函数内的模型句柄可直接写 `net(input)`，包括函数参数 `fn infer(net, x) { return net(x); }`。
仅接受一个符合模型输入宽度的有限 f32 向量，返回模型输出列表；无默认参数、多输入或广播。
调用先查当前函数的局部绑定（包括参数）；模型句柄执行前向，其他局部值报不可调用，
没有局部绑定才解析普通函数、模型构造器或内置函数。块仍共享当前函数环境，不捕获调用者变量。
这不是通用函数对象或任意表达式调用：`factory()(x)` 不受支持。
`net(x)` 执行模型前向，不隐式训练、更新优化器或保存制品；
无显式写入的前向不改变参数、revision 或训练步数。forward 中显式 `write` 仍按原有事务契约提交，
失败则回滚；运行工作量仍计入执行预算。错误包含调用名、所在语句行号及形状不匹配时的预期/实际宽度。

普通函数的原生文件语法为 `save KIND VALUE to PATH [overwrite];` 和读取表达式
`load KIND from PATH`，KIND 为 `text/json/model/checkpoint`；`load model` 必填 `using OPTIMIZER`，
checkpoint 使用保存的优化器。值、路径各求值一次；相对路径按脚本目录解析，父目录须存在。
默认拒绝覆盖，显式 overwrite 只原子替换普通可写文件；每文件最多 64 MiB，按编码字节消耗 operation budget。
宿主策略模式拒绝直接路径 IO，须用授权 capability。模型与断点不能混用，load 不执行 main 或训练。
`reload model TARGET from PATH using OPTIMIZER;` / `reload checkpoint TARGET from PATH;` 在同句柄中原子替换同结构模型状态，所有别名可见；model重新初始化优化器，checkpoint恢复保存状态与步数，运行时revision递增并使旧梯度失效。结构不匹配或候选失败保留旧状态。
`save weights NET to PATH [overwrite];` 导出标准 safetensors 独立参数；
`reload weights NET from PATH using OPTIMIZER;` 将权重绑定到已显式构造的同结构模型。
仅支持有限 F32、精确 name/shape 和 C row-major little-endian，不猜图、不自动改名/转置/下载/训练。
成功原子替换，模型句柄别名保留，重置优化器、步数与自定义状态；失败保留原状态和 revision。
顺序模型 view/add 关系保留，只保存独立参数；forward 可用 none/sgd/adam，顺序模型用 sgd/adam，using 必填。
文件不含图或训练状态，不能作为 checkpoint 或直接 load weights；完整边界见[权重与训练](native-training.md)。
`assert CONDITION;` 或 `assert CONDITION, MESSAGE;` 要求严格 bool，只失败才求字符串消息，默认 `assertion failed`。解释/AOT 相同，错误含语句行号。
完整签名、格式、安全发布和 JSON 限制见[文件与断言语法](file-io.md)。

控制流为 `if condition { ... } else { ... }`、`while condition { ... }`、
`for name in list { ... }`、`break;`、`continue;`、`return expr;`。
当前函数内共享变量环境，if/for 不创建词法局部作用域；重复 let 覆盖绑定。
普通脚本可导入 `module NAME` 普通脚本模块：固定标准模块用逻辑名，本地开发模块用相对 `./module.wgt` 路径；模块可导出普通及 async 函数，不是 forward/loss 的张量算子，也不能在那里调用。纯张量 `operators NAME` 库是另一套仅供 forward/loss 的导入机制，见[自定义算子](custom-operators.md)。`std/openresponses/harness` 提供 `next(session,profile)`、`submit(session,results)` 与 `finished(step)` 协议辅助；显式工具映射和调用循环由普通 WGT 控制流负责。解释执行、默认编译和显式轻客户端 AOT 已覆盖双 worker 闭环及共享账本；AOT 在不保留模型源码、且 PATH 无构建工具链时运行。性能与吞吐尚未验收；`wgtime` 完整模型执行迁移仍按[OpenResponses 专题](openresponses.md)中的边界处理。
无任意 Python、shell 或宿主回调。源码函数嵌套与求值均有预算限制。

训练、流式 JSONL、强化学习、保存与 benchmark 的完整内置签名、默认值及限制见
[原生训练](native-training.md#内置接口)。常用模型调用为 `net(input)`；显式训练使用
`train model on inputs against targets {rate:rate; loss:mse;}`，不会由推理调用隐式触发。表达式形式返回更新前的 f32 批次平均损失；`rate` 必填，`loss` 默认 `mse`，也可选 `cross_entropy` 或 `loss`。`clip` 可选且当前仅 CPU 支持。旧 `fit(...)` / `train(...)` 调用和 tuple 训练形式拒绝。

### 统一括号切片 `x[start:end]`

WGT 0.5 本地实现支持一维、零基、左闭右开切片；`x[1:3]` 选择索引 1 和 2。
`x[:end]` 默认 start=0，`x[start:]` 默认 end=长度，`x[:]` 选择全部。
必须满足 `0 <= start <= end <= 长度`，端点为有限整数；越界报错，不静默裁剪。
不支持负索引、步长 `x[::2]`、多轴 `x[:,0]` 或下标赋值。这是当前子集，不是完整 Python 切片。

普通函数的操作数必须是列表，端点可为运行时表达式；结果是新列表，保留元素类型，允许空结果
（包括 `[][0:0]` 和 `x[len(x):]`）。字符串、对象和模型句柄不支持切片。
forward/loss 的操作数为现有一维 `f32[N]` 张量，端点必须能在编译期计算：数字字面量、
括号、一元负号及 `+ - * / %` 数字常量表达式；不能引用运行时张量或调用函数。
张量结果为 `f32[end-start]`，沿唯一向量轴选取；现有 shape 必须非零，因此空张量切片明确拒绝。
反向传播只向选中的区域累加梯度。

切片、索引和 `.mean` 从左到右结合，优先于一元和二元运算，可跟在调用、括号表达式及其他后缀之后：
`values()[1:][0]`、`(x-y)[:2].mean`。操作数、start、end 按此顺序各求值一次；
解释器与 AOT 使用相同边界检查，张量 lowering 复用既有 Slice 指令及其梯度。

原张量函数 **`slice(x,start,length)` 保持不变**：第三个参数是长度，且必须为正整数。
例如 `slice(x,1,1)` 从第二个元素开始取一个元素，等价于 `x[1:2]`，不是 `x[1:1]`。
旧函数的 start/length 仍要求数字字面量，不新增普通列表的同名函数。

```wgt
wgt 0.5
fn main() {
    let x = [10,20,30,40];
    let start = 1;
    assert x[start:3] == [20,30], "half-open bounds";
    assert x[4:] == [], "empty list slice";
    return x[:2];
}
```

包含列表、forward/loss 和显式训练的完整示例：
[`slicing/model.wgt`](../examples/native_training/slicing/model.wgt)。

### GPU 与 WGSL 内置函数

WGT 0.5 新增 `devices()`、`training_device(model,device,memory_bytes,max_batch)`、
`training_status(model)`、`wgsl_kernel(source,entry,device,memory_bytes)`、
`wgsl_run(kernel,input,output_count,workgroups)`，参数全部必填，没有隐式设备回退。
顺序模型可显式选择 GPU 执行 MSE、交叉熵和 REINFORCE 训练；forward/loss 程序仍仅 CPU。
自定义 WGSL 为普通函数的显式计算，不参与自动求导；固定一维 f32 输入/输出缓冲，
workgroups 为三个工作组数量。内核句柄不能序列化为 JSON。
完整签名、设备/预算限制、宿主策略、AOT、有效例子见 [原生 WGSL 与 GPU 训练](wgsl.md)。

## 自更新前向函数

同一个模型可用 `fn forward(x)` 定义有状态前向，替代 layer/init 声明。
函数参数必须与 input 名称一致，input 使用 `f32[N]`；output 可使用 `f32[N]`、`logits[N]` 或 `probabilities[N]`。
参数显式声明为 `parameter w: f32[O,I] = [...]` 或 `frozen w: f32[O,I] = [...]`；
初值是行优先扁平有限 f32 数组，元素数必须与正整数形状匹配。
自更新模型还可声明标量 `parameter p: prob = 0.7` 或 `frozen p: prob = 1`。
可训练 prob 的初值须严格在 (0,1)，内部保存 logit 并通过 sigmoid 读取；冻结 prob 可使用闭区间 [0,1] 的精确端点。prob 不支持 forward 的 `write`，也不能直接作为 linear 权重或偏置。
`learn bool` 提供带连续 shadow 的可学习布尔架构参数；普通 `f32` 参数不会自动限制在 `[0,1]` 或 `{0,1}`。
`frozen` 禁止本次调用内 write 及优化器更新；不意味着该参数的数学梯度为零。

`fn forward` 复用普通函数的 let、赋值、return、if、while、break、continue 语法，
操作数为定长 f32 张量。支持下述纯张量辅助算子；暂不支持 forward 中的 for、数据 IO 或优化器调用。
当前前向表达式签名如下，除 linear 的 b 外没有默认参数：

| 表达式/语句 | 语义与约束 |
| --- | --- |
| `read(w)` | 读取参数当前值，结果为扁平向量；直接引用参数也会读取 |
| `write(w, value);` | 全量替换非冻结参数，元素数须相同 |
| `linear(x,w[,b])` | w=[O,I]、x=[I]、结果=[O]，可选 b=[O] |
| `add(a,b)` / `mul(a,b)` / `a+b` / `a*b` | 等长逐元素运算，无广播 |
| `relu(x)` / `tanh(x)` / `sigmoid(x)` | 等长逐元素运算 |
| `softmax(x)` / `softmax(x,axis)` | `softmax(x)` 保持 rank-1 向量语义；多轴形式沿非负整数字面量轴归一化，使用稳定计算和 f64 累加，行为待验证 |
| `reshape(x,[d1,...,dn])` | 固定 shape 重解释，1..8 个正整数常量轴，每轴 1..16384、总元素数不超过 16777216；元素数须相同，行为待验证 |
| `x.T` / `transpose(x)` | rank-2 张量交换两个轴，两种写法等价 |
| `a @ b` / `matmul(a,b)` | rank-2 `[M,K] @ [K,N] -> [M,N]`，内轴须相等，无广播，两种写法等价 |
| `permute(x,[...])` | rank 1..8 的完整轴排列；interpreter 为零拷贝 strided view，其他算子按需物化 |
| `batch_matmul(a,b)` | rank-3 `[B,M,K] @ [B,K,N] -> [B,M,N]`，batch 维相等，不广播 |
| `scale(x,literal)` | 乘以有限有符号数值字面量，不广播且保持 shape，行为待验证 |
| `conv(x,kernel,bias,stride,padding,dilation,groups)` | 完整七参数签名，无默认值；无批次、通道优先的 f32 卷积。输入 `x` 为 `[Cin,S1,...,Sd]`，kernel 为 `[Cout,Cin/groups,K1,...,Kd]`，bias 必须为 `[Cout]`；空间 rank `d` 为 1..3，输出为 `[Cout,O1,...,Od]`，`Oa=floor((Sa+2*padding[a]-(dilation[a]*(Ka-1)+1))/stride[a])+1`。stride/dilation 各为长度 d 的正整数常量列表，元素 1..16384；padding 为长度 d 的非负整数常量列表，元素 0..16384，对称零填充；groups 为 1..16384 的整数常量，必须整除 Cin、Cout，kernel 输入通道轴须等于 Cin/groups。输入/权重参数以行优先扁平 f32 存储，必须显式 `reshape` 出所需几何；单向量模型入口不提供 batch 轴。完整语义见[卷积专题](convolution.md)。core 卷积 3/3 和真实 wgtime service 3 项已通过，覆盖训练、权重恢复续训和释放；这是局部验证 |
| `gate(p,x)` | 返回 p*x；p 的 shape 为 [1] 或与 x 完全相同，每项须在 [0,1]；[1] 门按元素应用；推理时 p 全为 0 跳过 x 表达式 |
| `route(p,a,b)` | 返回 p*a+(1-p)*b；a/b shape 完全相同，p 的 shape 为 [1] 或与 a 完全相同且各项在 [0,1]；仅元素数相同不足以匹配。推理时 p 全为 1 只求值 a，全为 0 只求值 b，否则计算两边 |
| `slice(x,start,length)` | start 非负整数、length 正整数，范围不得越界 |
| `positive(x)` | 仅作为条件；x 必须 f32[1]，严格大于零选真分支 |
| `let y = x;` | 复制张量值并保留梯度，不创建参数别名 |
| `memory_alloc(slots,width)` | 两个 1..16384 的整数常量，分配零初始化 f32[N,D] 工作区并返回专用句柄；禁止复制或当作张量 |
| `memory_read(bank,address)` | address:f32[N] 每项在 [0,1]，返回加权读取的 f32[D]；不隐式归一化，支持内存和地址梯度 |
| `memory_write(bank,address,value);` | value:f32[D]，每槽更新为 `(1-address[i])*旧值+address[i]*value`；支持地址、旧值和新值梯度 |
| `memory_free(bank);` | 释放工作区；未分配、重复释放或释放后访问报错 |
| `resource_state()` | f32[3]：剩余前向工作量、剩余工作区字节、存活工作区字节；无梯度 |

局部张量的长度由表达式推导，重复赋值不能改变长度；输入不可覆盖。
同一模型内 forward 与 layer 不混用。模型通过 `Name(optimizer)` 构造，optimizer 必填
`"none"`、`"sgd"` 或 `"adam"`；后两者允许声明式 `train` / `reinforce` 优化整个已执行 forward 的非冻结初始参数。
普通 `handle(input)` 执行并提交成功写入；失败回滚，步数计入外层操作预算。
无需 `register`、标签、`jump` 或 `branch` 源码指令，这些仅存在于内部 IR。

完整可执行示例：[self_modifying/model.wgt](../examples/self_modifying/model.wgt)。

固定 shape 多轴几何与 attention 算子合同（行为待验证）见[固定 shape 张量几何](tensor-geometry.md)。实际使用新几何算子或多轴 softmax 时使用 adaptive IR v3；旧 IR v1/v2 不因 AST 中出现新 intrinsic 而全定义升级。参数读取始终为 flat vector，矩阵参数须显式 reshape。既有 forward 入口为单个 f32 向量，矩阵输出须 reshape 回向量；typed 操作多输入的本地验收范围见[模型操作](model-operations.md)。完整模型示例在 [T10 语法总览注释](../language/syntax-tour/model.wgt) 与后端测试 fixture `crates/wgtime-core/tests/fixtures/attention/model.wgt`；本地验证为 core 7 suites / 30 tests、真实 service 集成 1 项和 macro fixture 编译 1 项通过；native-cpu attention 100 步 SGD 的 MSE 从 0.046146184 降至 0.00000076705595，并覆盖权重 roundtrip、同 SGD 续训和资源释放。核心补充 IR 安全校验后须回归。未验证完整 Transformer、多头 attention、动态 batch 或 GPU 路径。
可微路由训练示例：[soft_routing/model.wgt](../examples/native_training/soft_routing/model.wgt)，
完整训练、状态与保存约束见[可微程序](differentiable-programs.md)。上述三种算子的张量梯度仅用于 forward，
全部参数必填，无温度等隐式选项。`p=1` 表示完整通过或完全采用 a 的输出。
推理短路作用于直接写在 gate/route 内的表达式（支持嵌套），不延迟此前已执行的 let；
比较使用精确 0/1，无 0.999 等近似阈值。混合向量门不会按元素拆分专家执行。
forward 中也可写 `mix p { then { a } else { b } }`，等价于 `route(p,a,b)`；
分支内各为一个张量表达式，可选尾随分号。`mix` 和 `then` 是保留名称。
train/reinforce 和显式 VJP 记录梯度时仍计算两边，包括 frozen 门，以保留完整数学梯度。
短路示例：[lazy_routing/model.wgt](../examples/native_training/lazy_routing/model.wgt)。
路由选择梯度为 a-b，门梯度为 x；标量门梯度在特征轴求和。
语句 if 保持硬分支；表达式 if 的特定 learn 条件使用 STE，详见下文。源文件文本、宿主函数和 I/O 不因此可微。
事务、原生一阶自动微分、显式状态及预算细节见[自更新运行时](self-modifying-runtime.md)。

工作内存独立于参数，每次任务/训练样本开始为空，完成或失败释放，不保存到模型。
alloc/read/write 额外按 N×D 计费；释放不丢弃训练 tape 中的历史版本。
普通函数新增 `execution_start(model,input,max_work,memory_bytes)`、
`execution_resume(model,instructions)`、`execution_cancel(model)`，需 compute_budget，
仅针对 forward 推理；暂停保持状态，完成才提交写入，错误或取消丢弃事务。
完整形状、预算、返回字段、训练及生命周期契约见[工作内存与分步执行](working-memory.md)，
可运行例子为 [working_memory/model.wgt](../examples/native_training/working_memory/model.wgt)。
forward 工作内存、`resource_state()` 和普通脚本的 `execution_start/resume/cancel` 分段执行
见[临时工作内存与分段执行](self-modifying-runtime.md#临时工作内存与分段执行)。

## 静态用户训练策略

`train MODEL on INPUTS against TARGETS using STRATEGY;` 调用同编译单元的同步普通用户函数
`fn STRATEGY(model,inputs,targets)`，恰好三参，支持前向声明；拒绝内置函数、模型构造器、缺失函数、错误形参数与async函数。
目标静态绑定声明，不受局部同名值影响；三个实参从左到右各求值一次，策略只调用一次，并返回策略实际结果。
返回值限纯数据，允许 null，最大嵌套深度 32、计量 1 MiB，拒绝资源句柄；超参写在策略函数内，using与配置块不能混用。
策略自行决定differentiate/at/commit及更新次数，语句不隐式提交或重试；每个commit原子，整个策略函数不自动回滚，失败不撤销此前成功的显式提交。
完整例子 [train_strategy/model.wgt](../examples/native_training/train_strategy/model.wgt) 及限制见
[原生训练](native-training.md#静态训练策略-train--using)。本语法不使普通host函数自动可微，也不扩展GPU训练范围。

## 自定义优化器更新

CPU `forward` 模型可用 `differentiate(model,inputs,targets)` 显式求 `fn loss` 的批次平均损失与梯度；
要求模型使用 `"none"` 优化器。`gradient_snapshot(handle)` 返回 loss、parameters、gradients、names、shapes、state、steps。
普通 WGT 函数用列表/循环定义更新规则，`commit_gradients(handle,candidate_parameters,next_state)` 一次原子提交权重与用户状态，
返回更新前 loss 并释放句柄；`release_gradient(handle)` 放弃快照。所有参数必填，无默认算法或默认裁剪。
参数按稳定顺序包含全部非冻结参数，每个参数为行优先扁平 f32 列表；普通更新函数不自动可微。
最多16个存活句柄、共64MiB计量预算；状态深度最多32、计量大小最多1MiB，禁止嵌套资源句柄。
旧/已释放句柄、模型版本冲突、错误布局及非有限值报错，不部分提交。
`save checkpoint` / `load checkpoint` 的 forward v2 保存/恢复用户状态及步数，读取保留 v1 格式；`save model` 导出不含优化器的模型。
完整签名、预算计量、约束和续训示例见[原生训练](native-training.md#wgt-自定义更新算法cpu-forward)及
[custom_momentum/model.wgt](../examples/native_training/custom_momentum/model.wgt)。目前不支持此接口的顺序/GPU模型，
不自动执行算法、恢复训练循环或导入含循环的 host 模块。
`differentiate_at(gradient,candidate_parameters)` 在临时参数上使用首次句柄保存的同一 f32 批次重新求一阶梯度，
不写真实模型；返回的新句柄仍以原 parameters/state/steps 为提交基准，loss/gradients来自临时点。
它与原接口共享句柄/内存预算，拒绝过期版本；批次保留与临时绑定副本也计入预算。
[custom_sam/model.wgt](../examples/native_training/custom_sam/model.wgt) 用两次一阶求导与一次提交定义 SAM+SGD/动量；
这不提供高阶自动微分、SAM专用对象或GPU自定义优化。完整同批次/状态语义见[原生训练](native-training.md#用普通-wgt-编写-sam)。

## 模型组合与有界控制表达式

以下是 WGT 0.5 的本地实现，尚未发布。core 定点五组共 33/33 通过：adaptive、compiled_forward、control 10、std 5、learn 6；另有 architecture 6/6 和 conv 3/3 通过；真实 service 的 conv 3 项及 architecture 1 项通过，覆盖训练、权重恢复续训和释放。真实 control facade 1/1 已覆盖 native-cpu 子模型、三种 learn 数组、indexed repeat、if、blend 和半平方 loss，以及训练、权重回载、续训一致和释放归零。矩阵分支正反例及 5→5.5 已通过；解析或局部测试通过不等于模块导入与编译 forward 已全面验收。

`learn name: TYPE[N] = INITIAL;` 的 N 为 1..16384 的固定整数，可省略 `[N]` 表示标量；TYPE 为 `bool`、`float`、`float in low..=high`、`float > low` 或 `int in low..=high`，范围写在数组维度之后。INITIAL 必填，可以是同类型标量（填充全部项）或恰好 N 项的列表，逐项遵守标量类型和范围。数组维度必须紧跟类型名，如 `int[2] in 0..=3`，不能放在范围末尾；此语法及错误形式的拒绝均已实现并通过测试，syntax-tour T12 已解析通过。参数数组读取为 f32[N]，索引只能是编译期常量或 repeat 展开索引，不支持运行时动态索引。具体投影及整数限制见[可学习参数](learnable-architecture-parameters.md)。

`blend { weight => value; ... }` 至少一项、每项分号必填，无默认权重；每项计算 gate 后相加，权重逐项位于 [0,1]，shape 可为 [1] 或与值完全相同的门，值形状须匹配，不对权重归一化。`if condition { yes } else { no }` 的两个块各含一个表达式，末尾分号可省，两个结果形状须匹配。直接以 learn bool 或其固定索引为条件，以及 `learn_int >= 整数字面量`，使用硬前向和 STE 路由；其他受支持的标量比较、`positive(x)` 或 bool 字面量使用硬分支，不对分支选择求代理梯度。语句形式 `if` 保持硬分支。

`repeat depth [indexed i] carrying state = initial { body }` 必填深度、状态名、初值和单个 body 表达式，无默认值；body 末尾分号可省。depth 是编译期整数常量表达式或有界 learn int（可固定索引），范围必须在 0..=256。每次 body 使用上一轮 state 并产生同形下一状态，零次返回 initial；可选 i 是从 0 开始的编译期展开索引，必须与 carrying 状态名不同。learn 深度训练时按上界展开并计算候选，硬门用 STE 传递梯度；推理按硬门跳过未启用的内联候选。不是运行时动态索引或一般循环可微。

```wgt
wgt 0.5
model A {
input a: f32[1]
output b: f32[1]
parameter weight: f32[1] = [2]
fn forward(a) { return a * weight; }
}
model Root
input x: f32[1]
output y: f32[1]
blocks: A[2];
learn depth: int in 0..=2 = 1;
fn forward(x) {
    return repeat depth indexed i carrying h=x { blocks[i](h) };
}
fn loss(prediction,target) {
    return 0.5 * ((prediction-target) ** 2).mean;
}
```

`model A { ... }` 定义子模型，结束 `}` 须独占一行；根模型仍用 `model Root`。`name: A[N];` 声明 1..256 个实例，省略 `[N]` 为一个；`name[index](input)` 接受一个输入，索引必须在编译期确定且在界内。调用输入与返回结果必须符合子模型声明的输入和输出合同。不同索引的参数独立，同一索引重复调用共享参数。子模型 forward 当前仅接受不可变 let 和最终 return；子模型不能嵌套定义，loss 属于根模型。定义本身不初始化服务实例或训练权重。

张量 `literal * x` 或 `x * literal` 使用有限有符号数值字面量缩放，保持 shape；不允许把一般 f32[1] 张量当作广播标量。`x ** 2` 仅支持数字字面量指数 2，等价 `square(x)`；其他指数拒绝。除明确的 gate/route 门语义外，张量运算不隐式广播。

`std/attention` 和 `std/numerics` 不是预导入算法。显式写 `import attn from "std/attention"` 或 `import num from "std/numerics"`。`attn.dot_product_attention(q,k,v)` 接受 `[Tq,D]`、`[Tk,D]`、`[Tk,Dv]`，返回 `[Tq,Dv]`；q 由调用方预先缩放，不提供 mask、batch、多头或缓存。`num.relax(estimate,observation,gain)` 要求三个同形非空 f32 张量，返回 `estimate + gain * (observation-estimate)`，gain 不限制在 [0,1]。两者参数全部必填、无默认值和广播，不创建权重或状态；导入及服务路径仍需验收。

## 自定义算子与导入

`fn name(args) { let local = expression; return expression; }` 可在 forward/loss 中作为纯算子调用。
全部实参必填，按位置、从左到右求值一次；只接收/返回定长 f32 向量，无默认值或隐式广播。
不可捕获参数或调用者局部变量；模型参数须显式传入。函数体仅支持不可重复的 let 和最后一个 return，
可组合已有纯张量原语和其他纯算子，纯 helper 条件允许比较、`positive` 和布尔字面量；不支持递归、I/O、状态写入、循环或任意宿主函数。
支持的原语、shape/梯度契约、深度/展开上限见[自定义算子](custom-operators.md)。

库用 `operators NAME` 声明；`import alias from "./ops.wgt"` 导入本地库，
`import alias from "MODULE/LibraryName"` 导入已声明的包依赖，在 forward/loss 调用 `alias.function(args)`。
库不提供模型或训练入口；模型模块解析器中实现了嵌套导入，仍待行为验收。函数公开、路径与其他分发限制见同页。
导入与编译不运行算子；原生 IR 和 AOT 内嵌展开结果，保存后不依赖算子源码。
这是本地 0.5 实现，未发布；完整有效例子为
[custom_operators/model.wgt](../examples/native_training/custom_operators/model.wgt)。

## 实现与验证依据

- 编译单元和静态图计划：`crates/wgt-language/src/source.rs`。
- 共用函数语法：`crates/wgt-language/src/script/parse.rs`。
- 模型构造与普通函数运行：`crates/wgtime-core/src/script.rs`；源入口和 AST 在 `crates/wgt-language/src/script.rs`。
- 静态 forward IR：`crates/wgt-language/src/adaptive.rs`；执行、事务及梯度：`crates/wgtime-core/src/adaptive/`。
- 原生代码生成：`crates/wgtime-core/src/script/codegen.rs`。
- 回归：`crates/wgtime-core/tests/unified.rs`、`adaptive*.rs`、`script.rs`，
  `crates/wgt/tests/script_compiler.rs` 验证解释/AOT 的结果和操作数一致。

保留模型和实验在 `weights/`，不因升级版本头自动迁移或覆盖。详见[存储规范](storage.md)。

## 计算预算与原生能力

WGT 0.5 新增 `compute_budget(work,workspace_bytes,calls,deadline_ms)`、
`compute_status()`、`compute_capabilities()`、`compute_call(name,input)`、`compute_trace()`。
完整参数、计价、能力版本、失败语义和可运行示例见[计算预算](compute-budget.md)。
基础计算能力使用工作量、估算工作区和协作式截止时间额度；宿主授权与 Linux 进程限制见下文。

宿主可通过 `WGT_HOST_POLICY` 提供预算与只读文件/HTTP 端点授权；
`compute_host_status()` 返回累计读取量、请求次数、剩余额度和资源别名。
`file_read.v1`、`http_get.v1` 通过 `compute_call` 调用。启用策略会拒绝旧 I/O 内置接口，
权限与资源提供者共用初始化前固定的绝对截止时间；子预算失败只恢复父级上限与原截止时间，不退还已消费额度。HTTP 剩余不足 1 ms 时拒绝发出请求。完整兼容性、错误码和限制见[宿主策略](compute-budget.md#宿主策略与只读-io)。

Linux 宿主策略新增 `cpu_seconds`、`address_space_bytes`，由独立工作进程执行内核硬限制；
`compute_host_status()` 提供 CPU 时间与限制值。`file_write.v1` 通过输出别名新建文本文件，
累计写入与新建内容占用分别受宿主额度约束，不覆盖已有文件。参见[进程与磁盘契约](compute-budget.md#linux-进程硬限制与-cpu-计量)。

`compute_scope(function,args,limits)` 在九维子预算内执行 WGT 函数；子任务执行失败返回统一结果，消费计入父级，参数类型或目标函数无效则直接报错。
`compute_register(name,function,limits)` 注册单参数 `user.*` 能力，供 compute_call 调用。
参数、嵌套限制、权限与非事务语义见[子预算与自定义能力](compute-budget.md#子预算与自定义能力)。
成本感知原生策略梯度示例见[资源训练](compute-budget.md#成本感知强化学习)。

## 运行中权重的文件系统视图

Linux 训练、推理 CLI 及独立编译程序支持 [weightfs v1](weightfs.md)，在模型初始化后默认尝试挂载
内存权重视图。它是运行时能力，不增加语法或内置函数；检查和编译阶段不挂载。
环境变量 `WGT_WEIGHTFS=auto|on|off` 默认 auto。文件编辑校验形状和有限 f32，遵守冻结/共享
存储约束，在执行之间提交；冲突报错。持久化仍使用显式 `save model` / `save checkpoint` 语句，详见协议文档。

## GPU / NPU 资源调度

WGT 0.5 提供 `device_submit(model,input,alias)`、`device_wait(alias,job,timeout_ms)`、
`device_status(alias)`，使用宿主 devices 授权与有限队列，提交工作量同时消耗计算预算。
解释与 AOT 均支持；GPU 需构建 gpu 特性，NPU 需宿主注册原生后端。
完整签名、配额、失败、快照及支持范围见[GPU / NPU 设备资源](accelerator-resources.md)。

## 原生标量数学标准库

WGT 0.5 新增 `rsqrt(x)`、`fma(x,y,z)`、`hypot(x,y)`、`expm1(x)`、
`log1p(x)`、`sigmoid(x)`、`softplus(x)`、`silu(x)`，全部参数必填且为有限 f64。
原生直接调用、解释/AOT 共用实现；不表示同名图算子或张量自动微分已支持。
完整签名与上下文以 [std/prelude.wgt](../std/prelude.wgt) 为准；这些普通函数无需导入，不表示 forward/loss 支持同名张量计算。模块边界见[标准库](standard-library.md)。

向量标准库新增 `dot(x,y)`、`norm(x)`、`mean(x)`、`softmax(x)`、
`log_softmax(x)`、`logsumexp(x)`、`l2_normalize(x,eps)`、`mse(x,y)`、
`mae(x,y)`、`cross_entropy(logits,target)`。向量是非空一维有限 f64 列表；
双向量必须等长，eps>0，target 为合法零起始整数类别。参数无默认值。
返回形状、数值与预算边界及完整示例见[向量算法](standard-library.md#向量算法)。

### 互斥分类向量与 `expression.probabilities`

`output scores: logits[N]` 表示 N 个互斥类别的有限实数分数；`output choices: probabilities[N]`
表示已经计算好的类别概率。N 为固定整数 1..16384，只支持一个类别轴，不支持独立二分类 `logit[N]`。
这两种类型仅用于输出声明；输入、权重和普通参数声明仍用原有类型，标量 `prob` 的语义不变。
声明不会自动修改 forward 结果：logits 不自动 softmax，probabilities 不自动归一化。

模型调用返回的向量携带真实类型，局部变量、普通函数参数和返回值保留该类型；模型和训练断点保存、加载、
以及 AOT 也保留输出契约。带分类元数据的原生 Model/forward Program 使用 IR version 2；
普通 f32 仍写 version 1 且省略类型字段，新运行时读取旧 version 1 时默认为 f32。
旧运行时不能读取新分类制品，需要支持分类 IR v2 的运行时；这不是双向格式兼容。
v1 夹带分类元数据或概率验证指令、未知未来版本和任意未知字段都明确拒绝。
训练断点外层版本按既有契约，嵌入模型/Program 单独版本化。
`net(x)` 当前提交显式 `write`；规划中的 `decode` 将无梯度 tape 且丢弃 forward 写入，待实现验证。
索引返回普通数值，切片返回普通列表；改变向量的算术/列表操作不自动推断类别类型。

`scores.probabilities` 对 rank-1 普通 f32 分数或 logits 沿唯一类别轴执行减最大值的稳定 softmax，返回有类型的 probabilities；
例如 `linear(x,w,b).probabilities`。f32 即使数值恰为 `[0.2,0.8]` 也作为分数转换，不按数值猜测类型。
已是 probabilities 时逐项验证后保持原值，不重复 softmax、不静默归一化。每项必须有限且严格在 [0,1]，
总和使用 f64 累加，与 1 的绝对误差不得超过 **1e-5**；此容差覆盖最大支持宽度的 f32 softmax 舍入。
空向量、非有限值、错误形状和非法概率都明确报错；不对矩阵隐式选轴或展平，显式数学函数 `softmax(x)` 保留。
转换结果经局部变量、纯张量 helper 和链式 `.probabilities.probabilities` 保持概率类型，后续后缀只验证。
现有普通函数的非空有限一维数字列表采用同样的分数转换与类型保持规则（数值存储为 f64）。
后缀与索引从左向右结合，接在完整调用或括号之后，输入只求值一次，不受同名局部变量遮蔽。

forward/loss 的预测参数保留输出类型，后缀转换可微：f32 分数和 logits 使用 softmax Jacobian，probabilities 验证为恒等梯度。
`log_softmax(prediction)` 仍直接消费 logits，交叉熵原有输入契约不变，不强制先 softmax 再 log。
显式概率采样按实际总质量缩放一次 RNG 阈值，跳过零质量类别；这不修改存储向量。
CPU 路径已实现，GPU 新类型路径未验证。完整 CNN 示例见 [cnn_probabilities/model.wgt](../crates/wgtime-core/tests/fixtures/cnn_probabilities/model.wgt)。

```wgt
wgt 0.5
model Categories
input x: f32[3]
output scores: logits[3]
fn forward(x) { return x; }
fn main() {
    let net = Categories("adam");
    let scores = net([1000, 1000, -1000]);
    return scores.probabilities;
}
```

### 后缀平均 `expression.mean`

WGT 0.5 本地实现支持 `x.mean`，等价于 `mean(x)`；原函数形式继续支持。
没有参数、默认参数或隐式轴选择：普通函数接受非空一维有限 f64 列表，返回 f64 标量；
forward/loss 接受 f32[N]（N>0），沿唯一向量轴归约为 f32[1]，梯度为上游梯度除以 N。
空列表、非数值列表、嵌套列表和标量不接受；不隐式展平或广播。

后缀与索引从左到右结合，优先于一元负号和全部二元运算：`-x.mean` 是 `-(x.mean)`，
`(a-b).mean` 先做张量减法再平均，`rows[0].mean` 先取第一行再平均。
操作数只求值一次，解释、AOT、预算计费、自动求导及制品使用与 `mean(x)` 相同的实现。
`pkg.mean(x)` 仍是限定名称函数调用；`.mean` 不提供方法调用或通用对象属性机制。
当前只实现 `.mean`，不支持 `.sum`、`.min`、`.max` 或 `.mean[axis=1]`。

```wgt
wgt 0.5
fn main() {
    # 一维样本列表归约为一个标量；没有训练或模型初始化。
    let scores = [2, 4, 6];
    assert scores.mean == 4, "mean";
    return scores.mean;
}
```

可微前向与显式训练的完整例子见
[`postfix_mean/model.wgt`](../examples/native_training/postfix_mean/model.wgt)。这是本地语言能力，不表示已发布版本。

统计与排序新增 `min(x)`、`max(x)`、`variance(x,ddof)`、`stddev(x,ddof)`、
`median(x)`、`quantile(x,q)`、`sort(x)`、`argsort(x)`、`argmin(x)`、`argmax(x)`、
`cumsum(x)`、`clip(x,lower,upper)`、`cosine_similarity(x,y)`。
输入沿用非空有限 f64 一维列表；所有参数必填。ddof 是 0<=ddof<N 的整数，
q 在 [0,1]，裁剪边界有限且有序；余弦要求等长非零向量。
排序稳定，极值索引取首次。完整签名、复杂度、数值边界和预算见
[统计、排序与变换](standard-library.md#统计排序与变换)。

分段激活、损失与信号新增 `relu(x)`、`leaky_relu(x,alpha)`、`elu(x,alpha)`、
`hard_sigmoid(x)`、`hard_swish(x)`、`bce_logits(logits,targets)`、`huber(x,y,delta)`、
`log_cosh(x,y)`、`diff(x)`、`ema(x,alpha)`、`moving_mean(x,window)`。
激活处理有限 f64 标量；其余接受非空一维有限 f64 列表。所有参数必填；
完整参数范围、形状、公式与计费见[分段激活、稳健损失与信号处理](standard-library.md#分段激活稳健损失与信号处理)。
这些普通脚本函数不新增声明式图层支持、张量梯度或训练目标。

## 概率与分支语法的完整契约

`parameter p: prob = 0.7`、`frozen p: prob = 1`、`mix p { then { a } else { b } }`、
`sample p { then { a } else { b } }` 及 `fn loss(prediction,target)` 的参数、表示、梯度、
执行边界和可运行例子统一见[可微程序语法](differentiable-programs.md#概率混合自定义损失与采样语法)。
forward/loss 新增 `square(x)`（等长平方）、`mean(x)`（唯一向量轴平均为 f32[1]）、
等长逐元素 `a-b` 和 `-a`；全部无默认参数或广播。自定义损失通过训练配置块的 `loss: loss` 调用，
其表达式与模型共同求导；普通 sample 块不建立梯度图，forward/loss 中明确拒绝。

数据标准库新增 `zeros(count)`、`ones(count)`、`full(count,value)`、
`linspace(start,end,count)`、`one_hot(index,classes)`、`concat(x,y)`、
`vector_slice(x,start,end)`、`gather(x,indices)`、`compress(x,mask)`、`reverse(x)`。
全部参数必填；输入为一维有限 f64 列表，数据操作允许空列表，掩码严格为 bool。
索引为零起始整数，切片左闭右开；单次输出不超过 100 万项。
完整参数范围、输出形状、预算和运行例子见[数据构造与索引](standard-library.md#数据构造与索引)。
这些是普通脚本列表操作，不是 forward/loss 张量算子或零拷贝视图。

`onehot[N] index` 是普通函数中的独热表达式，如 `onehot[4] 2` 得到 `[0,0,1,0]`。
N 为 1..1000000 整数字面量，index 是运行时 [0,N) 整数；优先级同一元负号，复杂索引用括号。
原生等价调用 `one_hot(index,classes)` 支持运行时类别数。无默认值，不对索引求导，forward/loss 不支持。
完整边界、预算与示例见[独热编码](differentiable-programs.md#独热编码)。

概率分析新增 `normalize_probabilities(weights)`、`entropy(p)`、`kl_divergence(p,q)`、
`js_divergence(p,q)`、`cross_entropy_probs(p,q)`、`temperature_softmax(logits,temperature)`。
所有参数必填，输入非空一维有限 f64 列表；双分布必须等长，温度有限且>0。
分布逐项在 [0,1]，总和允许 1e-9 绝对误差，计算前修正该误差；任意权重须显式归一化。
零概率、公式、预算和例子见[概率分布分析](standard-library.md#概率分布分析)。
这些是普通脚本分析函数，不提供张量自动微分或新的训练目标。

现有路由示例优先使用 `mix`；独立可训练标量门使用 `prob`。固定类别数的独热编码优先写 `onehot[N] index`，动态类别数保留 `one_hot(index,classes)`。

## 原生 RL 训练语法

当前 WGT 0.5 内置训练使用 `train MODEL on INPUTS against TARGETS {rate:RATE; loss:LOSS; clip:MAX_NORM;}`；
`rate` 必填且为有限正 f32，`loss` 默认 mse，可选 mse/cross_entropy/loss，`clip` 可选。
表达式形式返回更新前 f32 批次平均损失。自定义 `fn loss` 和全局 L2 梯度裁剪目前仅 CPU forward 训练支持；裁剪后原子提交一次更新，失败不改变参数或优化器。
`target NAME: f32[N]` 可指定损失第二参数宽度（1..16384），NAME 须匹配 fn loss 第二形参；
未声明时沿用输出宽度。它不改变 forward 输入，也不改变内置 MSE 的目标契约。
新增 forward/loss 数字向量字面量、detach、exp、log、log_softmax、sum、minimum、clip。
完整签名、形状/梯度/边界/版本约束和完整可运行例子见[原生强化学习](reinforcement-learning.md)。
普通脚本的数值/列表函数不因此自动可微；不支持单独 backward/step 或跨调用梯度对象。

## 原生测试与评估

`wgt test` 原生发现 tests/*.wgt 的无参 test_* 函数，逐测试隔离状态与预算；
支持过滤、行号诊断、JSON 共存、JSONL 评估以及 CPU/GPU 前向和 CPU 训练步计时。
AOT 测试入口明确拒绝；完整参数、版本与限制见[原生测试](native-testing.md)，
可运行项目见[评估示例](../examples/native_evaluation/README.md)。

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
