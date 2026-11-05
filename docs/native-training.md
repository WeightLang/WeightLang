# 原生 WGT 训练脚本（本地实验实现）

模型/训练程序的源码入口统一为 `model.wgt`，执行函数仍为 `fn main()`。
独立示例按 `<示例名>/model.wgt` 组织，见[示例目录](../examples/native_training/README.md)。
相对数据与保存路径按入口所在目录解析；移动入口时须一并调整这些路径。
原生专用 `agent` / `environment` 声明、默认 `learn actor { ... }` 整批采集/自定义学习器与同步环境生命周期见
[agent/environment](agents-environments.md)，完整监督例子 [agent_environment/model.wgt](../examples/native_training/agent_environment/model.wgt)。
环境反馈不限定奖励，learner使用现有train或显式梯度提交；无整实例checkpoint或异步环境支持。
命令签名中的 SOURCE/FILE.wgt 表示路径占位符，模型入口应传入实际的 `model.wgt` 路径。

目标职责分工是：Rust 主程序负责数据准备、训练循环和业务控制，WGT `ModelDefinition` 负责模型内部计算，所有模型计算由唯一 `wgtime` service 执行。模型定义、运行时实例、权重和 checkpoint 分离；解析定义不创建实例，不初始化权重或训练状态，不训练、不写文件。

首期定义 API 提供纯 `ModelDefinition` 的 `parse` / `from_file`，以及接受 Rust 字符串字面量的 `model!` / `include_model!` 宏。三种入口共用 parser；纯计算模块导入已在源码接入，行为尚待完整验收。宏在编译期验证后嵌入源码；运行时构造定义仍是纯解析，不是预编译 IR。此 facade 已通过本地定点验证（语言 4 项、宏 7 项、facade 2 项，含真实三入口推理/训练/weights/drop 及删除源码后无工具链运行），但尚未发布；这不代表全仓、全 GPU 或性能验收通过。本文其余现有 WGT 训练语法描述当前脚本接口，不代表新 facade 具备相同 API。
## 多入口操作与训练策略

本地指定范围已验收，尚未发布：core 7 个 suite 共 31 项、typed import roundtrip 1/1；最终 daemon 重建后真实 IPC 5/5（operations 1、既有 service 1、CNN 3）。language 与 model 常规测试 16 项通过、8 项 ignored，相关服务测试单独执行。workspace all-targets 检查、typed `model!` 宏编译和完整示例解析均通过；解析不证明执行。覆盖范围及完整记录见[模型操作验收状态](model-operations.md#验收状态)，不代表全仓验收。

多入口 typed 模型可不声明 loss 或优化器；每个输入为固定 `f32[N]`（N=1..16384），总输入宽度不超过 16384，返回固定 `f32[N]` 或 void。它以原生代码执行，不使用 VM，也不支持 AD 或跨状态求导。同实例入口共享参数及独立状态，写后读应看到候选值，全部成功才原子提交，失败保持原值；普通权重不含状态。

模型定义出现 typed 签名或 `state`，或没有 `forward` 但有无参函数时进入操作模式。无参且无返回类型的函数是 void 入口而非纯 helper，不能返回张量；无参且有返回值须显式标注，如 `fn observe() -> f32[1] { ... }`。纯 helper 须至少有一个无类型标注参数。

旧 forward 训练路径的 `train_with_policy` 提供 `ContinuousOnly` 和 `StraightThrough`：前者在整个模型存在任意 `learn bool` 或 `learn int` 时拒绝，即使当前路径未读取；后者采用直通估计（STE）。旧 `train` 保留 STE 兼容行为。当前 facade 包含 `call`、`operations`、`parameters`、`state`、`update_parameters` 和 `train_with_policy`，签名见[Rust 后端契约](native-backend.md#多入口操作)；策略检查与训练由单个 `TrainStepWithPolicy` 服务请求原子处理，不持久设置策略。参数候选拒绝 frozen/probability 参数，bool/int 候选必须严格离散；没有 CAS 或状态 snapshot，revision 仅在服务响应中返回，不承诺拒绝过期候选。`update_parameters` 成功会清空旧 forward 优化器，失败保持原状态，重新训练需通过 `load_weights` 显式选回优化器。typed 操作无优化器和 AD，加载权重保留 state。语法及限制见[模型操作](model-operations.md)。

## 统一训练入口契约（本轮实现中）

监督训练使用同一个表达式入口，作为语句时忽略返回值：

```wgt
let loss = train net on inputs against targets {loss:mse; rate:0.01;};
train net on inputs against targets {loss:loss; rate:0.01; clip:1.0;};
let metrics = train net on inputs against targets using momentum;
```

内置更新返回本次更新前的批次平均标量损失；不自动打印或包装成 metrics。`rate` 必填且正有限；`loss` 默认 `mse`，可选 `cross_entropy` 或使用模型 `fn loss` 的 `loss`。`clip` 省略表示不裁剪，提供时必须正有限；当前仅 CPU 训练支持，对批次平均梯度做全局 L2 裁剪后原子提交，不是逐元素截断。没有默认学习率，也不重新选择模型已经配置的优化器。未知字段、重复字段和缺失必填项在解析时拒绝；损失、形状与模型能力不匹配在执行时明确报错。

模型、inputs、targets、配置字段表达式按源码顺序各求值一次；未执行分支不训练。配置中的 loss 名是固定选择，不是字符串求值或任意宿主回调。成功内置更新只有一次提交，失败保持权重、优化器状态和步数；执行工作仍计入预算。检查与编译不执行训练。

`using strategy` 保留用户定义同步 WGT 策略和既有返回值，不限制为 SGD/Adam，不隐式插入更新。策略函数接收 `(model, inputs, targets)`，自行管理 differentiate/候选参数/commit/release；SAM 可保留多次求梯度与显式提交语义，策略中已成功提交的更新不会因后续错误自动撤销。默认轻客户端向服务注册策略，在服务内处理参数、梯度和优化器状态，客户端只接收原有指标，不传回梯度计算。内置配置块与 `using` 互斥。

这是破坏性升级：旧 `fit(...)`、`train(...)` 以及 `train net on (inputs,targets) rate ... clip ...` 不再是合法源码；真实例子与回归随实现迁移。上述为本轮实现契约，下面尚未迁移的旧签名不作为新语法支持证明；完成后删除旧签名并记录实际验证范围。

当前只保留原生训练路径；旧 Python 训练入口已删除。
当前不是全部模型算子已经迁移，也不是已发布功能。

forward/loss 支持[原生自定义算子与分发](custom-operators.md)：同文件纯张量函数或
`operators NAME` 库经显式 import 在编译时展开，继承原语梯度、保存重载和解释/AOT 行为。
库导入不执行训练；完整可运行例子为
[custom_operators/model.wgt](../examples/native_training/custom_operators/model.wgt)。

```sh
cargo build -p weightlang-wgt
./target/debug/wgt run examples/native_training/supervised/model.wgt --check
./target/debug/wgt run examples/native_training/supervised/model.wgt
./target/debug/wgt run examples/native_training/reinforce/model.wgt
```

`--check` 只检查脚本结构、版本、嵌套深度与重复函数/形参，不读取数据、不构造模型、不运行调用。
它不做完整类型检查或证明未执行分支中的名称/调用有效。模型与实参在执行时检查。
`wgt run --max-operations N` 默认 10000000，限制解释器表达式、语句、循环和调用的执行次数；
这不是超时、内存限额或内置训练算子的 FLOP 限制。内置文件操作可能阻塞。
历史 `wgt-native script` 本地实现入口已随执行服务拆分移除；脚本运行请使用 `wgt run`，模型执行接口由 wgtime client/service 提供。

## 显式分类输出

互斥类别输出可声明 `output scores: logits[N]` 或 `output choices: probabilities[N]`（N=1..16384）。
`net(x)` 的返回值跨局部变量、函数参数和返回值保留类型，模型/训练断点保存加载与 AOT 也保留声明。
`scores.probabilities` 对 rank-1 f32 分数或 logits 做减最大值的稳定 softmax，可直接写 `linear(x,w,b).probabilities`；已是 probabilities 时验证并保持原值。
不按数值猜概率：普通 f32 `[0.2,0.8]` 仍做 softmax；转换结果保留概率类型，重复后缀仅验证。输出声明本身不执行 softmax。
矩阵、空向量、非有限值明确拒绝；f32/logits 复用 softmax 梯度，已是概率使用恒等梯度。
概率每项须有限且在 [0,1]，f64 累加的总和与 1 绝对误差不超过 1e-5，不静默归一化。
模型类别采样目标形式为 `decode policy(x) {method:sample;}`；其 logits/probabilities 行为、RNG 与事务契约待 parser/runtime 实现后补齐。
普通 f32 的显式 `softmax(x)` 保留；独立二分类 logit 类型未实现。
后缀转换支持 CPU 张量 loss 和梯度，`log_softmax(prediction)` 与交叉熵仍直接消费 logits，原有损失契约不变。
GPU 新类型路径尚未验证。详见[分类向量参考](reference.zh-CN.md#互斥分类向量与-expressionprobabilities)，
完整例子为 [cnn_probabilities/model.wgt](../crates/wgtime-core/tests/fixtures/cnn_probabilities/model.wgt)。

## 语法与值

首个非空、非注释行必须为 `wgt 0.5`。模型声明与下列函数属于同一种语言；完整模型声明和 forward 契约见[0.5 参考](reference.zh-CN.md)。旧模型、script、adaptive 版本头均拒绝。
花括号定义语句块，简单语句以分号结束，`#` 到行尾是注释。字符串使用 JSON 双引号转义。
程序顶层可包含模型声明和函数定义；执行入口必须为零参数 `fn main()`。

```wgt
wgt 0.5
fn score(x) {
    if x > 0 { return x * 2; } else { return 0; }
}
fn main() {
    let total = 0;
    for i in range(0, 10) {
        if i == 2 { continue; }
        if i == 5 { break; }
        total = total + score(i);
    }
    while total < 20 { total = total + 1; }
    assert total == 20, "unexpected control flow";
    return total;
}
```

- `let a = expression, b = a + 1;` 支持换行，依次求值并绑定；后项失败不撤销前项副作用。
  实参/列表/记录内的逗号不分隔声明；无尾逗号。解构采样、collect/learn 声明须独立书写。
  forward/loss 同样支持普通声明组，保留原有 shape 与重名规则。
  完整训练例子见 [continuous_let/model.wgt](../examples/native_training/continuous_let/model.wgt)。
- `let name = expression;` 在当前函数环境创建或重新绑定变量；`name = expression;` 要求变量已存在。
  块不另建作用域，循环变量执行后保留；函数有独立环境，没有全局变量、闭包或捕获。
- `fn name(arg, ...) { ... }` 参数按位置传递；`return expr;` 返回值，`return;`/自然结束返回 null。
  支持前向函数引用与有界递归；无默认参数、关键字参数或可变参数。不能覆盖内置函数名。
- `if bool { ... } else { ... }`、`while bool { ... }`、`for name in list { ... }`；else 可省略。
  `break;` / `continue;` 只作用于最近的循环；不支持 `else if` 简写，可嵌套 if。
- `let batch = collect { ... yield {field: expression, ...}; ... };` 采集独立记录批次；这是声明形式，不是任意表达式。
  `batch.field` 投影列，`batch[i]` 取记录、`batch[i].field` 取字段。`for part in batch size 16 { ... }` 按序共同分块，保留尾批，空批零次；size 只求值一次且为正整数，不默认洗牌。
  首行确定字段及递归固定形状，后续行须一致；yield 左到右求值后整条追加，不部分写入。细节及上限见[记录批次](reference.zh-CN.md#记录批次与对齐小批次)。
- 值为有限 f64、bool、字符串、null、列表、记录批次、JSON 对象、模型/数据流句柄。
  JSON 对象来自数据和指标结果，当前无对象字面量。列表按值传递；句柄副本指向同一资源。
  原生训练张量为 f32，转换溢出报错。
- `a[i]` 是非负整数列表索引，`row["input"]` 是对象索引。`a[start:end]` 是左闭右开列表切片，
  省略 start/end 时分别默认 0/长度；端点可为运行时整数，允许空列表，不裁剪越界端点。
  forward/loss 支持一维张量切片，但要求编译期数字端点且结果非空；旧 `slice(x,start,length)` 保持长度语义。
  不支持步长、多轴、负索引或下标赋值。详见[统一括号切片](reference.zh-CN.md#统一括号切片-xstartend)，
  完整可运行例子为 [slicing/model.wgt](../examples/native_training/slicing/model.wgt)。
- 从低到高：`||`、`&&`、`== !=`、`< <= > >=`、`+ -`、`* / %`、一元 `- !`、索引。
  `&& ||` 短路并要求 bool；if/while 不隐式把数字或张量转成 bool。
  `+` 支持数字加法或字符串连接。相等比较按值/句柄身份；数值运算不允许非有限结果。
  数字除以零或对零取余产生的非有限结果会报错；不会自动替换除数或将结果截断。
- 最大 100000 token、64 层语句块、128 层表达式树、64 层函数调用。
  错误带脚本路径、执行语句行号；无异常捕获、宿主语言执行或 FFI。原生 async fn/tasks/await/spawn 的第一阶段支持与限制见[异步任务](async-tasks.md)；普通函数不会隐式异步。

## 内置接口

新增 forward 工作内存及普通函数的 `execution_start(model,input,max_work,memory_bytes)`、
`execution_resume(model,instructions)`、`execution_cancel(model)` 见[工作内存与分步执行](working-memory.md)。
这三个接口不训练，活动事务期间拒绝同一模型的 fit/reinforce/predict/sample；
内存读写本身支持完整 forward 训练，示例为 [working_memory/model.wgt](../examples/native_training/working_memory/model.wgt)。

除表内明确标为可选的参数外，其余参数均必填；可选参数的默认值见表。相对路径相对脚本文件目录。
完整内置目录在 [language/prelude.json](../language/prelude.json)，同时驱动运行时名称与参数数量检查、编辑器快照；领域实现仍保留在目录所指的模块中。

普通函数中保存模型使用 `save model VALUE to PATH;`，保存训练断点使用
`save checkpoint VALUE to PATH;`。读取表达式为 `load model from PATH using OPTIMIZER` 和
`load checkpoint from PATH`；前者必须显式指定新优化器，不恢复训练状态，后者使用断点保存的优化器、步数和矩。
这些语句默认拒绝覆盖已有目标，路径父目录须预先存在。完整文件语法与边界见[file-io.md](file-io.md)。
显式模型加载、checkpoint 保存与恢复只使用上文声明式语法；未列出的旧函数拼写不属于 WGT 0.5 当前接口。
safetensors 是独立权重导入/导出链路，仅承载权重张量，不包含模型定义、实例状态、优化器、训练步数或完整 checkpoint；它不构成 PyTorch 训练状态互通。更高层的模型与权重绑定 API 仍在实现和契约确认中，具体签名与事务语义待接口冻结后补充。

| 签名 | 契约 |
| --- | --- |
| `len(list)` | 列表长度 |
| `range(start,end)` | 非负整数半开区间；end >= start，最多 1000000 项 |
| `append(list,value)` | 返回追加后的新列表，不原地改动；最多 1000000 项 |
| `column(rows,key)` | 从对象列表逐行提取字段，缺失即报错 |
| `sum(numbers)` | 求和，空列表为 0 |
| `sqrt(x)`, `log(x)`, `exp(x)` | 标量数学；域错误/溢出报错 |
| 原生 assert 语句 | `assert CONDITION;` 或 `assert CONDITION, MESSAGE;`；严格 bool，失败时才求消息 |
| `metric(name,value)` | 按名称保存 JSON 指标；同名覆盖，只保留最后值，不自动平均或 detach |
| `seed(integer)` | 设置脚本局部采样 RNG；整数范围 [0,1000000000] |
| `random()` | 确定性 SplitMix64 流，返回 [0,1)；不影响模型初始化 RNG |
| `mlp(widths,activation,seed,optimizer)` | 新建 CPU f32 顺序网络；widths 是包括输入输出的 2..64 个整数，每个 1..16384，最多 4000000 参数；隐藏层 activation 为 relu/tanh/sigmoid/identity，输出层始终 linear；optimizer 为 sgd/adam |
| `load model from PATH using OPTIMIZER` | 加载原生顺序 IR 或 forward Program IR JSON，创建新优化器；不是普通 .wgt 模型解析器，也不接受传统 manifest 制品 |
| `net(input)` | 输入单个 f32 向量，返回输出列表；不隐式训练，forward 模型可通过显式 `write` 提交参数更新 |
| `train model on inputs against targets { ... }` | 使用上文统一训练配置块；顺序模型支持 SGD/Adam，forward 自定义损失与 `clip` 目前仅 CPU 支持，见[原生 RL](reinforcement-learning.md) |
| `reinforce(model,observations,actions,advantages,lr)` | 一批观测的策略梯度更新，返回更新前的 mean(-advantage*log pi(action)) |
| 类别采样 | 目标语法为 `decode policy(input) {method:sample;}`；等待 parser/runtime 实现后补齐完整契约 |
| `gae(rewards,values,next_values,terminated,truncated,gamma,lambda)` | 同长度一维列表；后两种标记是 bool，gamma/lambda 在 [0,1]；返回优势列表 |
| `hf_dataset(dataset,config,split)` | Rust 原生 Hugging Face 分页数据流；全部参数必填，首次 batch 联网；HF_TOKEN 可选，详见 [数据集契约](datasets.md#原生数据集加载器wgt-05当前源码) |
| `jsonl(path)` | 打开 UTF-8 JSON Lines 数据流；每行一个 JSON 值，不允许空行 |
| `batch(reader,count)` | count 在 1..65536；返回最多 count 行；尾批保留，到 EOF 返回 []；单行 <=1MiB、单批 <=16MiB |
| `close(reader)` | 关闭流，重复关闭无操作；关闭后 batch 报错 |
| `save checkpoint VALUE to PATH;` | 保存顺序或 forward 训练断点：结构、权重、冻结关系、优化器类别、步数和矩状态；顺序模型另保留共享关系 |
| `load checkpoint from PATH` | 读取断点，返回模型及恢复后的优化器 |
| `save model VALUE to PATH;` | 导出原生推理 IR JSON；不包含优化器；forward 使用 Program 格式 |
| `benchmark(model,workload,warmup,repeats,name[,mode])` | mode 默认 cpu；支持同步 GPU 前向和 train:cpu MSE 训练步；warmup 0..10000，repeats 1..100000；完整契约见[原生测试](native-testing.md#benchmark) |

脚本整数参数目前限制为 [0,1000000000]；列表索引仍需满足实际长度。
每次脚本最多创建 64 个模型、1024 个流句柄（包括已关闭的流）。运行结束自动释放所有资源。
每个模型 prepare 的张量预算是 256 MiB，**不包含**解析、脚本列表、训练 tape、梯度、优化器和保存副本；
训练不是零分配。不要把该值当成进程内存上限。

## 梯度与参数更新

声明式顺序模型支持单向量 linear/relu/tanh/sigmoid/identity；`fn forward` 模型还支持
可微 gate/route/softmax、显式写回与实际执行的循环/分支计算，详见[可微程序](differentiable-programs.md)。
两类模型均可用声明式 `train` / `reinforce` 与 SGD/Adam；训练输入是向量的列表，逐样本
建立 tape 并累加梯度。训练裁剪仅支持 CPU。MSE 在样本及输出维度平均；交叉熵 targets 是每样本一个类别整数，
输入模型应输出未经 softmax 的 logits。REINFORCE 的 advantages 是显式常量，不对优势求导。

SGD 无 momentum/weight decay；Adam 固定 beta1=0.9、beta2=0.999、eps=1e-8，无 weight decay。
同一模型句柄保留优化器矩和步数，学习率在每次更新显式指定。当前不支持独立优化器对象、
混合精度、独立梯度累积接口、任意自定义损失反传或 forward 模型的 GPU 训练。顺序模型可用 training_device 选择 GPU；forward 模型支持受限的
`fn loss(prediction,target)` 一阶反传，约束与示例见[可微程序](differentiable-programs.md)。

原生 IR 的 resident/view/add 梯度汇总到存储所有者，共享参数只更新一次；冻结依赖沿用 Session
规则，冻结参数不更新。可训练的 file 存储拒绝，冻结 file 可读取。全部批次与候选更新检查通过后
再提交权重和优化器状态；非有限输入、损失、梯度、参数/矩状态均报错，不部分更新。
`mlp` 使用独立确定性 Xavier uniform 权重、零 bias；不承诺与 Python/PyTorch 的 seed 初始化逐位相同。

## 静态训练策略 `train ... using`

```wgt
train net on inputs against targets using momentum;
```

`momentum` 必须是同编译单元声明的同步普通用户函数，签名为 `fn momentum(model,inputs,targets)`，
恰好三个必填形参。函数可以在调用后声明；内置函数、模型构造器、缺失名称、错误形参数和 async 函数均拒绝。
`using` 绑定声明函数，局部同名变量不改变目标；不提供函数值或捕获。
模型表达式、inputs、targets 从左到右各求值一次，策略只调用一次，return 值由语句丢弃。
超参数在策略函数内显式定义；`using` 不能与配置块混用。

策略自行调用 differentiate/differentiate_at/commit_gradients，可选择零次、一次或多次更新。
高层语句不添加优化器、梯度裁剪或额外提交，不隐式重试；只有每次显式 commit 是原子的，
策略函数整体不是自动回滚事务，后续错误不撤销之前成功提交。普通host策略不自动可微。
此语法也不扩展底层自定义优化器的CPU forward能力范围。
完整可运行例子为 [train_strategy/model.wgt](../examples/native_training/train_strategy/model.wgt)。

## WGT 自定义更新算法（CPU forward）

`differentiate(model,inputs,targets)` 要求 CPU `forward` 模型以 `"none"` 构造，且定义受限可微 `fn loss`。
它计算更新前批次平均损失及非冻结初始参数梯度，不提交 forward 写入，也不更新权重或优化器状态。
普通 host 函数仍不自动可微；用户用现有循环、列表、索引和标量数学编写更新公式。
完整例子为 [custom_momentum/model.wgt](../examples/native_training/custom_momentum/model.wgt)，将 beta 改为 0 即 SGD。

| 签名 | 契约 |
| --- | --- |
| `differentiate(model,inputs,targets)` | 返回不透明梯度句柄；inputs/targets 为逐样本 f32 向量列表，非空且数量相同，形状遵循 forward/loss。参数与梯度均为更新前快照；无隐式优化器或裁剪。 |
| `differentiate_at(gradient,parameters)` | 使用该句柄保留的原批次，在临时绝对参数上重新求一阶梯度，返回新句柄。真实模型不写入；新快照 parameters/state/steps 保持原提交基准，loss/gradients 是临时点结果。拒绝过期句柄或不匹配候选。 |
| `gradient_snapshot(gradient)` | 返回对象，字段为 `loss`、`parameters`、`gradients`、`names`、`shapes`、`state`、`steps`。parameters/gradients 按稳定参数顺序，仅含非冻结参数；每个参数是行优先扁平列表。state 初始 null，steps 是已提交自定义更新次数。 |
| `commit_gradients(gradient,parameters,state)` | parameters 是全部可训练参数的候选绝对值，与快照顺序/长度一致。state 为用户定义的 JSON 兼容数据。检查全部完成后原子替换参数和状态，增加一次 steps，返回快照 loss 并释放句柄。 |
| `release_gradient(gradient)` | 放弃快照并释放句柄，返回 null；已释放/提交的句柄再次使用报错。 |

最多 16 个存活梯度句柄，总计量预算 64 MiB；释放/成功提交后立即归还句柄预算，Runtime 结束释放全部资源。
计量包含参数、梯度、候选预留、保留的 f32 输入/目标批次及元数据/状态；临时求导还预留参数绑定副本，剩余预算用于单样本 forward/loss tape；求导结束不保留 tape。
它不是进程总内存上限，普通列表副本、解析及分配器开销不在此上限内。
求导扣除 forward/loss 指令工作量，以及按 tape f32 存储项和参数项计量的反传/归约工作；预算失败不退款。
普通更新函数继续受脚本操作预算限制。

状态允许有限数字、bool、字符串、null、列表及对象，禁止任何层级的模型/梯度等资源句柄；最大深度 32，
按每节点 16 字节加字符串/对象键字节计量，最多 1 MiB。参数转换必须是有限 f32。
错误形状、非有限候选、状态超限、活动执行事务、过期快照（包括 weightfs 修改后）均拒绝提交，参数和状态保持原值。
冻结参数不可更新；同一参数的重复使用梯度由现有 VJP 合并。自定义路径目前不接受顺序模型或 GPU。

`save checkpoint` 在首次自定义提交后写 `weightlang.forward-training.v2`，包含模型、用户状态和自定义步数；
`load checkpoint from PATH` 同时接受旧 v1 与新 v2，严格拒绝未知断点字段。恢复不自动调用更新函数或训练，
调用者需继续显式使用同一算法及状态版本；`save model` 不含优化器状态。
数据游标、环境和 RNG 不自动恢复。目录须预先创建，断点仍拒绝覆盖已有文件。

```sh
mkdir -p weights/custom_momentum
./target/debug/wgt run examples/native_training/custom_momentum/model.wgt
```

### 用普通 WGT 编写 SAM

[custom_sam/model.wgt](../examples/native_training/custom_sam/model.wgt) 先求原始梯度 g，
用 `rho*g/(sqrt(sum(g*g))+eps)` 构造临时参数，再调用 `differentiate_at`。
第二次的梯度交给 WGT 自定义基础优化器，从原始 w 计算候选并只提交一次；两次求导均不更新权重、速度或步数。
没有 SAM 专用关键词/对象，也不对第一次梯度再次求导；这是两次一阶求导，不是高阶自动微分。
范数仅遍历快照中的非冻结唯一参数；重复参数使用已经由 VJP 汇总，零梯度扰动为零。
第二次失败或版本改变不会提交任何更新，首次句柄仍可释放；成功提交后其余同版本句柄过期，须释放。

首次求导保存转换后的 f32 批次，二次不能传入另一批次，也不会重新执行 host RNG/环境采样。
每个样本从相同模型初始状态出发，forward write 只在本次 tape 内参与求导，不提交；冻结参数保持相同。
当前 tensor forward/loss 没有 RNG 原语。工作预算继续消耗，`resource_state()` 的预算观测会随调用变化，
因此含资源观测的模型不保证两次前向仅参数不同；这类状态需调用方明确考虑。
checkpoint 只保存最终提交的原权重候选/基础优化器状态/单次步数，不保存梯度句柄或临时扰动。

```sh
mkdir -p weights/custom_sam
./target/debug/wgt run examples/native_training/custom_sam/model.wgt
```

## 强化学习与 benchmark 边界

[REINFORCE 示例](../examples/native_training/reinforce/model.wgt) 用普通 WGT 函数定义奖励环境，
从随机策略开始采样、累积批次、更新。RL 是原生可执行的策略梯度和 GAE 原语，当前没有内置
PPO/DQN 黑盒训练接口、replay buffer、Gym 适配器、多进程环境或外部模拟器。
现在可使用 [train 语句与可微损失](reinforcement-learning.md)，由 WGT 编写 PPO；
[两步延迟奖励示例](../examples/native_training/ppo/model.wgt) 包含原生 Actor-Critic/GAE 和裁剪更新。

GAE 使用 `delta[t] = reward[t] + gamma*(1-terminated[t])*next_value[t] - value[t]`；
在 terminated 或 truncated 时停止优势递推。截断仍使用最终观测的 next_value，调用者不得
误传自动 reset 后的观测价值。输入是一条顺序轨迹（可串接带显式边界的 episode），不是交错环境批次。
最后一项以调用者提供的 next_value bootstrap，未提供未来轨迹的优势尾项视为 0。

benchmark 默认 CPU 前向，也支持显式同步 GPU 前向及 CPU MSE 训练步。
p50/p95 为排序后 floor((N-1)*q) 项，无插值；同步范围、训练更新与限制见[原生测试](native-testing.md#benchmark)。
输出的训练数据损失、采样奖励只描述相应合成例子，不能视为泛化/真实任务质量证明。

## 保存与续训

原生语法优先使用 `save model net to PATH;` / `load model from PATH using "adam"`，
以及 `save checkpoint net to PATH;` / `load checkpoint from PATH`。
模型加载显式创建新优化器，断点恢复记录的优化器；不执行原脚本 main 或训练。
保存语句可显式加 `overwrite` 原子替换普通可写文件，默认拒绝目标存在。
文本与 JSON、64 MiB 上限、按字节计费和宿主授权边界见[文件与断言语法](file-io.md)。

文件父目录须先存在。保存先写同目录临时文件、sync 后以 hard link 原子发布；目标存在即拒绝，
包括符号链接，不接受隐式覆盖。文件系统须支持硬链接。完整模型、断点和报告应保留在
`weights/<experiment>/`，见[存储约定](storage.md)。
原生训练断点不是传统 WeightLang manifest 制品，由原生 `load checkpoint from PATH` 读取，不能当作推理模型文件读取。
模型初始化代码/脚本、数据源与加载位置、脚本局部变量/RNG、环境状态和训练进度不自动写入断点；
恢复的是模型与优化器，调用者需显式恢复数据/环境/采样进度。相同后续批次在同一后端可逐位续接 Adam 更新；跨 CPU/GPU 以数值容差比较。
`save model` 导出结构和实际权重，保留 view/add 关系，冻结文件权重被物化，不依赖原始参数文件。

当前同一编译单元可声明原生向量模型并在 main 中调用 `模型名("adam")` 构造。支持固定 f32 向量、linear/relu/tanh/sigmoid/identity；复杂图算子及 std block 尚未迁入原生构造器。也可构造 MLP 或读写原生 IR；
将全部 WGT 图、标准库、模型制品和 GPU 训练迁移到原生前端仍是后续工作，不通过 Python 回退掩盖缺口。

编辑器高亮及本地 Tree-sitter 已覆盖 WGT 0.5 函数和控制流；语义诊断仍由 Rust 入口提供。安装及发布边界见 [编辑器支持](editors.md)。

持久化的完整例子是 [checkpoint/model.wgt](../examples/native_training/checkpoint/model.wgt)。在仓库根目录执行：

```sh
mkdir -p weights/native_training
./target/debug/wgt run examples/native_training/checkpoint/model.wgt
```

它只创建全新的 `weights/native_training/checkpoint.json` 与 `model.json`；再次运行会拒绝覆盖。
前者验证 Adam 续训；推理请通过 wgtime service client 执行，不使用已移除的 `wgt-native` 命令。

## 原生 weightc 模型构建器

本地新增 Rust `weightc` 二进制，与已有 Python 分发的同名命令不同；使用明确路径避免 PATH
歧义。旧的 Python 构建入口和 `wgt build` 源码打包入口已移除。

```sh
cargo build -p weightlang-wgt --bin weightc
./target/debug/weightc check examples/native_training/build/model.wgt
./target/debug/weightc build examples/native_training/build/model.wgt -o weights/native_build
```

`weightc check SOURCE` 只调用原生脚本解析器，限制与 `wgt run --check` 相同。
`weightc build SOURCE [-o DIRECTORY] [--max-operations N]` 执行 `wgt 0.5` 的零参数
`main()`，必须返回模型句柄。默认输出为当前目录的 `weights/<源码文件名去扩展名>/`，
默认操作预算 10000000，N 必须正整数；数据等相对路径仍相对原始脚本目录。
完整示例 [build/model.wgt](../examples/native_training/build/model.wgt) 从初始化开始原生训练回归模型，
返回训练后的模型；整个构建与重载不需要 Python/PyTorch、虚拟环境或外部进程。

输出目录包括：`model.json`（结构和实际权重）、`checkpoint.json`（WeightLang 模型与优化器状态，不是 PyTorch training-state 互通格式）、
`source.wgt`（入口快照）、`manifest.json`（格式 weightlang.native-build.v1、源码/模型 SHA256，
编译器版本、更新步数、预算和指标）。最后发布 manifest 作为完成标记；消费者应拒绝没有
完成标记的构建目录。历史 `wgt-native` 本地诊断入口已移除；模型推理由 wgtime service 执行。
断点可通过脚本 `load checkpoint from PATH` 恢复，再训练并返回模型到新的输出目录。

已有输出目录或符号链接一律拒绝，不提供覆盖选项。构建预留目标目录，同盘暂存产物；
执行失败时删除空预留目录，发布中失败保留部分输出供诊断，不宣称整个目录原子发布。
脚本显式 `save model` / `save checkpoint` 语句的外部写入不在构建回滚范围内。

零更新标记 initialized，有优化器更新历史标记 updated；updated 不意味着通过任务质量验收。
恢复断点的步数包含历史更新。该 checkpoint 是 WeightLang 原生格式，不承诺 PyTorch optimizer 或 training-state 互通；safetensors 只覆盖独立权重链路。入口快照与哈希并不包含数据、外部模型依赖或脚本 RNG/数据进度，不保证仅凭制品完整复现训练；应另行保留数据和依赖。原生支持范围仍限于本页列出的算子，
构建入口必须有 main 并显式返回模型句柄；模型声明可与 main 共存。不自动执行训练、不回退 Python。

## 同一脚本的解释执行与 Rust 编译执行

WGT 脚本负责初始化、训练和显式保存权重；weightc 负责编译，不在编译阶段执行这些操作。
两种模式共用解析器、动态值语义、内置函数和原生训练 runtime。

```sh
cargo build -p weightlang-wgt --bin wgt --bin weightc
# 解释执行，默认模式：
./target/debug/wgt run examples/native_training/reinforce/model.wgt
# 先编译后执行同一脚本：
./target/debug/wgt run examples/native_training/reinforce/model.wgt --compiled
# 只编译，不执行；输出文件不能已经存在：
./target/debug/weightc compile examples/native_training/reinforce/model.wgt -o .cache/weightlang/reinforce
# 后续独立运行，无需源码、Python、Cargo 或 rustc：
./.cache/weightlang/reinforce --max-operations 10000000
```

完整命令契约：

- `wgt run SOURCE [--check | --compiled] [--max-operations N]`：默认解释执行；
  --check 与 --compiled 互斥，默认 N=10000000。--compiled 调用 Cargo 后运行临时可执行程序，
  结束后清理该程序；编译失败直接报错，不回退解释器。
- `weightc compile SOURCE -o EXECUTABLE`：SOURCE 必须是 `wgt 0.5`。
  `-o/--output` 必填，自动创建父目录，拒绝覆盖文件/目录/符号链接；生成主机平台可执行文件。
  没有 main 的模型声明、交叉编译不在此入口支持范围内；GPU/WGSL 需要使用 gpu feature 构建的编译器。
- 生成程序接受 `--root DIRECTORY` 与 `--max-operations N`。默认 root 是编译时原始脚本的
  绝对目录；移动程序/数据后应显式设置 root。默认预算 10000000，零预算在第一次操作时失败。
  数据和模型依赖不嵌入二进制，文件读写仍遵循脚本的路径与拒绝覆盖规则。

编译把函数、表达式、分支和循环直接生成为 Rust 代码，交给 Cargo/rustc 的 release 后端；
生成程序不解析 WGT、不遍历 AST。动态值检查、函数分派、变量环境及操作预算仍保留。
训练/保存内置函数与解释器调用同一份 Rust 实现，不复制自动微分、优化器或算子。
两种模式均要求 main 返回可序列化 JSON 值；保存模型应显式写 `save model` 或 `save checkpoint` 语句。
短路求值、函数局部作用域、块内重绑定、递归限制、预算计数和行号诊断保持一致。
编译不增加静态类型推断：未执行分支中的未知名称和错误调用仍可在执行时失败。

本地编译需要 Cargo、rustc、系统链接器、匹配版本的 `wgtime-core` 源码及已缓存的 Rust 依赖。
默认使用构建 weightc 时仓库内的 core 路径；搬移工具链后可设置
`WGT_CONTROL_SOURCE=/path/to/crates/wgtime-control`。Cargo 使用 --offline，不隐式联网下载。
构建中间文件在当前工作目录 `.cache/weightlang/aot/` 的磁盘缓存；临时 Rust 项目结束即清理，
Cargo target 保留加速后续编译。不同编译调用通过锁保护共享产物，最终文件禁止覆盖。
这不是已实现的可独立分发编译 SDK，编译时仍需上述源码与工具链。

首次编译成本与运行成本分开；编译主要省去 AST 调度，并不保证训练吞吐量普遍提高。
CPU 张量算子原本已经是 Rust，训练加速幅度取决于脚本调度占比。

完整权重生成例子 [compiled/model.wgt](../examples/native_training/compiled/model.wgt)：

```sh
mkdir -p weights/native_script
./target/debug/weightc compile examples/native_training/compiled/model.wgt -o .cache/weightlang/train-native
./.cache/weightlang/train-native
```

该脚本显式训练并保存原生 model.json 与 checkpoint.json；重复执行拒绝覆盖。此 legacy-build 制品格式不是 PyTorch optimizer/training-state 互通格式；safetensors 支持属于独立权重导入/导出链路。
也可通过 `wgt run ...` 解释执行或 `wgt run ... --compiled` 编译执行。
先前实验性模型构建入口保留为 `weightc legacy-build`（`build` 为兼容别名）；它仍消费返回模型
并自动导出，不是新的编译工作流。原 Python 同名 CLI 和 wgt build 打包入口已移除。

## 单文件依赖注释与直接执行

单文件可在版本头之前使用一个 TOML 注释块，格式与项目 `[dependencies]` 相同：

```wgt
# /// script
# [dependencies]
# "github.com/WeightLang/WeightLang/packages/gates" = "v0.2.0"
# ///
wgt 0.5
fn main() {
    # 这里只定位已发布包；不会把旧模型制品误当原生训练 IR 加载。
    return dependency("github.com/WeightLang/WeightLang/packages/gates");
}
```

```sh
wgt model.wgt
wgt model.wgt --compiled
wgt --offline --locked model.wgt
wgt model.wgt --check
weightc --offline --locked compile model.wgt -o demo
```

`wgt FILE.wgt` 等价于 `wgt run FILE.wgt`，支持 --compiled、--check、--max-operations，
以及全局 -C/--directory、--offline、--locked。只有模型声明时可检查；运行必须显式定义 main。
完整本地无下载例子：[inline/model.wgt](../examples/native_training/inline/model.wgt)。

注释块必须在版本头之前，仅允许一个 `# /// script` / `# ///` 配对；每行必须以 # 开始，
移除 # 和一个可选空格后按 TOML 解析。当前唯一字段为 dependencies，类型是模块路径到精确
版本字符串的表；省略默认空表。模块格式 github.com/owner/repo[/subdir]，版本为 vMAJOR.MINOR.PATCH。
未知字段、重复块/键、未闭合块、非法路径与非精确版本报错。版本头前允许空行和普通注释，
诊断仍使用原文件行号。此规则只改原生脚本，不改变普通模型的版本规则。

依赖来源有且只有一个：

- 有注释块：独立脚本，包括显式空表，不继承项目依赖；锁位于
  `<WEIGHTLANG_CACHE>/scripts/<canonical-script-path SHA256>/wgt.sum`（未设环境变量时使用默认包缓存）。
  不在脚本旁生成项目配置，移动脚本会改变锁位置。
- 无注释块且属于项目：读取最近的 wgt.toml（兼容既有 wgt.mod），复用项目 wgt.sum，
  不改写依赖声明，不创建 Python 环境。
- 二者都没有：按无依赖脚本执行。

执行/编译前复用已有包解析器，解析精确版本、下载缺失发布制品、校验 commit 与 SHA256。
--offline 禁止下载，缺少锁或缓存报错；--locked 要求非空依赖已有锁并禁止更新。
--check 只验证注释和脚本语法，不下载、不初始化、不执行，因此不证明依赖实际可用。
同一脚本/项目的依赖准备有文件锁。复用现有发布包机制及其能力限制，不引入 pip/npm/Cargo
依赖声明、版本区间或新的传递依赖求解器。

新增 `dependency(module: string) -> string`：参数为已声明模块的完整路径，无默认参数，
返回已校验缓存目录的绝对路径；未声明或未解析时报错。可拼接包内路径交给已有文件 API。
它不导入 WGT 函数，不扩展原生 `load model` 接受的格式：传统 manifest/safetensors 制品不能自动作为
原生 IR JSON 训练。原生训练支持范围仍以 model/restore 等接口契约为准。

编译模式会把已解析目录映射写入程序；数据/包文件不嵌入可执行文件，移动缓存后需重新编译。
独立可执行程序不自动联网解析或重新校验包，--root 仅影响脚本相对路径，不改变依赖缓存路径。
`wgt-model` Rust facade 与 `wgtime` 服务不负责 WGT 包解析；包依赖解析属于 `wgt` CLI 脚本入口。Rust 模型定义入口的纯计算模块导入仍待行为验收；CLI 包依赖解析不等同于该入口的模块导入。

## 计算能力调用策略

原生训练可以将 `compute_status()` 的数值字段作为显式策略输入，再由 WGT 控制流调用
`compute_call(name,input)`。接口完整契约见[计算预算](compute-budget.md)，
监督路由示例见 [compute_policy/model.wgt](../examples/native_training/compute_policy/model.wgt)。
程序调用不可微；训练策略参数不会自动对排序、查找或求和程序求导。

宿主资源示例：[host_resources/model.wgt](../examples/native_training/host_resources/model.wgt)。
`compute_host_status()` 可为策略提供 I/O 剩余额度；启用 `WGT_HOST_POLICY` 时既有模型加载、
保存与数据流接口会被拒绝，当前不能把旧训练数据加载脚本直接当成受限模式运行。
详见[宿主策略](compute-budget.md#宿主策略与只读-io)。

宿主指定 cpu_seconds/address_space_bytes 时，训练与原生计算在独立 Linux worker 中受内核上限约束，
超过 CPU 上限或地址空间分配失败可能终止该 worker。file_write.v1 支持显式写入授权文本输出，
不是模型/优化器断点保存接口。失败写入可能留下部分内容，详见[计算资源](compute-budget.md)。

## weightfs 自动挂载

当前 Linux CLI 和解释/AOT 脚本在构造、加载或恢复模型后，默认尝试将模型挂载到
`$XDG_RUNTIME_DIR/weightlang/weightfs/<pid>-<session>/`，stderr 报告实际路径。
可像文件一样读取和修改参数 JSON；单文件提交与训练/预测互斥，旧版本编辑返回冲突。
手工编辑保留优化器历史，不自动保存。`WGT_WEIGHTFS=on|auto|off` 控制挂载要求（默认 auto）；
纯检查与编译不挂载。完整文件协议、环境要求、错误及预算见 [weightfs v1](weightfs.md)。

子预算与成本感知强化学习已提供可执行示例：
[custom_budget/model.wgt](../examples/native_training/custom_budget/model.wgt)、
[cost_aware_policy/model.wgt](../examples/native_training/cost_aware_policy/model.wgt)。
compute_register 注册原生 WGT 函数，compute_scope 统一结算嵌套工作与 I/O 消费；
训练示例使用实际 work 成本形成策略梯度奖励，而非把程序输出当作可微算子。
完整签名及边界见[资源接口](compute-budget.md#子预算与自定义能力)。

## 设备推理队列

`device_submit` 捕获当前原生模型的权重快照，异步执行设备推理；后续 fit/reinforce 不会修改已排队快照。
WGT 脚本默认 CPU；顺序模型可显式选择 GPU，见 [WGT GPU 与 WGSL](wgsl.md)。Rust GPU 训练 API 见下文。
[device_queue/model.wgt](../examples/native_training/device_queue/model.wgt) 演示两次提交、等待与 CPU 结果对照；
授权、构建特性、设备内存和计算额度见[设备资源](accelerator-resources.md)。

## 标量数值辅助函数

脚本可直接使用 [原生数学标准库](standard-library.md) 中的倒平方根、融合乘加、
稳定指数/对数、欧氏长度和稳定激活。仅处理 f64 标量，无参数默认值、张量广播或梯度。

向量监控可直接使用 `dot`、`norm`、`mean`、`softmax`、`log_softmax`、
`logsumexp`、`l2_normalize`、`mse`、`mae`、`cross_entropy`。
这些函数接受非空一维 f64 列表，返回标量或同形列表；不连接训练梯度。
完整签名、必填参数及预算见[向量算法](standard-library.md#向量算法)。
WGT 0.5 也支持 `values.mean`，与 `mean(values)` 等价；列表只接受非空一维有限数值。
在 forward/loss 内，`square(prediction-target).mean` 则使用可微张量平均，返回 f32[1]。
解释与 AOT 共用此语义；完整优先级与限制见[后缀平均](reference.zh-CN.md#后缀平均-expressionmean)，
可运行训练例子见 [postfix_mean/model.wgt](../examples/native_training/postfix_mean/model.wgt)。

数据分析与监控另可使用 `min/max`、`variance/stddev`、`median/quantile`、
`sort/argsort`、`argmin/argmax`、`cumsum`、`clip` 和 `cosine_similarity`。
所有参数必填，仅处理有限 f64 一维列表；无训练梯度。
完整定义及排序计费规则见[统计、排序与变换](standard-library.md#统计排序与变换)。

监控和预处理新增 `bce_logits(logits,targets)`、`huber(x,y,delta)`、`log_cosh(x,y)`、
`diff(x)`、`ema(x,alpha)`、`moving_mean(x,window)`；所有参数必填。
另有标量 `relu`、`leaky_relu`、`elu`、`hard_sigmoid`、`hard_swish`。
损失函数不自动求梯度，不可把函数名直接当作新增 `fit` 目标。
完整公式、形状、参数范围与预算见[标准库](standard-library.md#分段激活稳健损失与信号处理)。

数据准备可调用 `zeros`、`ones`、`full`、`linspace`、`one_hot`、`concat`、
`vector_slice`、`gather`、`compress`、`reverse`。输入输出为数值列表，无张量梯度。
允许空筛选结果；传给要求非空的统计或训练接口前须自行处理。
完整签名、参数、长度上限及预算见[数据构造与索引](standard-library.md#数据构造与索引)。

## 采样与独热目标

普通函数可用 `sample p { then { a } else { b } }` 只执行抽中的表达式，seed 控制随机流。
`onehot[4] action` 将动作索引转成四类数值列表；复杂索引用括号。等价原生函数为
`one_hot(index,classes)`，类别数 1..1000000，索引为 [0,classes) 整数，分配前扣除 classes 操作预算。
两者不使离散动作可微；onehot 目标可用于 MSE/自定义损失，cross_entropy/reinforce 仍接收类别索引。
详见[语法与边界](differentiable-programs.md)，可运行例子 [sample_branch/model.wgt](../examples/native_training/sample_branch/model.wgt)。

## 实验性 Rust GPU 训练 API

**本地实验性实现；WGT 已接入 training_device，尚未发布**。`training::Trainer::new` 与 `restore` 默认 CPU，
CPU 的运算顺序、SGD/Adam 和 `weightlang.native-training.v1` 断点格式保持不变。
开启 `wgtime-core` 的 `gpu` feature 后，可显式选择 `TrainingBackend::Gpu`；WGT 使用普通内置函数选择，
不依赖设备池。支持顺序 IR 的 linear/relu/tanh/sigmoid/identity、f32、MSE、交叉熵与 REINFORCE，
优化器支持无动量 SGD 和 Adam。不支持任意 forward 程序或多 GPU，也不自动回退 CPU。

```rust,no_run
use std::path::Path;
use wgtime_core::training::{
    mlp, GpuTraining, Objective, Optimizer, Trainer, TrainingBackend,
};
# fn main() -> anyhow::Result<()> {
let budget = 64 * 1024 * 1024;
let mut trainer = Trainer::new(mlp(&[2, 4, 1], "tanh", 42)?, Path::new("."), budget, Optimizer::Sgd)?;
// 使用 execution::devices() 当前枚举的精确硬件 ID；编号不代表永久设备身份。
let gpu = GpuTraining::new("vulkan:0", 32, budget)?;
trainer.set_backend(TrainingBackend::Gpu(gpu))?;
let inputs = [vec![0.2, -0.1], vec![0.3, 0.4]];
let targets = [vec![0.5], vec![-0.2]];
let (loss, gradients) = trainer.gradients(&inputs, Objective::Mse(&targets))?;
let before_update_loss = trainer.step(&inputs, Objective::Mse(&targets), 0.01)?;
// 父目录须存在，目标须为新文件；回读后的权重可导出为 CPU Model 的原生 IR。
trainer.export(Path::new("weights/my-experiment/model.json"))?;
# Ok(())
# }
```

- `GpuTraining::new(device: &str, max_batch: usize, budget_bytes: usize)`：三个参数必填，无隐式默认。
  device 必须是当前硬件枚举的精确 ID，不接受 cpu/auto/gpu；不选择软件 Vulkan。
  max_batch 在 `1..=u32::MAX`，budget_bytes 正数；模型尺寸还须满足设备 buffer、u32 索引和 dispatch 上限。
  构造仅准备设备及内核，不训练。`device_info()` 返回实际 ID、名称与后端信息。
- `Trainer::set_backend(TrainingBackend) -> Result<()>`：GPU 支持 SGD/Adam；切换保留权重、步数、revision、
  优化器历史。`backend()` 返回当前后端。设备选择不写入断点，恢复后显式重选 GPU。
- `gradients(inputs, Objective::Mse(targets))`：输入 `[B,I]`、目标 `[B,O]`，B 非零且不超过上限；
  返回 batch/output 双重平均损失及按参数声明顺序排列的梯度。view/add 梯度在 GPU 上逆序汇总到 owner，
  冻结链截断；重复 owner 只更新一次。此调用不改变权重、步数或 revision。
- `gradients(inputs, Objective::Policy { actions, advantages })`：输入 `[B,I]`，模型输出未经 softmax 的
  logits `[B,O]`；actions 是 B 个 `usize` 类别索引，范围 `0..O`；advantages 是 B 个有限 f32 常量。
  交叉熵传全 1；REINFORCE 传调用者计算的奖励减基线（或其他优势估计），可为负或零，不对奖励/基线求导。
  每样本先减最大 logit，再计算 log-sum-exp；返回 `mean(-advantage * log pi(action))`，
  logit 梯度为 `advantage * (softmax - onehot(action)) / B`。不隐式采样、不内置环境或 baseline 拟合。
- `step(inputs, objective, rate)`：rate 正有限，返回更新前损失。前向 tape、损失、反传和候选 SGD/Adam
  更新在 GPU 执行；全部阶段、回读与有限性检查成功后才一次性提交 CPU 权重、矩状态、步数和 revision。
  Adam 固定 beta1=0.9、beta2=0.999、eps=1e-8，与 CPU 相同：
  `m=0.9*m+0.1*g; v=0.999*v+(0.001*g)*g`，方向为 `m_hat/(sqrt(v_hat)+eps)`。
  偏差校正使用 CPU 同一 `powf(step as f32)` 结果；冻结参数保留原矩，共享 owner 只更新一次。
  溢出检测在 WGSL 算术前执行，回读再检查有限性；不依赖 GPU 对 NaN/Inf 的传播约定。
  设备错误、非法输入、超预算或非有限数值均不提交任何候选权重或矩状态。
- 训练 GPU 错误保留为 `TrainingError::{Unavailable,Unsupported,InvalidInput,OutOfBudget,NonFinite,Device}`，
  可从既有 anyhow 错误链 downcast；原有 CPU、学习率和步数检查保留原错误契约。

CPU Session 与 Trainer 是已提交权重及优化器矩的权威副本：每次 GPU 事务上传当前参数/矩，
设备内保留整个批次的 tape 和候选更新，
前向、反传和优化器共用一次提交，梯度不在两阶段之间往返 CPU；事务结束回读，等大小设备 arena 跨调用复用。没有跨步常驻权重或吞吐量提升承诺。`predict` 仍用 CPU，
`export`/`save` 使用已提交 CPU 副本，设备丢失后仍可保存；后续 GPU 调用报错，可显式切回 CPU。
外部 Session/weightfs 参数写入会被下一次训练读取，保留已有矩和步数；weightfs 仍通过同一模型锁串行访问。
`save checkpoint` 保存完整 `first`/`second`/`steps`，格式仍是 `weightlang.native-training.v1`；CPU 断点可恢复后显式
选择 GPU 续训，GPU 断点可直接由默认 CPU 恢复，跨后端允许 f32 误差，不承诺逐位相同。
冻结 file/packed 参数可物化用于计算，可训练 file 仍拒绝。

GPU budget 是每次调用的保守数值 payload 上界：Session managed bytes、两个既有优化器数组、
完整梯度（包括冻结参数），
以及 host/device arena、回读、返回梯度/候选、物化暂存和操作 uniform 的预留。
不包括调用者 inputs/targets、对象元数据、allocator 多余容量、驱动与命令分配，不能当进程内存硬隔离。
默认 CPU 训练的预算范围不变。

数值测试对 CPU 梯度使用 `atol=2e-5, rtol=2e-4`，20 步权重对照使用 `3e-5, 3e-4`；
容差涵盖 f32 归约顺序及设备激活函数差异，不保证逐位相同或适用于任意病态模型。
GPU 损失中心有限差分取 h=1e-3，使用 `atol=rtol=2e-3`，平衡 f32 相减舍入与二阶截断误差，
ReLU 测试避开零点。真实硬件测试默认 ignored；显式运行，缺硬件时打印 SKIP，验收设置
`WGT_REQUIRE_GPU=1` 则缺硬件失败：

```sh
WGT_REQUIRE_GPU=1 cargo test -p wgtime-core --features gpu --test gpu_training -- --include-ignored --nocapture
WGT_REQUIRE_GPU=1 cargo test -p wgtime-core --features gpu --lib training::gpu::tests::device_loss_preserves_last_committed_model -- --ignored --nocapture
```

模型输出分析可使用 `normalize_probabilities`、`entropy`、`kl_divergence`、
`js_divergence`、`cross_entropy_probs` 和 `temperature_softmax`。
完整签名、分布检查、零概率及预算规则见[概率分布分析](standard-library.md#概率分布分析)。
这些接口仅计算脚本数值；不能作为新增 `fit` 损失名称或自动微分表达式使用。

## 原生测试与评估

`wgt test` 原生发现 tests/*.wgt 的无参 test_* 函数，逐测试隔离状态与预算；
支持过滤、行号诊断、JSON 共存、JSONL 评估以及 CPU/GPU 前向和 CPU 训练步计时。
AOT 测试入口明确拒绝；完整参数、版本与限制见[原生测试](native-testing.md)，
可运行项目见[评估示例](../examples/native_evaluation/README.md)。


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
