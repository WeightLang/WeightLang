# WeightLang 0.5 语言参考

当前源码只接受 **`wgt 0.5`**。这是破坏性语言更新，未发布；
`wgt 0.1/0.2/0.3`、`wgt script 0.1`、`wgt adaptive 0.1` 全部拒绝。
模型声明、训练函数和自更新前向函数属于同一个编译单元，不能通过选择命令改变语法版本。
工具包版本、模型 JSON 的 `version` 和训练断点版本是独立概念，不随语言版本替换。

当前实现为纯 Rust。原 Python 前端的完整图算子、block/import 等尚未全部迁入原生构造器；
`std/` 根目录的旧图源码作为迁移材料保留，不能因为文件头改为 0.5 就宣称可执行；
分类子目录中的原生标准库示例可单独运行，完整目录见[标准库](standard-library.md)。不支持的声明显式报错，不回退 Python。

## 语法与能力边界的机器可读规格

[上下文能力规格](../language/source_capabilities.json) 记录当前源码 edition、解析入口及普通函数、
张量 forward/loss、模型声明和编辑器的边界；[共同正负语料](../language/conformance.json)
由 Rust 与编辑器本地测试共同读取。这是已测试边界的有限规格，不是完整内置函数表。
普通函数与 forward/loss 共用函数 parser，但张量 lowering（转换为张量指令）另行限制表达式、
控制流与形状；顶层模型声明由独立解析入口处理。函数解析成功不代表运行时调用有效。
[primitive_specs.json](../language/primitive_specs.json) 描述制品 graph 原语，供制品验证使用，
不能当作当前源码全部可调用内置函数的清单，也不改变旧制品的读取兼容性。

## 无需 import 的内置入口

[language/prelude.json](../language/prelude.json) 是唯一手写内置目录，集中名称、签名、默认值、
用途、适用上下文与实现路径。Rust 的 [script/prelude.rs](../crates/wgt-runtime/src/script/prelude.rs)
直接读取它驱动普通函数名称保留、参数数量检查及兼容别名；编辑器生成器从同一目录生成快照。
`ordinary_function` 供解释与 AOT 共用；`tensor_forward` / `tensor_loss` 是独立张量转换契约，
不能因名字出现在目录中就用于普通函数。`write` 等标注为语句，`positive` 只用于张量条件。
目录也列出 `true`、`false`、`null` 常量及 `f32`、`prob`、`logits`、`probabilities` 声明类型；这些不是可调用函数。
`std/` 模块不自动导入，仍须按各模块的使用契约显式引入。

## 编译单元与入口

首个非空、非注释行必须恰好是 `wgt 0.5`，版本头只能出现一次。
`#` 到行尾为注释，函数字符串使用双引号和 JSON 转义。顶层模型声明一行一条；
函数使用 `fn name(args) { ... }`，语句以分号结束，控制流块无需分号。
名称使用 ASCII 字母或下划线开头、后接字母/数字/下划线；`__` 前缀供编译器使用。

一个文件可声明一个模型和多个函数。`fn main()` 无参数，执行程序时才调用。
模型名成为构造函数，调用才初始化会话；只解析/编译不访问训练数据、不训练、不保存。
只有模型声明的文件可检查，但执行或编译可执行文件时必须显式定义 main。

```sh
weightc check examples/native_training/supervised.wgt
wgt run examples/native_training/supervised.wgt
wgt run examples/native_training/supervised.wgt --compiled
weightc compile examples/native_training/supervised.wgt -o .cache/weightlang/regression
```

`wgt script` 是 `wgt run` 的命令别名，不是另一套语言。
解释和编译模式使用同一前端；编译模式生成 Rust 控制流和静态模型计划，运行时不解析 WGT。
`weightc check` 检查语法及受支持的模型声明，不证明所有函数调用、路径初始化或循环终止。

## 声明式模型

以下为当前原生构造支持的完整声明集；端口不含 batch 轴。

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
        # fit 输入为 [样本, 特征]，MSE 对样本与输出元素取平均。
        fit(net, [[0], [1]], [[1], [3]], 0.03, "mse");
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

WGT 0.5 普通函数还支持 `let action = sample from net(input);`：`net` 必须是当前函数内的模型句柄变量或参数，
只接受一个输入表达式，按模型输入宽度转换为有限 f32 向量。模型查找在输入求值之前，输入只求值一次；
不支持 `sample from factory()(x)`、限定名或任意模型表达式。括号属于采样语法，`net(input)` 不会先按普通调用执行；
整个采样结果可继续参与索引、后缀或二元运算，必要时用括号分组。
返回零基类别索引，范围为 `[0, output_width)` 的整数数值。模型输出按显式类型抽样：`logits[N]` 使用稳定 softmax，`probabilities[N]` 验证后按实际总质量抽样；旧 `f32[N]` 保持原有 logits 解释及 RNG 数值顺序；
每次成功抽样消耗一次脚本 RNG，受 `seed` 和操作预算控制。它执行无 tape 推理，不提交 forward 中显式 `write`，
不更新权重或优化器；普通 `net(input)` 则提交 forward 写入。forward/loss 中拒绝此离散表达式。
旧 `sample(net,input)` 函数入口已移除，会提示迁移到 `sample from net(input)`；二选一 `sample p { ... }` 语法保留。
例如 `fn choose(net,x) { return sample from net(x); }`，可运行完整训练示例见 [reinforce/model.wgt](../examples/native_training/reinforce.wgt)。

值包括有限数值、bool、字符串、null、列表、JSON 对象和不透明资源句柄。
`let name = expr;` 绑定变量，`name = expr;` 修改已有绑定。
支持函数调用、列表/对象索引、一元 `- !`、算术 `+ - * / %`、比较及短路 `&& ||`。
数值运算产生无穷大或 NaN 时报错，包括数字除以零及对零取余；不会隐式替换除数。
同次采样可用声明语句 `let {action, log_prob} = sample from net(input);` 同时绑定动作与当时的自然对数概率；
支持 `let {action: a, log_prob: lp} = sample from net(input);`，也可只选择一个字段。
字段只能为 `action` / `log_prob`，字段和绑定变量均不得重复，空绑定拒绝；这不是通用对象解构表达式。
模型与输入的求值顺序、一次 forward、一次 RNG 和不提交写入的规则与标量形式相同；取得概率不会再次调用模型。
logits 的 log probability 直接用 f64 减最大值和 log-sum-exp 计算，避免先 softmax 后取 log 的下溢；
显式 probabilities 按实际抽样分布返回 `log(p_action) - log(sum(p))`，零概率类别不会被选中。
旧 f32 的动作沿用原有采样算术；其 log probability 用同次输出的 f64 稳定公式，舍入可能与旧 f32 `log_softmax` 有差异。
两个字段全部计算成功后才绑定变量。forward/loss 不支持此声明；解释和 AOT 共用采样核心。
完整例子见 [sample_record/model.wgt](../examples/native_training/sample_record/model.wgt)。
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
`net(x)` 与兼容接口 `predict(net,x)` 共用执行路径，不隐式训练、更新优化器或保存制品；
无显式写入的前向不改变参数、revision 或训练步数。forward 中显式 `write` 仍按原有事务契约提交，
失败则回滚；运行工作量仍计入执行预算。错误包含调用名、所在语句行号及形状不匹配时的预期/实际宽度。

普通函数支持原生 `assert CONDITION [, MESSAGE];` 语句，条件严格为 bool 且只求值一次；仅失败才求字符串消息，默认 `assertion failed`。括号条件可写 `assert(condition);`。旧两实参 `assert(condition,message)` 保持急求值与返回 null 的兼容契约。解释/AOT 行为一致，错误含语句行号，详见[原生断言](assertions.md)。

普通函数的原生文件语法为 `save KIND VALUE to PATH [overwrite];` 和读取表达式
`load KIND from PATH`，KIND 为 `text/json/model/checkpoint`；`load model` 必填 `using OPTIMIZER`，
checkpoint 使用保存的优化器。值、路径各求值一次；相对路径按脚本目录解析，父目录须存在。
默认拒绝覆盖，显式 overwrite 只原子替换普通可写文件；每文件最多 64 MiB，按编码字节消耗 operation budget。
宿主策略模式拒绝直接路径 IO，须用授权 capability。模型与断点不能混用，load 不执行 main 或训练。
`reload model TARGET from PATH using OPTIMIZER;` / `reload checkpoint TARGET from PATH;` 在同句柄中原子替换同结构模型状态，所有别名可见；model重新初始化优化器，checkpoint恢复保存状态与步数，运行时revision递增并使旧梯度失效。结构不匹配或候选失败保留旧状态。
`assert CONDITION [, MESSAGE];` 要求严格 bool，只失败才求字符串消息，默认 `assertion failed`；
旧两实参 `assert(condition,message)` 保持急求值兼容。解释/AOT 相同，错误含语句行号。
完整签名、格式、安全发布和 JSON 限制见[文件与断言语法](file-io.md)。

控制流为 `if condition { ... } else { ... }`、`while condition { ... }`、
`for name in list { ... }`、`break;`、`continue;`、`return expr;`。
当前函数内共享变量环境，if/for 不创建词法局部作用域；重复 let 覆盖绑定。
无任意 Python、shell 或宿主回调。源码函数嵌套与求值均有预算限制。

训练、流式 JSONL、强化学习、保存与 benchmark 的完整内置签名、默认值及限制见
[原生训练](native-training.md#内置接口)。常用 `model(input)`、
`fit(model,inputs,targets,rate,loss)` 没有隐式训练或保存。
`save_model` 保存模型，`save_checkpoint` 保存优化器断点，两者不能混用。

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
    assert(x[start:3] == [20,30], "half-open bounds");
    assert(x[4:] == [], "empty list slice");
    return x[:2];
}
```

包含列表、forward/loss 和显式训练的完整示例：
[`slicing/model.wgt`](../examples/native_training/slicing/model.wgt)。

## 自更新前向函数

同一个模型可用 `fn forward(x)` 定义有状态前向，替代 layer/init 声明。
函数参数必须与 input 名称一致，input 使用 `f32[N]`；output 可使用 `f32[N]`、`logits[N]` 或 `probabilities[N]`。
参数显式声明为 `parameter w: f32[O,I] = [...]` 或 `frozen w: f32[O,I] = [...]`；
初值是行优先扁平有限 f32 数组，元素数必须与正整数形状匹配。
自更新模型还可声明标量 `parameter p: prob = 0.7` 或 `frozen p: prob = 1`。
可训练 prob 的初值须严格在 (0,1)，内部保存 logit 并通过 sigmoid 读取；冻结 prob 可使用闭区间 [0,1] 的精确端点。prob 不支持 forward 的 `write`，也不能直接作为 linear 权重或偏置。
目前没有布尔参数类型；普通 `f32` 参数不会自动限制在 `[0,1]` 或 `{0,1}`。
`frozen` 禁止本次调用内 write 及优化器更新；不意味着该参数的数学梯度为零。

`fn forward` 复用普通函数的 let、赋值、return、if、while、break、continue 语法，
操作数为定长 f32 张量。暂不支持 forward 中的 for、辅助函数调用、数据 IO 或优化器调用。
当前前向表达式签名如下，除 linear 的 b 外没有默认参数：

| 表达式/语句 | 语义与约束 |
| --- | --- |
| `read(w)` | 读取参数当前值，结果为扁平向量；直接引用参数也会读取 |
| `write(w, value);` | 全量替换非冻结参数，元素数须相同 |
| `linear(x,w[,b])` | w=[O,I]、x=[I]、结果=[O]，可选 b=[O] |
| `add(a,b)` / `mul(a,b)` / `a+b` / `a*b` | 等长逐元素运算，无广播 |
| `relu(x)` / `tanh(x)` / `sigmoid(x)` | 等长逐元素运算 |
| `softmax(x)` | f32[N] 沿唯一向量轴归一化，返回 f32[N]；稳定指数实现，支持一阶梯度 |
| `gate(p,x)` | 返回 p*x；x 为 f32[N]，p 为 f32[1] 或 f32[N]，每项须在 [0,1]；标量门广播；推理时 p 全为 0 跳过 x 表达式 |
| `route(p,a,b)` | 返回 p*a+(1-p)*b；a/b 同为 f32[N]，p 为 f32[1] 或 f32[N] 且在 [0,1]；推理时 p 全为 1 只求值 a，全为 0 只求值 b，否则计算两边 |
| `slice(x,start,length)` | start 非负整数、length 正整数，范围不得越界 |
| `positive(x)` | 仅作为条件；x 必须 f32[1]，严格大于零选真分支 |
| `let y = x;` | 复制张量值并保留梯度，不创建参数别名 |

局部张量的长度由表达式推导，重复赋值不能改变长度；输入不可覆盖。
同一模型内 forward 与 layer 不混用。模型通过 `Name(optimizer)` 构造，optimizer 必填
`"none"`、`"sgd"` 或 `"adam"`；后两者允许 `fit` / `reinforce` 优化整个已执行 forward 的非冻结初始参数。
普通 `handle(input)` 执行并提交成功写入；失败回滚，步数计入外层操作预算。
无需 `register`、标签、`jump` 或 `branch` 源码指令，这些仅存在于内部 IR。

完整可执行示例：[self_modifying/model.wgt](../examples/self_modifying/model.wgt)。
可微路由训练示例：[soft_routing.wgt](../examples/native_training/soft_routing.wgt)，
完整训练、状态与保存约束见[可微程序](differentiable-programs.md)。上述三种算子的张量梯度仅用于 forward，
全部参数必填，无温度等隐式选项。`p=1` 表示完整通过或完全采用 a 的输出。
推理短路作用于直接写在 gate/route 内的表达式（支持嵌套），不延迟此前已执行的 let；
比较使用精确 0/1，无 0.999 等近似阈值。混合向量门不会按元素拆分专家执行。
forward 中也可写 `mix p { then { a } else { b } }`，等价于 `route(p,a,b)`；
分支内各为一个张量表达式，可选尾随分号。`mix` 和 `then` 是保留名称。
fit/reinforce 和显式 VJP 记录梯度时仍计算两边，包括 frozen 门，以保留完整数学梯度。
短路示例：[lazy_routing.wgt](../examples/native_training/lazy_routing.wgt)。
路由选择梯度为 a-b，门梯度为 x；标量门梯度在特征轴求和。
普通 if 保持硬分支；源文件文本、宿主函数、I/O 和离散控制决策不因此可微。
事务、原生一阶自动微分、显式状态及预算细节见[自更新运行时](self-modifying-runtime.md)。

## 静态用户训练策略

`train MODEL on (INPUTS,TARGETS) using STRATEGY;` 调用同编译单元的同步普通用户函数
`fn STRATEGY(model,inputs,targets)`，恰好三参，支持前向声明；拒绝内置函数、模型构造器、缺失函数、错误形参数与async函数。
目标静态绑定声明，不受局部同名值影响；三个实参从左到右各求值一次，策略只调用一次，return由语句丢弃。
没有默认超参、函数值或捕获；超参写在策略函数内，using与rate/clip不能混用，既有rate/clip语句保持可用。
策略自行决定differentiate/at/commit及更新次数，语句不隐式优化、额外提交或重试；每个commit原子，整个策略函数不自动回滚。
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
`save_checkpoint/load_checkpoint` 的 forward v2 保存/恢复用户状态及步数，兼容旧 v1读取；保存模型不含优化器。
完整签名、预算计量、约束和续训示例见[原生训练](native-training.md#wgt-自定义更新算法cpu-forward)及
[custom_momentum/model.wgt](../examples/native_training/custom_momentum/model.wgt)。目前不支持此接口的顺序/GPU模型，
不自动执行算法、恢复训练循环或导入含循环的 host 模块。
`differentiate_at(gradient,candidate_parameters)` 在临时参数上使用首次句柄保存的同一 f32 批次重新求一阶梯度，
不写真实模型；返回的新句柄仍以原 parameters/state/steps 为提交基准，loss/gradients来自临时点。
它与原接口共享句柄/内存预算，拒绝过期版本；批次保留与临时绑定副本也计入预算。
[custom_sam/model.wgt](../examples/native_training/custom_sam/model.wgt) 用两次一阶求导与一次提交定义 SAM+SGD/动量；
这不提供高阶自动微分、SAM专用对象或GPU自定义优化。完整同批次/状态语义见[原生训练](native-training.md#用普通-wgt-编写-sam)。

## 实现与验证依据

- 编译单元和静态图计划：`crates/wgt-runtime/src/source.rs`。
- 共用函数语法：`crates/wgt-runtime/src/script/parse.rs`。
- 模型构造与普通函数运行：`crates/wgt-runtime/src/script.rs`。
- forward 降低、事务及梯度：`crates/wgt-runtime/src/adaptive/`。
- 原生代码生成：`crates/wgt-runtime/src/script/codegen.rs`。
- 回归：`tests/unified.rs`、`tests/adaptive*.rs`、`tests/script.rs`（runtime crate），
  `crates/wgt/tests/script_compiler.rs` 验证解释/AOT 的结果和操作数一致。

保留模型和实验在 `weights/`，不因升级版本头自动迁移或覆盖。详见[存储规范](storage.md)。

## 运行中权重的文件系统视图

Linux 训练、推理 CLI 及独立编译程序支持 [weightfs v1](weightfs.md)，在模型初始化后自动挂载
内存权重视图。它是运行时能力，不增加语法或内置函数；检查和编译阶段不挂载。
环境变量 `WGT_WEIGHTFS=auto|on|off` 默认 auto。文件编辑校验形状和有限 f32，遵守冻结/共享
存储约束，在执行之间提交；冲突报错。持久化仍使用显式 save/save_model，详见协议文档。

## 概率与分支语法的完整契约

`parameter p: prob = 0.7`、`frozen p: prob = 1`、`mix p { then { a } else { b } }`、
`sample p { then { a } else { b } }` 及 `fn loss(prediction,target)` 的参数、表示、梯度、
执行边界和可运行例子统一见[可微程序语法](differentiable-programs.md#概率混合自定义损失与采样语法)。
forward/loss 新增 `square(x)`（等长平方）、`mean(x)`（唯一向量轴平均为 f32[1]）、
等长逐元素 `a-b` 和 `-a`；全部无默认参数或广播。自定义损失通过 `fit(...,"loss")` 调用，
其表达式与模型共同求导；普通 sample 块不建立梯度图，forward/loss 中明确拒绝。

`onehot[N] index` 是普通函数中的独热表达式，如 `onehot[4] 2` 得到 `[0,0,1,0]`。
N 为 1..1000000 整数字面量，index 是运行时 [0,N) 整数；优先级同一元负号，复杂索引用括号。
原生等价调用 `one_hot(index,classes)` 支持运行时类别数。无默认值，不对索引求导，forward/loss 不支持。
完整边界、预算与示例见[独热编码](differentiable-programs.md#独热编码)。

## 原生标量数学标准库

WGT 0.5 新增 `rsqrt(x)`、`fma(x,y,z)`、`hypot(x,y)`、`expm1(x)`、
`log1p(x)`、`sigmoid(x)`、`softplus(x)`、`silu(x)`，全部参数必填且为有限 f64。
原生直接调用、解释/AOT 共用实现；不表示同名图算子或张量自动微分已支持。
完整定义、定义域、硬件边界及分类示例见[标准库](standard-library.md)。

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
`net(x)` 仍提交显式 `write`，`sample from net(x)` 仍无梯度 tape 且丢弃 forward 写入。
索引返回普通数值，切片返回普通列表；改变向量的算术/列表操作不自动推断类别类型。

`scores.probabilities` 对 logits 沿唯一类别轴执行减最大值的稳定 softmax，返回有类型的 probabilities；
已是 probabilities 时逐项验证后保持原值，不重复 softmax、不静默归一化。每项必须有限且严格在 [0,1]，
总和使用 f64 累加，与 1 的绝对误差不得超过 **1e-5**；此容差覆盖最大支持宽度的 f32 softmax 舍入。
空向量、非有限值、错误形状和非法概率都明确报错。普通 f32 向量不能使用该后缀，显式数学函数 `softmax(x)` 保留。
后缀与索引从左向右结合，接在完整调用或括号之后，输入只求值一次，不受同名局部变量遮蔽。

forward/loss 的预测参数保留输出类型，后缀转换可微：logits 使用 softmax Jacobian，probabilities 验证为恒等梯度。
`log_softmax(prediction)` 仍直接消费 logits，交叉熵原有输入契约不变，不强制先 softmax 再 log。
显式概率采样按实际总质量缩放一次 RNG 阈值，跳过零质量类别；这不修改存储向量。
CPU 路径已实现，GPU 新类型路径未验证。完整示例见 [categorical/model.wgt](../examples/native_training/categorical/model.wgt)。

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
    assert(scores.mean == 4, "mean");
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
这些普通脚本函数不新增声明式图层支持、张量梯度或 `fit` 训练目标。


数据标准库新增 `zeros(count)`、`ones(count)`、`full(count,value)`、
`linspace(start,end,count)`、`one_hot(index,classes)`、`concat(x,y)`、
`vector_slice(x,start,end)`、`gather(x,indices)`、`compress(x,mask)`、`reverse(x)`。
全部参数必填；输入为一维有限 f64 列表，数据操作允许空列表，掩码严格为 bool。
索引为零起始整数，切片左闭右开；单次输出不超过 100 万项。
完整参数范围、输出形状、预算和运行例子见[数据构造与索引](standard-library.md#数据构造与索引)。
这些是普通脚本列表操作，不是 forward/loss 张量算子或零拷贝视图。


概率分析新增 `normalize_probabilities(weights)`、`entropy(p)`、`kl_divergence(p,q)`、
`js_divergence(p,q)`、`cross_entropy_probs(p,q)`、`temperature_softmax(logits,temperature)`。
所有参数必填，输入非空一维有限 f64 列表；双分布必须等长，温度有限且>0。
分布逐项在 [0,1]，总和允许 1e-9 绝对误差，计算前修正该误差；任意权重须显式归一化。
零概率、公式、预算和例子见[概率分布分析](standard-library.md#概率分布分析)。
这些是普通脚本分析函数，不提供张量自动微分或新的 `fit` 目标。

## 原生 RL 训练语法

当前 WGT 0.5 支持 `train MODEL on (INPUTS,TARGETS) rate RATE clip MAX_NORM;`，
等价 `train(model,inputs,targets,rate,max_norm)`，函数形式返回更新前批次平均损失。
所有参数必填，仅含 fn loss 且使用 sgd/adam 的 forward 模型支持；rate/max_norm 为有限正 f32。
整批求导取平均、全局 L2 梯度裁剪，然后原子提交一次更新；失败不改变参数或优化器。
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
