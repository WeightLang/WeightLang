# 可微 WGT 程序（0.5 本地实现）

一个 WGT 文件可同时定义可训练参数、神经网络计算、数据与训练循环。
优化器优化的是 `fn forward` 计算所依赖的连续初始参数，不是源码文本。
普通 main/辅助函数、数据读取、文件写入、比较与抽样不会自动获得梯度。
当前实现为 Rust CPU f32 一阶反向自动微分；不是全部语言或标准库均可微，也不是发布声明。

## 原生神经网络表达式

以下均只用于 forward，无省略参数、关键字参数或隐式温度。

| 完整签名 | 形状、含义与梯度 |
| --- | --- |
| `gate(p,x)` | x:f32[N]，p:f32[1] 或 f32[N]；结果 f32[N] 为 p*x；dp=x，dx=p |
| `route(p,a,b)` | a/b:f32[N]，p:f32[1] 或 f32[N]；结果 p*a+(1-p)*b；dp=a-b，da=p，db=1-p |
| `softmax(x)` | x:f32[N]，结果 f32[N]；沿唯一向量轴稳定归一化；反向为 y*(g-sum(g*y)) |

N 为固定正整数。p 每项必须有限且位于 [0,1]，越界报错，不静默裁剪。
标量门沿特征轴广播，反向梯度在该轴求和；普通 add/mul 仍要求等长。
sigmoid(router_logits) 可生成二路门；softmax 可生成多路权重，再用 slice、gate 和 add 组合专家。
softmax 的输入是 logits，不要求处于 [0,1]。

route 是两路结果的加权混合，不是随机抽样。推理（predict/sample、Rust run/evaluate）
先求值并检查 p；p 全为 1 时只计算 a，全为 0 时只计算 b，其余情况计算两边。
gate 推理时 p 全为 0 直接返回同形状零向量，跳过 x 表达式；全为 1 返回 x。
精确比较 0/1，不对 0.999 等近似端点进行阈值截断。向量门含混合值时不做按元素的专家裁剪。
短路仅省略计算，模型参数和寄存器仍分配内存，所有分支仍须通过静态形状/名称检查。

短路只作用于直接写在调用中的表达式，例如 `route(p,linear(x,a),linear(x,b))`；
若提前 `let other=linear(x,b);`，该语句仍执行，不进行跨语句的延迟或代码移动。
支持嵌套路由；未选中表达式不消耗执行指令预算，其数值错误不会触发。
实际选择的表达式及 p 的错误照常报错；失败仍回滚本次 forward 写入。
完整例子：[lazy_routing/model.wgt](../examples/native_training/lazy_routing/model.wgt)。

训练（声明式 train/reinforce）和显式 VJP（Rust/JSON forward）仍计算两边，即使 p=0/1 或来自 frozen 参数。
这是为了保留 dp=a-b、gate 的 dp=x 等完整梯度，不提供伪造的零梯度。
因此推理因短路成功的程序，若未选分支存在溢出，求梯度时仍可能报错。
if positive(x) 仍是硬分支，只沿执行路径反传，比较和循环次数本身无梯度。
有限精度下 sigmoid 可饱和为 0/1，此时其导数可能为零，不保证每个初值都能学好。

## 同文件训练

完整带注释例子：[soft_routing/model.wgt](../examples/native_training/soft_routing/model.wgt)。
它从初始化开始，训练一个按输入符号混合 x/-x 的路由和一个开放门，近似 abs(x)。
专家固定，router/openness 可训练；同一模型句柄的循环迭代共享同一组参数。

```sh
weightc check examples/native_training/soft_routing/model.wgt
wgt run examples/native_training/soft_routing/model.wgt
wgt run examples/native_training/soft_routing/model.wgt --compiled
```

- `模型名(optimizer)`：optimizer 必填 none/sgd/adam 字符串；只初始化，不训练。
- `train model on inputs against targets {rate: lr; loss: mse;}`：forward 模型支持与顺序模型相同的 mse/cross_entropy；
  inputs 是单向量样本列表，MSE targets 同形状，CE targets 为类别整数列表；空批次拒绝。
  rate 为正有限 f32；CE 模型输出 logits，而非先做 softmax 的概率。表达式返回更新前 f32 批次平均损失。
  配置 `loss: loss` 可调用同文件 `fn loss(prediction,target)`；两参数均为单样本输出宽度的 f32 向量，
  返回 f32[1]，按样本求梯度再求批次平均。损失函数使用受限张量表达式，不能读取模型参数。
  完整示例：[prob_mix_loss/model.wgt](../examples/native_training/prob_mix_loss/model.wgt)。
- `reinforce(model,observations,actions,advantages,lr)`：forward 模型支持相同的策略梯度目标；
  advantage 是外部常量，不反传进环境或抽样决策。
- forward 模型采样目标语法为 `decode policy(observation) {method:sample;}`；需等待 parser/runtime 接线完成后，核实短路、RNG 和失败事务细节再公开承诺。
- `model(input)`：保持已有语义，执行并提交 forward 的显式 write；不执行优化器。

训练对每个样本从同一组批次起始参数开始计算；write 只在该次计算中生效且保留梯度链。
不把最终 write 状态当作训练结果提交，最终状态梯度为零；优化器更新初始的非 frozen 参数。
frozen 在 forward 中禁止写入，在训练中禁止优化器更新，但仍可传递输入梯度，底层 VJP 仍能
查询 frozen 的数学梯度。相同参数的重复读取、共享循环使用会汇总梯度，只更新一次。
任一样本失败、预算耗尽或非有限损失/梯度/更新，整批不提交权重或优化器状态。

SGD 无 momentum/weight decay；Adam beta1=0.9、beta2=0.999、eps=1e-8。
每个 forward 会话张量预算 64 MiB，每个训练样本 tape 另有 64 MiB 预算；不含优化器矩、
梯度累加、候选更新、解析或容器开销，不是进程 RSS 限额。
训练 forward 工作量从脚本剩余操作预算逐样本扣除；内存操作按大小额外计费，反向和优化器工作不是逐 FLOP 计数。
当前 forward 模型不支持任意 WGT 损失的自动反传、二阶梯度、任意张量形状或 GPU 训练。
支持有界、无副作用的[纯张量辅助算子及算子包](custom-operators.md)，不支持任意普通函数调用。
benchmark/device_submit/legacy-build 仍只接受原顺序模型，不能据此宣称支持 forward 模型。

## 保存与加载

工作内存的 `memory_alloc/read/write/free` 以及 `resource_state` 现已接入同一个前向执行器。
读写对连续地址和值反传；内存生命周期、额度和硬控制动作不求导。
训练仍一次完成整段轨迹，逐样本清空工作区；分步暂停恢复只用于推理。
内存值版本另占 tape 额度，不能通过 free 退还反向传播需要的历史。
完整签名、梯度、计费、执行状态与例子见[工作内存](working-memory.md)。

- `save model VALUE to PATH;`：forward 模型写出内部 Program IR JSON，包含计算结构、实际参数和
  frozen 标记，不包含优化器；`load model from PATH using OPTIMIZER` 用指定的新优化器创建可执行模型。
  此格式可供当前本地推理实现读取，不是原顺序 IR 的 layers 格式。
  `save checkpoint VALUE to PATH;` 写出训练断点，包括 Program、优化器种类、步数及一二阶矩；
  `load checkpoint from PATH` 识别 forward 与原顺序模型断点并恢复状态。加载不会执行 forward 或 main。
  这些路径将迁接到目标 wgtime service；服务架构尚未实现，不能作为已完成的系统级推理入口。

全部 path 为必填字符串，相对原脚本目录；父目录须已存在，目标存在（含符号链接）即拒绝覆盖。
沿用同目录临时文件与硬链接发布。模型输出保留于 weights/<experiment>/。
不保存脚本 RNG、数据进度、外部资源或 FUSE revision；手动修改参数保留优化器历史。

## 概率、混合、自定义损失与采样语法

`parameter p: prob = 0.7` 必须提供有限标量初值 0<p<1。内部参数为 log(p/(1-p))，
直接引用 p 或 read(p) 返回 sigmoid(logit) 的 f32[1]，优化器更新 logit。
`frozen p: prob = 1` 直接保存概率，可使用 [0,1] 的精确端点，不更新。
prob 无向量形状或默认初值，不能通过 write 改写，也不能直接作为 linear 权重/bias。
原生 parameter、FUSE、checkpoint 与 parameter_gradient 对非冻结 prob 暴露 logit 及其梯度，
对 frozen prob 暴露概率本身。模型重载不重复转换；有限精度 sigmoid 仍可能饱和为 0/1。

`mix p { then { a } else { b } }` 是 route 的结构化表达式，形状、推理短路和梯度相同。
每支为一个张量表达式，可选尾随分号，可嵌套；暂不支持块内 let/return。mix/then 为保留名称。

同文件 `fn loss(prediction,target)` 的两个参数名可自选，第一参数为模型输出形状 f32[N]，第二参数默认同形；
可用 `target NAME: f32[M]` 指定第二参数宽度，返回须为 f32[1]。详见[原生 RL](reinforcement-learning.md)。
仅访问两个输入及局部变量，不访问模型参数，不调用任意普通辅助函数或执行 I/O/sample；
可以调用[纯张量算子](custom-operators.md)，梯度包含展开后的原语计算。
使用 forward 张量算子和 let/if/while/return；新增 square(x)、mean(x)、a-b、-a。
square 返回逐元素平方，mean 沿唯一特征轴平均；减法要求等长，无广播。
例如 `return mean(square(prediction-target));`，用 `train m on x against y {rate: lr; loss: loss;}` 执行。
训练按批次轴取平均，targets 为目标声明宽度（未声明则同输出形状）的向量列表；仅 forward 模型支持，且当前仅 CPU 支持自定义 forward 损失训练。
损失定义不是宿主回调，不能在 main 中直接调用；构造/检查/编译不会训练。
模型与损失 tape 共享每样本 64 MiB 张量预算，损失指令也扣除脚本预算；预算不含会话寄存器和优化器。
整批成功才更新权重和优化器；损失程序保存在 Program/训练断点中，重载后可继续使用。
完整例子：[prob_mix_loss/model.wgt](../examples/native_training/prob_mix_loss/model.wgt)。

普通 main/辅助函数可写 `sample p { then { a } else { b } }`；p 是有限 f64 且在 [0,1]，
先且仅求值 p 一次，只执行选中的表达式，返回其结果。分支可嵌套、可选尾随分号，无块内 let/return。
p=0/1 不消耗随机数，否则从 seed/random 的同一局部流抽取一次，按 random()<p 选择 then。
支持括号概率表达式 `sample (p+q) {...}`；模型类别采样目标使用 `decode policy(observation) {method:sample;}`，待实现后补充完整契约。
这是离散采样，不隐式反传；在 forward/loss 中使用明确报错。策略梯度仍由 reinforce 显式接收动作/优势。
完整例子：[sample_branch/model.wgt](../examples/native_training/sample_branch/model.wgt)。

## 独热编码

普通函数的 `onehot[N] index` 将索引转换为长度 N 的数值列表：`onehot[4] 2` 得到 `[0,0,1,0]`。
N 是必填十进制整数字面量，范围 1..1000000；index 在运行时求值一次，须为 0<=index<N 的整数，
不接受 bool、列表、负数、小数或越界值。onehot 为保留名称，优先级同一元负号；
复杂索引用 `onehot[4] (action+1)`，访问结果用 `(onehot[4] action)[2]`。
底层等价于 `one_hot(index,N)`，该函数也接受运行时类别数，边界相同。
分配前按 N 扣除操作预算；不对离散索引求导，仅普通函数可用，forward/loss 不支持。
结果可显式作为模型输入或 MSE/自定义损失的目标；cross_entropy/reinforce 的类别参数仍为索引。
示例见 [sample_branch/model.wgt](../examples/native_training/sample_branch/model.wgt)，解释与 AOT 使用相同检查。

现有 soft_routing 示例采用 `mix` 和 `parameter openness: prob = 0.5`；后者的内部初值仍是 logit=0，保持原初始化与训练行为。lazy_routing 示例使用 `mix` 保留精确端点短路。按模型 logits 抽取类别的目标形式为 `decode policy(observation) {method:sample;}`；等待 parser/runtime 实现并验证后再迁移示例，不能替换成二选一概率采样。

## 后缀平均

WGT 0.5 的 forward/loss 支持 `x.mean` 和 `square(prediction-target).mean`，
与 `mean(x)`、`mean(square(prediction-target))` 使用相同的张量指令和反向传播。
f32[N] 沿唯一特征轴归约为 f32[1]，N>0；每项梯度为上游梯度除以 N。
损失中的平均针对单样本输出轴，训练接口另对样本轴平均；保存/重载没有新增制品格式。
普通函数中的列表平均返回 f64 标量，不自动建立梯度图。无多轴、隐式展平或广播。
完整[语法契约](reference.zh-CN.md#后缀平均-expressionmean)及
[训练例子](../examples/native_training/postfix_mean/model.wgt)。

原生 RL 扩展：声明式 `train ... on ... against ... { ... }`、`using` 策略、独立 target 声明、常量向量、detach、exp/log/log_softmax/sum/minimum/clip 的完整签名和梯度边界见[原生强化学习](reinforcement-learning.md)。
