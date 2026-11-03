# WeightLang 0.5 语言参考

当前源码只接受 **`wgt 0.5`**。这是破坏性语言更新，未发布；
`wgt 0.1/0.2/0.3`、`wgt script 0.1`、`wgt adaptive 0.1` 全部拒绝。
模型声明、训练函数和自更新前向函数属于同一个编译单元，不能通过选择命令改变语法版本。
工具包版本、模型 JSON 的 `version` 和训练断点版本是独立概念，不随语言版本替换。

当前实现为纯 Rust。原 Python 前端的完整图算子、block/import 等尚未全部迁入原生构造器；
`std/` 根目录的旧图源码作为迁移材料保留，不能因为文件头改为 0.5 就宣称可执行；
分类子目录中的原生标准库示例可单独运行，完整目录见[标准库](standard-library.md)。不支持的声明显式报错，不回退 Python。

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
| `output y: f32[M]` | 一个输出，M 必须等于最后一层宽度 |
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
    return predict(net, [1]);
}
```

## 普通函数、训练与数据

值包括有限数值、bool、字符串、null、列表、JSON 对象和不透明资源句柄。
`let name = expr;` 绑定变量，`name = expr;` 修改已有绑定。
支持函数调用、列表/对象索引、一元 `- !`、算术 `+ - * / %`、比较及短路 `&& ||`。
数值运算产生无穷大或 NaN 时报错，包括数字除以零及对零取余；不会隐式替换除数。
条件必须 bool，无隐式数值或张量真假转换；函数参数按位置传递，参数数目严格匹配。

控制流为 `if condition { ... } else { ... }`、`while condition { ... }`、
`for name in list { ... }`、`break;`、`continue;`、`return expr;`。
当前函数内共享变量环境，if/for 不创建词法局部作用域；重复 let 覆盖绑定。
无任意 Python、shell 或宿主回调。源码函数嵌套与求值均有预算限制。

训练、流式 JSONL、强化学习、保存与 benchmark 的完整内置签名、默认值及限制见
[原生训练](native-training.md#内置接口)。常用 `predict(model,input)`、
`fit(model,inputs,targets,rate,loss)` 没有隐式训练或保存。
`save_model` 保存模型，`save` 保存优化器断点，两者不能混用。

## 自更新前向函数

同一个模型可用 `fn forward(x)` 定义有状态前向，替代 layer/init 声明。
函数参数必须与 input 名称一致，input/output 仍使用相同的 `f32[N]` 端口语法。
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
普通 `predict(handle,input)` 执行并提交成功写入；失败回滚，步数计入外层操作预算。
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
