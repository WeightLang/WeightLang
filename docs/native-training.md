# 原生 WGT 训练脚本（本地实验实现）

`wgt run FILE.wgt` 在 Rust 进程内解析并执行脚本，不启动 Python、不加载 Python 回调，
无需 PyTorch。`wgt-native script FILE.wgt` 是同一执行器的独立入口。
当前只保留原生训练路径；旧 Python 训练入口已删除。
当前不是全部模型算子已经迁移，也不是已发布功能。

```sh
cargo build -p weightlang-wgt
./target/debug/wgt run examples/native_training/supervised.wgt --check
./target/debug/wgt run examples/native_training/supervised.wgt
./target/debug/wgt run examples/native_training/reinforce.wgt
```

`--check` 只检查脚本结构、版本、嵌套深度与重复函数/形参，不读取数据、不构造模型、不运行调用。
它不做完整类型检查或证明未执行分支中的名称/调用有效。模型与实参在执行时检查。
`wgt run --max-operations N` 默认 10000000，限制解释器表达式、语句、循环和调用的执行次数；
这不是超时、内存限额或内置训练算子的 FLOP 限制。内置文件操作可能阻塞。
`wgt-native script` 使用相同默认预算，只接受文件及可选 `--check`。

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
    assert(total == 20, "unexpected control flow");
    return total;
}
```

- `let name = expression;` 在当前函数环境创建或重新绑定变量；`name = expression;` 要求变量已存在。
  块不另建作用域，循环变量执行后保留；函数有独立环境，没有全局变量、闭包或捕获。
- `fn name(arg, ...) { ... }` 参数按位置传递；`return expr;` 返回值，`return;`/自然结束返回 null。
  支持前向函数引用与有界递归；无默认参数、关键字参数或可变参数。不能覆盖内置函数名。
- `if bool { ... } else { ... }`、`while bool { ... }`、`for name in list { ... }`；else 可省略。
  `break;` / `continue;` 只作用于最近的循环；不支持 `else if` 简写，可嵌套 if。
- 值为有限 f64、bool、字符串、null、列表、JSON 对象、模型/数据流句柄。
  JSON 对象来自数据和指标结果，当前无对象字面量。列表按值传递；句柄副本指向同一资源。
  原生训练张量为 f32，转换溢出报错。
- `a[i]` 是非负整数列表索引，`row["input"]` 是对象索引；无切片、负索引、属性访问或下标赋值。
- 从低到高：`||`、`&&`、`== !=`、`< <= > >=`、`+ -`、`* / %`、一元 `- !`、索引。
  `&& ||` 短路并要求 bool；if/while 不隐式把数字或张量转成 bool。
  `+` 支持数字加法或字符串连接。相等比较按值/句柄身份；数值运算不允许非有限结果。
  数字除以零或对零取余产生的非有限结果会报错；不会自动替换除数或将结果截断。
- 最大 100000 token、64 层语句块、128 层表达式树、64 层函数调用。
  错误带脚本路径、执行语句行号；无异常捕获、异步、线程、宿主语言执行或 FFI。

## 内置接口

所有参数均必填，除表内明确写出的固定内部行为外没有隐式默认参数。相对路径相对脚本文件目录。

| 签名 | 契约 |
| --- | --- |
| `len(list)` | 列表长度 |
| `range(start,end)` | 非负整数半开区间；end >= start，最多 1000000 项 |
| `append(list,value)` | 返回追加后的新列表，不原地改动；最多 1000000 项 |
| `column(rows,key)` | 从对象列表逐行提取字段，缺失即报错 |
| `sum(numbers)` | 求和，空列表为 0 |
| `sqrt(x)`, `log(x)`, `exp(x)` | 标量数学；域错误/溢出报错 |
| `assert(condition,message)` | bool 为 false 时终止脚本 |
| `metric(name,value)` | 按名称保存 JSON 指标；同名覆盖，只保留最后值，不自动平均或 detach |
| `seed(integer)` | 设置脚本局部采样 RNG；整数范围 0..1000000000 |
| `random()` | 确定性 SplitMix64 流，返回 [0,1)；不影响模型初始化 RNG |
| `mlp(widths,activation,seed,optimizer)` | 新建 CPU f32 顺序网络；widths 是包括输入输出的 2..64 个整数，每个 1..16384，最多 4000000 参数；隐藏层 activation 为 relu/tanh/sigmoid/identity，输出层始终 linear；optimizer 为 sgd/adam |
| `model(path,optimizer)` | 加载原生顺序 IR 或 forward Program IR v1 JSON，使用新优化器；forward 另可选 none；不是普通 .wgt 模型解析器，也不接受传统 manifest 制品 |
| `predict(model,input)` | 输入单个 f32 向量，返回输出列表；不改变权重/优化器、不训练 |
| `fit(model,inputs,targets,lr,loss)` | 一个批次一次更新，返回更新前损失；lr 正有限；loss 为 mse/cross_entropy |
| `reinforce(model,observations,actions,advantages,lr)` | 一批观测的策略梯度更新，返回更新前的 mean(-advantage*log pi(action)) |
| `sample(model,observation)` | 将模型输出视为 logits，稳定 softmax 后由脚本 RNG 抽取动作索引；forward 模型使用无 tape 的端点短路推理，不提交 write |
| `gae(rewards,values,next_values,terminated,truncated,gamma,lambda)` | 同长度一维列表；后两种标记是 bool，gamma/lambda 在 [0,1]；返回优势列表 |
| `hf_dataset(dataset,config,split)` | Rust 原生 Hugging Face 分页数据流；全部参数必填，首次 batch 联网；HF_TOKEN 可选，详见 [数据集契约](datasets.md#原生数据集加载器wgt-05当前源码) |
| `jsonl(path)` | 打开 UTF-8 JSON Lines 数据流；每行一个 JSON 值，不允许空行 |
| `batch(reader,count)` | count 在 1..65536；返回最多 count 行；尾批保留，到 EOF 返回 []；单行 <=1MiB、单批 <=16MiB |
| `close(reader)` | 关闭流，重复关闭无操作；关闭后 batch 报错 |
| `save(model,path)` | 保存顺序或 forward 训练断点：结构、权重、冻结关系、优化器类别、步数和矩状态；顺序模型另保留共享关系 |
| `restore(path)` | 读取断点，返回模型及恢复后的优化器 |
| `save_model(model,path)` | 导出原生推理 IR v1 JSON；不包含优化器；forward 使用 Program 格式，供 model() 或 wgt-native adaptive 加载 |
| `benchmark(model,input,warmup,repeats,name)` | CPU 单向量前向计时；warmup 0..10000，repeats 1..100000；返回并登记 us 单位 mean/p50/p95、次数与范围 |

脚本整数参数目前限制为 0..1000000000；列表索引仍需满足实际长度。
每次脚本最多创建 64 个模型、1024 个流句柄（包括已关闭的流）。运行结束自动释放所有资源。
每个模型 prepare 的张量预算是 256 MiB，**不包含**解析、脚本列表、训练 tape、梯度、优化器和保存副本；
训练不是零分配。不要把该值当成进程内存上限。

## 梯度与参数更新

声明式顺序模型支持单向量 linear/relu/tanh/sigmoid/identity；`fn forward` 模型还支持
可微 gate/route/softmax、显式写回与实际执行的循环/分支计算，详见[可微程序](differentiable-programs.md)。
两类模型均可用 fit/reinforce、SGD/Adam；训练输入是向量的列表，逐样本
建立 tape 并累加梯度。MSE 在样本及输出维度平均；交叉熵 targets 是每样本一个类别整数，
输入模型应输出未经 softmax 的 logits。REINFORCE 的 advantages 是显式常量，不对优势求导。

SGD 无 momentum/weight decay；Adam 固定 beta1=0.9、beta2=0.999、eps=1e-8，无 weight decay。
同一模型句柄保留优化器矩和步数，学习率在每次更新显式指定。当前不支持独立优化器对象、
混合精度、梯度累积接口、裁剪、任意自定义损失反传或 GPU 训练。forward 模型支持受限的
`fn loss(prediction,target)` 一阶反传，约束与示例见[可微程序](differentiable-programs.md)。

原生 IR 的 resident/view/add 梯度汇总到存储所有者，共享参数只更新一次；冻结依赖沿用 Session
规则，冻结参数不更新。可训练的 file 存储拒绝，冻结 file 可读取。全部批次与候选更新检查通过后
再提交权重和优化器状态；非有限输入、损失、梯度、参数/矩状态均报错，不部分更新。
`mlp` 使用独立确定性 Xavier uniform 权重、零 bias；不承诺与 Python/PyTorch 的 seed 初始化逐位相同。

## 强化学习与 benchmark 边界

[REINFORCE 示例](../examples/native_training/reinforce.wgt) 用普通 WGT 函数定义奖励环境，
从随机策略开始采样、累积批次、更新。RL 是原生可执行的策略梯度和 GAE 原语，当前没有内置
PPO/DQN、replay buffer、Gym 适配器、多进程环境或外部模拟器。

GAE 使用 `delta[t] = reward[t] + gamma*(1-terminated[t])*next_value[t] - value[t]`；
在 terminated 或 truncated 时停止优势递推。截断仍使用最终观测的 next_value，调用者不得
误传自动 reset 后的观测价值。输入是一条顺序轨迹（可串接带显式边界的 episode），不是交错环境批次。
最后一项以调用者提供的 next_value bootstrap，未提供未来轨迹的优势尾项视为 0。

benchmark 仅计 CPU 前向，排除脚本 dispatch、数据读取和训练。warmup 前另做一次验证前向；
p50/p95 为排序后 floor((N-1)*q) 项，无插值。没有跨模型速度结论、GPU 同步测量或统计显著性承诺。
输出的训练数据损失、采样奖励只描述相应合成例子，不能视为泛化/真实任务质量证明。

## 保存与续训

文件父目录须先存在。保存先写同目录临时文件、sync 后以 hard link 原子发布；目标存在即拒绝，
包括符号链接，不接受隐式覆盖。文件系统须支持硬链接。完整模型、断点和报告应保留在
`weights/<experiment>/`，见[存储约定](storage.md)。
原生训练断点不是传统 WeightLang manifest 制品，不可交给 weightlang.load 或权重版本库直接使用。
模型初始化代码/脚本、数据源与加载位置、脚本局部变量/RNG、环境状态和训练进度不自动写入断点；
恢复的是模型与优化器，调用者需显式恢复数据/环境/采样进度。相同后续批次在同一后端可逐位续接 Adam 更新；跨 CPU/GPU 以数值容差比较。
`save_model` 导出结构和实际权重，保留 view/add 关系，冻结文件权重被物化，不依赖原始参数文件。

当前同一编译单元可声明原生向量模型并在 main 中调用 `模型名("adam")` 构造。支持固定 f32 向量、linear/relu/tanh/sigmoid/identity；复杂图算子及 std block 尚未迁入原生构造器。也可构造 MLP 或读写原生 IR；
将全部 WGT 图、标准库、模型制品和 GPU 训练迁移到原生前端仍是后续工作，不通过 Python 回退掩盖缺口。

编辑器高亮及本地 Tree-sitter 已覆盖 WGT 0.5 函数和控制流；语义诊断仍由 Rust 入口提供。安装及发布边界见 [编辑器支持](editors.md)。

持久化的完整例子是 [checkpoint.wgt](../examples/native_training/checkpoint.wgt)。在仓库根目录执行：

```sh
mkdir -p weights/native_training
./target/debug/wgt run examples/native_training/checkpoint.wgt
```

它只创建全新的 `weights/native_training/checkpoint.json` 与 `model.json`；再次运行会拒绝覆盖。
前者验证 Adam 续训，后者可交给 `wgt-native weights/native_training/model.json 1048576 --device cpu`
并从标准输入逐行发送 `[0.25]` 等单样本向量。

## 原生 weightc 模型构建器

本地新增 Rust `weightc` 二进制，与已有 Python 分发的同名命令不同；使用明确路径避免 PATH
歧义。旧的 Python 构建入口和 `wgt build` 源码打包入口已移除。

```sh
cargo build -p weightlang-wgt --bin weightc
./target/debug/weightc check examples/native_training/build.wgt
./target/debug/weightc build examples/native_training/build.wgt -o weights/native_build
```

`weightc check SOURCE` 只调用原生脚本解析器，限制与 `wgt run --check` 相同。
`weightc build SOURCE [-o DIRECTORY] [--max-operations N]` 执行 `wgt 0.5` 的零参数
`main()`，必须返回模型句柄。默认输出为当前目录的 `weights/<源码文件名去扩展名>/`，
默认操作预算 10000000，N 必须正整数；数据等相对路径仍相对原始脚本目录。
完整示例 [build.wgt](../examples/native_training/build.wgt) 从初始化开始原生训练回归模型，
返回训练后的模型；整个构建与重载不需要 Python/PyTorch、虚拟环境或外部进程。

输出目录包括：`model.json`（结构和实际权重）、`checkpoint.json`（模型与优化器）、
`source.wgt`（入口快照）、`manifest.json`（格式 weightlang.native-build.v1、源码/模型 SHA256、
编译器版本、更新步数、预算和指标）。最后发布 manifest 作为完成标记；消费者应拒绝没有
完成标记的构建目录。推理使用 `wgt-native DIRECTORY/model.json BUDGET_BYTES --device cpu`。
断点可通过脚本 `restore(path)` 恢复，再训练并返回模型到新的输出目录。

已有输出目录或符号链接一律拒绝，不提供覆盖选项。构建预留目标目录，同盘暂存产物；
执行失败时删除空预留目录，发布中失败保留部分输出供诊断，不宣称整个目录原子发布。
脚本显式 save/save_model 的外部写入不在构建回滚范围内。

零更新标记 initialized，有优化器更新历史标记 updated；updated 不意味着通过任务质量验收。
恢复断点的步数包含历史更新。入口快照与哈希并不包含数据、外部模型依赖或脚本 RNG/数据进度，
不保证仅凭制品完整复现训练；应另行保留数据和依赖。原生支持范围仍限于本页列出的算子，
构建入口必须有 main 并显式返回模型句柄；模型声明可与 main 共存。不自动执行训练、不回退 Python。

## 同一脚本的解释执行与 Rust 编译执行

WGT 脚本负责初始化、训练和显式保存权重；weightc 负责编译，不在编译阶段执行这些操作。
两种模式共用解析器、动态值语义、内置函数和原生训练 runtime。

```sh
cargo build -p weightlang-wgt --bin wgt --bin weightc
# 解释执行，默认模式：
./target/debug/wgt run examples/native_training/reinforce.wgt
# 先编译后执行同一脚本：
./target/debug/wgt run examples/native_training/reinforce.wgt --compiled
# 只编译，不执行；输出文件不能已经存在：
./target/debug/weightc compile examples/native_training/reinforce.wgt -o .cache/weightlang/reinforce
# 后续独立运行，无需源码、Python、Cargo 或 rustc：
./.cache/weightlang/reinforce --max-operations 10000000
```

完整命令契约：

- `wgt run SOURCE [--check | --compiled] [--max-operations N]`：默认解释执行；
  --check 与 --compiled 互斥，默认 N=10000000。--compiled 调用 Cargo 后运行临时可执行程序，
  结束后清理该程序；编译失败直接报错，不回退解释器。
- `weightc compile SOURCE -o EXECUTABLE`：SOURCE 必须是 `wgt 0.5`。
  `-o/--output` 必填，自动创建父目录，拒绝覆盖文件/目录/符号链接；生成主机平台可执行文件。
  没有 main 的模型声明、交叉编译、GPU 训练不在此入口支持范围内。
- 生成程序接受 `--root DIRECTORY` 与 `--max-operations N`。默认 root 是编译时原始脚本的
  绝对目录；移动程序/数据后应显式设置 root。默认预算 10000000，零预算在第一次操作时失败。
  数据和模型依赖不嵌入二进制，文件读写仍遵循脚本的路径与拒绝覆盖规则。

编译把函数、表达式、分支和循环直接生成为 Rust 代码，交给 Cargo/rustc 的 release 后端；
生成程序不解析 WGT、不遍历 AST。动态值检查、函数分派、变量环境及操作预算仍保留。
训练/保存内置函数与解释器调用同一份 Rust 实现，不复制自动微分、优化器或算子。
两种模式均要求 main 返回可序列化 JSON 值；保存模型应显式调用 save_model/save。
短路求值、函数局部作用域、块内重绑定、递归限制、预算计数和行号诊断保持一致。
编译不增加静态类型推断：未执行分支中的未知名称和错误调用仍可在执行时失败。

本地编译需要 Cargo、rustc、系统链接器、匹配版本的 runtime crate 源码及已缓存的 Rust 依赖。
默认使用构建 weightc 时仓库内的 runtime 路径；搬移工具链后可设置
`WEIGHTLANG_RUNTIME_SOURCE=/path/to/crates/wgt-runtime`。Cargo 使用 --offline，不隐式联网下载。
构建中间文件在当前工作目录 `.cache/weightlang/aot/` 的磁盘缓存；临时 Rust 项目结束即清理，
Cargo target 保留加速后续编译。不同编译调用通过锁保护共享产物，最终文件禁止覆盖。
这不是已实现的可独立分发编译 SDK，编译时仍需上述源码与工具链。

首次编译成本与运行成本分开；编译主要省去 AST 调度，并不保证训练吞吐量普遍提高。
CPU 张量算子原本已经是 Rust，训练加速幅度取决于脚本调度占比。

完整权重生成例子 [compiled.wgt](../examples/native_training/compiled.wgt)：

```sh
mkdir -p weights/native_script
./target/debug/weightc compile examples/native_training/compiled.wgt -o .cache/weightlang/train-native
./.cache/weightlang/train-native
```

该脚本自己调用 mlp、fit、save_model、save，生成 model.json 与 checkpoint.json；重复执行拒绝覆盖。
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
wgt demo.wgt
wgt demo.wgt --compiled
wgt --offline --locked demo.wgt
wgt demo.wgt --check
weightc --offline --locked compile demo.wgt -o demo
```

`wgt FILE.wgt` 等价于 `wgt run FILE.wgt`，支持 --compiled、--check、--max-operations，
以及全局 -C/--directory、--offline、--locked。只有模型声明时可检查；运行必须显式定义 main。
完整本地无下载例子：[inline.wgt](../examples/native_training/inline.wgt)。

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
它不导入 WGT 函数，不扩展 model() 接受的格式：传统 manifest/safetensors 制品不能自动作为
原生 IR JSON 训练。原生训练支持范围仍以 model/restore 等接口契约为准。

编译模式会把已解析目录映射写入程序；数据/包文件不嵌入可执行文件，移动缓存后需重新编译。
独立可执行程序不自动联网解析或重新校验包，--root 仅影响脚本相对路径，不改变依赖缓存路径。
底层 wgt-native script 与 Rust run/run_file 不负责包管理；需要自动依赖时使用 wgt 入口。

## weightfs 自动挂载

当前 Linux CLI 和解释/AOT 脚本在构造、加载或恢复模型后，默认尝试将模型挂载到
`$XDG_RUNTIME_DIR/weightlang/weightfs/<pid>-<session>/`，stderr 报告实际路径。
可像文件一样读取和修改参数 JSON；单文件提交与训练/预测互斥，旧版本编辑返回冲突。
手工编辑保留优化器历史，不自动保存。`WGT_WEIGHTFS=on|auto|off` 控制挂载要求（默认 auto）；
纯检查与编译不挂载。完整文件协议、环境要求、错误及预算见 [weightfs v1](weightfs.md)。

## 采样与独热目标

普通函数可用 `sample p { then { a } else { b } }` 只执行抽中的表达式，seed 控制随机流。
`onehot[4] action` 将动作索引转成四类数值列表；复杂索引用括号。等价原生函数为
`one_hot(index,classes)`，类别数 1..1000000，索引为 [0,classes) 整数，分配前扣除 classes 操作预算。
两者不使离散动作可微；onehot 目标可用于 MSE/自定义损失，cross_entropy/reinforce 仍接收类别索引。
详见[语法与边界](differentiable-programs.md)，可运行例子 [sample_branch.wgt](../examples/native_training/sample_branch.wgt)。

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

## 实验性 Rust GPU 训练 API

**Rust API 实验性，WGT 绑定尚未接入**。`training::Trainer::new` 与 `restore` 默认 CPU，
CPU 的运算顺序、SGD/Adam 和 `weightlang.native-training.v1` 断点格式保持不变。
开启 runtime 的 `gpu` feature 后，可显式选择 `TrainingBackend::Gpu`；不增加脚本语法或 CLI 参数，
不依赖设备池。支持顺序 IR 的 linear/relu/tanh/sigmoid/identity、f32、MSE、交叉熵与 REINFORCE，
优化器支持无动量 SGD 和 Adam。不支持任意 forward 程序或多 GPU，也不自动回退 CPU。

```rust,no_run
use std::path::Path;
use weightlang_runtime::training::{
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
事务结束回读并释放临时设备 arena；没有跨步常驻权重或吞吐量提升承诺。`predict` 仍用 CPU，
`export`/`save` 使用已提交 CPU 副本，设备丢失后仍可保存；后续 GPU 调用报错，可显式切回 CPU。
外部 Session/weightfs 参数写入会被下一次训练读取，保留已有矩和步数；weightfs 仍通过同一模型锁串行访问。
`save` 保存完整 `first`/`second`/`steps`，格式仍是 `weightlang.native-training.v1`；CPU 断点可恢复后显式
选择 GPU 续训，GPU 断点可直接由默认 CPU 恢复，跨后端允许 f32 误差，不承诺逐位相同。
冻结 file/packed 参数可物化用于计算，可训练 file 仍拒绝。

GPU budget 是每次调用的保守数值 payload 上界：Session managed bytes、两个既有优化器数组、
更新阶段保留的完整梯度（包括冻结参数），
以及 host/device arena、回读、返回梯度/候选、物化暂存和操作 uniform 的预留。
不包括调用者 inputs/targets、对象元数据、allocator 多余容量、驱动与命令分配，不能当进程内存硬隔离。
默认 CPU 训练的预算范围不变。

数值测试对 CPU 梯度使用 `atol=2e-5, rtol=2e-4`，20 步权重对照使用 `3e-5, 3e-4`；
容差涵盖 f32 归约顺序及设备激活函数差异，不保证逐位相同或适用于任意病态模型。
GPU 损失中心有限差分取 h=1e-3，使用 `atol=rtol=2e-3`，平衡 f32 相减舍入与二阶截断误差，
ReLU 测试避开零点。真实硬件测试默认 ignored；显式运行，缺硬件时打印 SKIP，验收设置
`WGT_REQUIRE_GPU=1` 则缺硬件失败：

```sh
WGT_REQUIRE_GPU=1 cargo test -p weightlang-runtime --features gpu --test gpu_training -- --include-ignored --nocapture
WGT_REQUIRE_GPU=1 cargo test -p weightlang-runtime --features gpu --lib training::gpu::tests::device_loss_preserves_last_committed_model -- --ignored --nocapture
```

模型输出分析可使用 `normalize_probabilities`、`entropy`、`kl_divergence`、
`js_divergence`、`cross_entropy_probs` 和 `temperature_softmax`。
完整签名、分布检查、零概率及预算规则见[概率分布分析](standard-library.md#概率分布分析)。
这些接口仅计算脚本数值；不能作为新增 `fit` 损失名称或自动微分表达式使用。
