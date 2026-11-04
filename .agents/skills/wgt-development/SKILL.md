---
name: wgt-development
description: 编写和验证本仓库当前原生 WGT 0.5 模型、训练与概率控制表达式。
---

以 docs/reference.zh-CN.md、docs/differentiable-programs.md 和 Rust 实现为准；不是已发布功能承诺。
只接受 wgt 0.5，模型、forward、loss 和 main 可在同文件；执行不得依赖 Python。
语境能力见 language/source_capabilities.json；共同正负语料在 language/conformance.json。
函数共用 parser 不表示普通 fn 与 tensor forward/loss 语义相同；模型声明独立解析。
language/primitive_specs.json 是制品 graph 原语域，不是源码内置表。编辑器容错不证明源码有效。
无需 import 的函数、张量 intrinsic、常量和声明类型统一登记在 language/prelude.json；
script/prelude.rs 直接消费它，编辑器快照由它生成，领域实现路径见各 entry。
普通函数与 tensor 上下文分开登记，不自动引入 std 模块。
CPU forward 自定义算法：以 none 构造，differentiate(model,inputs,targets) 对受限 fn loss 求批次平均梯度；
gradient_snapshot(handle) 提供loss/parameters/gradients/names/shapes/state/steps，普通WGT循环自行计算候选参数和状态，
commit_gradients(handle,candidates,state) 原子提交并释放；release_gradient(handle) 放弃快照。
最多16存活句柄、64MiB计量预算，状态最多32层/1MiB且不含资源句柄；普通host程序不自动可微。
forward-training.v2 checkpoint保存自定义状态和步数，仍读旧v1；不自动执行算法，不支持该路径的顺序/GPU模型。
完整契约 docs/native-training.md；真实例子 examples/native_training/custom_momentum/model.wgt，beta=0为SGD。
保存推理模型用 save_model(model,path)，加载用 load_model(path,optimizer) 创建新优化器；
保存训练断点用 save_checkpoint(model,path)，load_checkpoint(path) 恢复原优化器状态。
旧 model/save/restore 仅为 legacy 别名，新例子使用新名；所有参数必填，保存拒绝已有目标。
本地运行 cargo test -p weightlang-runtime --test source_conformance 与 node scripts/test_source_conformance.cjs。

- parameter p: prob = 0.7 使用可训练 logit，初值必须在 (0,1)；frozen p: prob = 1 支持精确端点。
  读取为 f32[1] 概率，底层参数/梯度暴露 logit；不得 write(prob) 或直接作为 linear 的参数。
- 普通 f32 权重不默认限制到 [0,1]；当前没有 0/1 布尔参数或自定义权重范围声明。区间含零不能防止除零，普通脚本的除零和非有限结果报错。
- mix p { then { a } else { b } } 用于 forward/loss 张量混合，每支一个表达式。
  与 route 相同，推理在精确端点短路内联分支；训练/VJP 保留完整梯度。提前 let 的计算不会省略。
- fn loss(prediction,target) 是无模型参数的张量损失，返回 f32[1]；显式 fit(m,x,y,lr,"loss") 或 train 才训练。
  支持 mean、square、减法与已有张量算子；不是任意宿主函数自动微分。
  平均可写 `x.mean`、`square(prediction-target).mean`，等价 `mean(x)`；后缀优先于一元/二元运算。
  普通函数接受非空一维有限 f64 列表并返回标量；forward/loss 的 f32[N] 归约为可微 f32[1]。
  不支持其他归约后缀或 axis 参数；完整例子 examples/native_training/postfix_mean/model.wgt。
- `x[start:end]` 是零基左闭右开一维切片，省略端点默认 0/长度，不裁剪越界值。
  普通列表允许运行时整数端点和空结果；forward/loss 仅接受编译期数字常量表达式且拒绝空结果，梯度只进入选中区域。
  可接调用/括号/索引/切片及 `.mean`，不支持负索引、步长、多轴或下标赋值。
  `slice(x,start,length)` 保持旧张量长度语义：`slice(x,1,1)` 等价 `x[1:2]`。
  契约见 docs/reference.zh-CN.md；可运行例子 examples/native_training/slicing/model.wgt。
- sample p { then { a } else { b } } 仅用于普通函数，概率为 [0,1] 标量，仅执行一边。
  seed 控制 RNG，精确端点不抽随机数；forward/loss 中拒绝，类别采样使用 sample from model(observation)，旧 sample(model,observation) 已移除。
- mix/sample 分支可嵌套、可选尾随分号，不支持块内 let/return；普通 if 仍要求 bool。

完整可运行例子：examples/native_training/prob_mix_loss.wgt、sample_branch.wgt、lazy_routing.wgt。
新例子解释轴、共享参数与初始化/训练区别。检查完整源码，验证解释/AOT、梯度、回滚与保存续训。
公共语言变更同步主参考、专题文档和此技能；运行 git diff --check。
模型构造/加载与训练分离；保留模型放 weights/，不覆盖已有产物。遵循 docs/storage.md。


独热编码使用 `onehot[N] index`（N 为 1..1000000 字面量），例如 `onehot[4] 2`。
复杂索引用括号；等价 `one_hot(index,classes)` 支持动态类别数。只限普通函数，不对索引求导。
输出长度按类别数扣预算，索引必须为 [0,N) 整数；参考 sample_branch.wgt。

普通脚本标准库另见 docs/standard-library.md；不要把列表函数当作 forward/loss 张量算子。
数据接口：zeros(count)、ones(count)、full(count,value)、linspace(start,end,count)、
one_hot(index,classes)、concat(x,y)、vector_slice(x,start,end)、gather(x,indices)、
compress(x,mask)、reverse(x)。参数全部必填；有限 f64 一维列表，允许空列表；
mask 必须 bool，索引为合法零起始整数，切片左闭右开；输出上限 100 万项。
linspace 含端点，count=1 只给起点；独热 classes>=1。预算在输出分配前检查。
完整可运行例子：std/data/construction.wgt、std/data/indexing.wgt。


概率分析标准库：normalize_probabilities(weights)、entropy(p)、kl_divergence(p,q)、
js_divergence(p,q)、cross_entropy_probs(p,q)、temperature_softmax(logits,temperature)。
全部参数必填，非空有限 f64 一维列表，双分布等长；分布项 [0,1]、总和绝对误差<=1e-9，
通过后修正总和；权重归一化要求非负且有正质量，温度有限且>0。单位 nat。
KL/交叉熵 p>0 且 q=0 报错，JS 支持不相交分布。无张量梯度或新增 fit 目标。
例子 std/probability/distributions.wgt、temperature.wgt；完整契约见 docs/standard-library.md。

原生 RL：`train model on (inputs,targets) rate lr clip max_norm;` 等价 train(model,inputs,targets,lr,max_norm)，仅 forward + fn loss，sgd/adam，正有限 lr/max_norm；全批次平均梯度做全局 L2 裁剪后原子更新。`target NAME: f32[N]` 指定 loss 第二形参宽度/名称，默认同输出，不增加 forward 输入。forward/loss 支持有限数字向量常量、detach、exp、log、log_softmax、sum、minimum（相等各半梯度）、clip（有符号字面量边界、端点零梯度）。全部无广播/默认值。完整契约 docs/reinforcement-learning.md，运行例子 examples/native_training/ppo/model.wgt；不支持单独 backward/step、跨调用梯度对象、环境/RNG 全进程续训。

## 原生测试与评估

`wgt test` 发现 tests/*.wgt 的无参 test_* 函数，每个测试独立运行且不执行 main。
语法与示例见 references/native-testing.md；完整契约 docs/native-testing.md。
