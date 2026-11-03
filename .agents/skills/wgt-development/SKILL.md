---
name: wgt-development
description: 编写和验证本仓库当前原生 WGT 0.5 模型、训练与概率控制表达式。
---

以 docs/reference.zh-CN.md、docs/differentiable-programs.md 和 Rust 实现为准；不是已发布功能承诺。
只接受 wgt 0.5，模型、forward、loss 和 main 可在同文件；执行不得依赖 Python。

- parameter p: prob = 0.7 使用可训练 logit，初值必须在 (0,1)；frozen p: prob = 1 支持精确端点。
  读取为 f32[1] 概率，底层参数/梯度暴露 logit；不得 write(prob) 或直接作为 linear 的参数。
- mix p { then { a } else { b } } 用于 forward/loss 张量混合，每支一个表达式。
  与 route 相同，推理在精确端点短路内联分支；训练/VJP 保留完整梯度。提前 let 的计算不会省略。
- fn loss(prediction,target) 是无模型参数的张量损失，返回 f32[1]；fit(m,x,y,lr,"loss") 才训练。
  支持 mean、square、减法与已有张量算子；不是任意宿主函数自动微分。
- sample p { then { a } else { b } } 仅用于普通函数，概率为 [0,1] 标量，仅执行一边。
  seed 控制 RNG，精确端点不抽随机数；forward/loss 中拒绝，既有 sample(model,observation) 保留。
- mix/sample 分支可嵌套、可选尾随分号，不支持块内 let/return；普通 if 仍要求 bool。

完整可运行例子：examples/native_training/prob_mix_loss.wgt、sample_branch.wgt、lazy_routing.wgt。
新例子解释轴、共享参数与初始化/训练区别。检查完整源码，验证解释/AOT、梯度、回滚与保存续训。
公共语言变更同步主参考、专题文档和此技能；运行 git diff --check。
模型构造/加载与训练分离；保留模型放 weights/，不覆盖已有产物。遵循 docs/storage.md。


独热编码使用 `onehot[N] index`（N 为 1..1000000 字面量），例如 `onehot[4] 2`。
复杂索引用括号；等价 `one_hot(index,classes)` 支持动态类别数。只限普通函数，不对索引求导。
输出长度按类别数扣预算，索引必须为 [0,N) 整数；参考 sample_branch.wgt。
