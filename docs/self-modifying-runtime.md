# WGT 0.5 自更新模型

自更新前向和普通训练函数使用同一 `fn` 语法，文件头只接受 `wgt 0.5`。
完整声明、操作符签名、默认值与限制见[语言参考](reference.zh-CN.md#自更新前向函数)。
旧的 adaptive 文件头、register、branch、jump 和标签源码不再支持；
已有内部 Program JSON 是单独版本化的模型制品，不是源语言。

模型先声明 `model`、`input/output ...: f32[N]` 及显式初值的 `parameter/frozen`，
然后用 `fn forward(x)` 表达张量计算、`write`、`if positive(...)` 和 `while`。
同文件 main 调用 `模型名("none")` 与 `predict`。输入不含 batch；当前仅 CPU。
可执行且带轴、状态、梯度说明的完整示例：[model.wgt](../examples/self_modifying/model.wgt)。

```sh
weightc check examples/self_modifying/model.wgt
wgt run examples/self_modifying/model.wgt
wgt run examples/self_modifying/model.wgt --compiled
# 保留底层有状态张量会话接口，用于连续输入或原生 VJP：
wgt-native adaptive examples/self_modifying/model.wgt 4096 100 --check
```

`wgt-native adaptive SOURCE BUDGET_BYTES MAX_STEPS [--check]` 的后三项分别是字节预算、
正整数指令预算、可选检查。SOURCE 可以是同一 0.5 源码或内部 Program JSON。
`--check` 不运行 main/forward。实际会话的 stdin 每行是输入 JSON 向量，
成功响应为 `{"output":[...],"revision":N,"steps":N}`，失败为 `{"error":"..."}`。
输入 `[2,3]`、`[4,3]`、`[-2,7]` 时示例依次输出 `[3]`、`[9]`、`[7]`。
不预热自更新模型，避免加载过程改变状态。

## 状态与失败

- 一次调用是一个权重事务：内部写入立即可见，成功 return 后一起提交。
- 非有限运算、未定义寄存器、落出程序或步数耗尽，回滚本次全部参数写入。
- 有写入的成功调用 revision 加一，多次 write 仍只加一；只读调用不改变 revision。
- 每次调用重置寄存器定义状态；参数保留，重新 prepare 从给定初值重新开始。
- 内存预算覆盖参数、回滚副本、所有寄存器、最大寄存器长度的临时缓冲；
  不包含源码/JSON 解析、程序元数据、容器元数据或进程 RSS。
- 每条实际执行指令计一步，包含标签、跳转、return。步数不是时间：矩阵乘法仍计一步，
  其规模受已检查的形状和内存预算约束。
- 当前没有共享别名、文件权重、量化、并行执行、GPU 或内置优化器状态。

上述事务提交规则对应 `run` 推理调用。下述可微 `forward` 始终返回显式状态，不提交写入；
开发者通过下一次调用的参数绑定选择延续哪些状态，基础模型不会被训练评估覆盖。

## 原生自动微分

Rust `AdaptiveSession::forward(input,max_steps,tape_budget_bytes)` 返回不提交状态的 ForwardPass。
`forward_with_parameters(input,bindings,max_steps,tape_budget_bytes)` 可显式替换本次初始参数。
`ForwardPass::backward(output_gradient,state_gradients)` 支持重复反向计算，
然后读取 input_gradient、parameter_gradient；最终参数状态通过 state(name) 获取。
所有结果均由 Rust 计算；支持一阶梯度，硬分支只沿实际路径传播，路由选择本身不可微。
缺省最终状态梯度为零；写回后的参数使用保持梯度链，write 不是 detach。
这些 API 的 tensor 预算必须为会话及保留 tape 留出空间，不是 RSS 上限。

JSON 传输命令 `forward` 接收 input、parameters（默认空映射）、tape_budget_bytes（默认剩余预算）；
`backward` 接收 tape、output_gradient、state_gradients（默认空映射）、retain（默认 false）；
`release` 释放 tape，`parameters` 读取当前参数。每会话最多 64 个活动 tape。
API 详情以 `crates/wgt-runtime/src/adaptive/transport.rs` 为准。
高层普通函数暂不暴露 VJP/外部优化器接口；`predict` 用于提交推理状态。

验证包括写后使用、循环多次写入、分支路径、数值差分、冻结参数、预算、回滚与解释/AOT 一致性。
