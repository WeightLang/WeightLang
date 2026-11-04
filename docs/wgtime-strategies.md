# wgtime 驻留 WGT 自定义训练策略

状态：先写本契约，再实现 `crates/wgtime-service/src/strategy.rs`；本地模块与定向测试已实现。模块验证、IPC 接线和整机性能验收分别记录，模块存在不表示 AOT 已完成迁移。现有语言的训练语义以 [native-training.md](native-training.md#静态训练策略-train--using) 为准。

## 编译与注册

`Strategy::compile(source)` 使用当前 WGT 0.5 公共 parser，选择同步 `strategy(model,inputs,targets)`；`compile_entry(source,entry)` 可选择其他同签名入口。也允许第四形参接收显式 JSON 配置，供客户端仅发送标量超参数和小型结构。入口及可达的同步辅助函数在注册时转换为私有控制流指令和表达式指令；执行不重新解析源码、不调用普通宿主解释器、不加载用户原生动态库。策略标识、owner、epoch、注册数量及源码帧上限由服务连接层管理。策略只读代码可以复用，可变执行状态每次独立创建。

支持子集：有限标量、bool、null、字符串、列表、索引、四则运算/比较、短路逻辑、let/赋值、if、while、for、break/continue、return、assert；静态同步辅助函数禁止递归。允许 `len`、`range`、`append`、`sqrt`、`abs`、`min`、`max`，以及现有 `differentiate`、`gradient_snapshot`、`differentiate_at`、`commit_gradients`、`release_gradient`。range 为零基左闭右开整数序列，两个参数。其他语法和调用在注册时明确拒绝，包含宿主文件、网络、工具、GPU、模型构造/加载/保存、异步、随机数及资源观测。未可达的模型 forward/loss 与 main 不作为策略执行；模型编译和加载仍由服务独立负责。完整源码中存在 main 不使 main 可执行。

服务端受限指令是 WGT 策略的编译目标，不是固定优化器名称映射。WGT 编译器/AOT 后续应提取同一入口并注册此模块，禁止让客户端执行 gradient_snapshot、下载参数/梯度或上传完整候选参数来替代驻留计算。本模块采用有界指令虚拟机；不宣称该首期 VM 已达到原生 AOT 策略性能，需按下述基准验收。

## 执行与所有权

`execute(&self, session, inputs, targets, remaining, memory)` 或带小型配置的 `execute_with_parameters` 在调用方持有的独占会话锁中同步完成。输入是服务已经验证的数据缓冲借用；模块不接收文件路径、不开放新的能力。模型句柄只代表该次传入的会话，不能转换为数字或跨会话使用。梯度是每次执行的局部不透明句柄，最多 16 个，执行成功或失败均销毁全部剩余梯度；策略返回值不出服务。外部响应固定只含提交次数、最后成功提交损失和剩余操作预算，不含参数、梯度或任意策略返回对象。checkpoint 由现有服务授权路径显式导出，单独计量。

各次 differentiate/临时求导使用现有 AdaptiveSession 原语，保留同一 f32 批次、稳定参数顺序、冻结规则和 stale revision 检查；每次 commit 复用现有原子参数/状态/步数提交。零次、一次和多次显式提交都合法。后续指令失败保留此前成功提交，服务不得自动重试整个策略。SAM 用 WGT 代码计算扰动、differentiate_at 与基础优化器；没有隐藏二次提交，也不会把扰动写到真实模型再回滚。用户状态和步数继续由现有 checkpoint v2 保存/恢复。

操作预算同时约束 VM、循环、复制和反向原语；耗尽不退款。内存参数限制局部梯度与 VM 累计分配计量，分配前校验，计量包含列表槽、字符串、执行输入/配置与状态快照；重复分配也消耗此预算，释放梯度只归还梯度保留额。这是保守有界执行配额，不是 OS RSS 上限，且不含既存模型、注册代码和服务帧存储。注册源码与可达调用图也有硬上限。服务须另行限制注册代码总量及已驻留模型/数据内存，不能把此参数当成整个进程的内存限制。

## 验证边界

定向验证需比较同模型、batch、seed 的直接 WGT 和驻留自定义优化器/SAM，包含参数、损失、自定义步数和 checkpoint 续训；验证零/多次提交、提交后错误、过期梯度、内存/计算预算失败和并发独立会话。内部测试允许在服务本地检查参数；公开执行返回类型不得携带它们。连接 owner/epoch、失效数据句柄及 IPC 统计由连接层测试负责。参数/梯度网络字节必须为 0；单测固定响应字段仅证明模块不返回这些数据，不能代替 IPC 字节统计。

本地 `cargo test -p wgtime-service --test strategy` 验证 6 项：零/多提交及错误保留、能力与递归拒绝、失效梯度及预算、原有 SAM 六步对直接 WGT 输出一致和步数、并发独立会话、checkpoint 续训和配置参数。固定响应只有 commits/loss/remaining 三字段，策略 return 丢弃。静态变量/字面量索引链降低为直接叶值读取，避免遍历权重时反复复制整个快照。

性能比较须使用 release 构建，区分注册编译、首次运行和 warm 执行，记录同一批次的原始多轮结果和完成的 commit 数。必须同时报告本地 WGT、自定义策略 VM 及完整 IPC 路径；本模块定向测试不能替代 [wgtime 性能合同](wgtime.md#性能验收) 的整机验收。
