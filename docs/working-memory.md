# 可微工作内存与分步执行（WGT 0.5，本地实现）

工作内存属于一次 forward 任务，不是模型参数、优化器状态或跨请求缓存。
控制器可以生成地址权重，实际读写由 Rust runtime 执行；`fit/reinforce` 的梯度穿过读写，
不需要 Python。构造、检查、编译与加载不执行前向或训练。

## forward 接口

所有参数必填，无默认参数。仅用于 `fn forward` 的张量执行环境，亦可用于相同张量子集的
`fn loss`；不能在普通 main 中直接调用。内存形状固定，实际存储到执行分配指令时才创建。

| 完整签名 | 参数、返回与约束 |
| --- | --- |
| `memory_alloc(slots,width)` | 两个源码整数常量，分别为 1..16384；返回专用内存句柄。按 `[slots,width]` 行优先分配零初始化 f32 工作区，计入当前会话内存上限。写法为 `let bank=memory_alloc(N,D);`。 |
| `memory_read(bank,address)` | address 为 f32[N]，各项有限且在 [0,1]，返回 f32[D]；`y[j]=sum_i address[i]*M[i,j]`。不隐式归一化；需要概率地址时显式 softmax。 |
| `memory_write(bank,address,value);` | address 为 f32[N]，value 为有限 f32[D]；`M'[i,j]=(1-address[i])*M[i,j]+address[i]*value[j]`。按槽位插值覆盖，不是累加写入。 |
| `memory_free(bank);` | 立即释放工作区并归还当前会话可用内存。重复释放、未执行分配或释放后读写报错。 |
| `resource_state()` | 返回 f32[3]：`[剩余前向工作量, 可用工作区字节, 当前工作区字节]`；在该指令自身计费后观察。是外部状态观测，无梯度。 |

one-hot 地址精确选择一个槽位；全零写地址保持原值，全零读地址返回零。
软地址会扫描全部 N×D 元素，即使 one-hot 也没有稀疏加速承诺。
read 对地址和旧内存求导；write 对地址、旧内存和新值求导。例如
`dM'/dM=1-address`、`dM'/dvalue=address`、`dM'/daddress=value-M`。
释放工作区不会截断此前的梯度链；tape 为反向传播保留已使用的值版本。

句柄不可复制、返回、保存为参数、参与张量运算或改变形状。循环中可在释放后重新执行
同一名称的分配，新的内容全部为零；未释放时再次分配同一句柄报错。
分配与释放的决定是离散控制流，没有对内存大小或分配动作的伪梯度。
可以用现有 `if positive(...)`、循环和可训练地址控制实际执行路径。

### 生命周期与计量

每次完整推理、evaluate 和每个训练样本都从空工作区开始；正常结束、失败、取消自动释放。
工作内存不进入 `save model` 文件或优化器断点。地址等参数仍正常保存和训练。

会话内存上限包括参数、回滚副本、寄存器、scratch 和当前存活的工作区 f32 数据；
普通脚本 forward 会话上限仍为 64 MiB。分配前检查剩余额度，不静默截断。
元数据、优化器、梯度、序列化和进程其他开销不在该额度中；它不是 RSS 硬上限。
训练 tape 另有既有的预算，包含值和梯度；释放工作区不会退还已保存的 tape 额度。

alloc/read/write 各计 `1+N*D` 工作单位，free/resource_state 各计 1；其他指令沿用既有计价。
因此 `steps()` 现在是工作量，不再总等于指令条数。计价不是 CPU 指令、FLOP 或墙钟时间。
拒绝超额指令不会执行该指令，此前完成的工作仍计费。反向传播不按 FLOP 单独计费。
资源向量用 f32 表示，大额度可能舍入；实际准入与累计账本使用整数。

## 普通 WGT 的分步执行接口

这些接口接收已经构造的 forward 模型；需先设置 `compute_budget`。
它们把同一个 forward 分成若干执行片段，保留程序位置、寄存器、工作区和事务内参数。
只有执行到 return 才向原模型提交显式 write；暂停时观察或导出原模型仍是开始前的参数。

| 完整签名 | 参数、返回与限制 |
| --- | --- |
| `execution_start(model,input,max_work,memory_bytes)` | 输入为匹配模型宽度的有限 f32 向量；工作和字节参数为 1..1000000000 整数，分别收紧到 compute 余量和 workspace 上限；内存另不得超过原会话上限。准备独立事务，不执行 forward；返回 null。每次尝试消耗一个能力调用名额，工作额度须大于零，内存至少容纳模型的静态受管区。 |
| `execution_resume(model,instructions)` | instructions 为 1..1000000；最多执行这么多 VM 指令，同时受剩余计算工作量、生命周期工作上限和当前作用域约束。返回 `{done,output,work,memory_bytes,peak_memory_bytes}`。未完成 output=null，完成后是 f32 输出向量。work 为本任务累计工作量，内存字段仅为动态工作区字节。 |
| `execution_cancel(model)` | 取消活动事务并丢弃中间内存和权重写入；有任务返回 true，无任务返回 false。已完成工作不退款。 |

每个模型至多一个活动任务。活动时不能 predict/sample/训练、替换参数或保存训练断点；
可以导出原模型的已提交参数，但不会包含执行状态。完成、取消或执行错误后可重新开始。
错误通过 WGT 运行错误报告，仍可由 `compute_scope` 捕获；执行错误丢弃整个事务，
调用参数错误或开始前的准入拒绝不会隐式取消已有任务。

工作量同时扣除 compute_budget 和脚本操作预算；恢复不会重置任何额度。
当前作用域的内存上限不得小于活动事务的内存上限。
期限在宿主执行检查点和 resume 之前检查；一个片段内的原生计算不被协作式期限抢占。
需要进程级硬限制时使用宿主策略，见[计算预算](compute-budget.md)。

分步事务使用额外的受限会话保存参数快照，原模型会话继续存在；memory_bytes 限制
这份额外会话，不是两者合计的进程上限。当前没有跨进程序列化恢复、线程/CPU 核分配、
forward 内 GPU/NPU 调度或任意形状张量；这不是通用操作系统调度器。
暂停恢复目前仅用于推理；训练一次记录完整实际执行轨迹，不能跨 resume 保留训练 tape。

Rust 对应 `AdaptiveSession::start_execution(input,max_work,memory_bytes)`、
`resume_execution(instructions,available_work)`、`cancel_execution()`；嵌入方负责外层账本。
Rust resume 的 work 是累计值，应只对差值计费。与脚本相同，失败后的 `steps()` 保留已消费量。

## 可运行例子

[working_memory/model.wgt](../examples/native_training/working_memory/model.wgt) 在两个槽位保存 x/-x，
训练地址参数选取正确内容，并用未参与训练的向量比较完整推理与分步执行。
该示例验证可微内存及执行闭环，不证明模型能完成任意任务或自主学习系统调度。

```sh
weightc check examples/native_training/working_memory/model.wgt
wgt run examples/native_training/working_memory/model.wgt
wgt run examples/native_training/working_memory/model.wgt --compiled
```

模型参数初始化与显式训练分开，示例不自动保存制品。
行为检查见 `crates/wgt-runtime/tests/workspace.rs`，包括有限差分、读写与释放、内存上限、
暂停/取消/错误事务、资源观测、训练、模型导出和 Adam 断点恢复。
