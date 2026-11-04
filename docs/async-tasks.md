# 原生异步任务（本地第一阶段）

WGT 0.5 在普通脚本中提供 `async fn`、`await`、`tasks { ... }` 和 `spawn`。
这是本地实现，尚未发布。解释执行与 AOT 使用同一任务调度和 I/O 核心；AOT 生成 Rust 原生 future，运行时不回放 WGT 或 AST。
完整例子见 [async_tasks/model.wgt](../examples/native_training/async_tasks/model.wgt)。

## 调用与作用域

`async fn name(args) { ... }` 参数按位置传递。调用立即求值实参并创建冷 future，函数体到 `await` 驱动时才执行。
普通 `fn` 保持同步，不因被异步函数调用就变成异步。参数和返回值必须递归为有限数字、bool、字符串、null、列表或对象；
第一阶段拒绝模型、Reader、Kernel、梯度、RecordBatch、future、task 作为异步函数的参数或返回值。
无闭包、隐式变量捕获或可发送资源支持。

`spawn FUTURE` 仅在同步函数的 `tasks` 块内可用，消耗冷 future，返回只属于该作用域的 task。
`await FUTURE` 驱动冷 future，`await TASK` 等待作用域任务；任意同步值不能加 await。
操作数只求值一次，同一个 future 只能启动一次、task 结果只能取一次。
await 一个 task 只等待该目标，同时继续轮询其他已提交子任务；无关 pending 任务不会使目标成功结果延迟到全体结束。作用域退出才 join 全部。已观察到的任一任务失败按作用域失败契约立即取消兄弟任务。
spawn 后子 future 由下一次 await 或作用域退出时的 executor 轮询推进，父同步代码本身不被抢占。

`tasks { BODY }` 正常退出 join 全部子任务，包括没有显式 await 的任务；return/break/continue 也先 join。
错误退出取消 pending future，并等待已提交阻塞 I/O 真正结束后返回；future 的局部释放逻辑在取消时运行。
任一任务失败会取消兄弟任务，保留任务来源及所有已观察到的错误；消费任务结果不会使作用域失败消失。
无默认脱离任务。第一阶段不支持嵌套 tasks，也不支持 async fn 内 tasks/spawn；嵌套 async 函数的 await 可用。

## 限制、预算和随机数

每个作用域最多保留 64 个任务，每个 Runtime 最多创建 10000 个冷 future；容量满明确报背压错误，不无限排队。
固定阻塞池为每个父 Runtime 2 个工作线程、64 个等待作业，不按任务新建线程。
所有子任务共享父操作预算，spawn、脚本 tick 和 I/O 提交/执行都计费，不重置子 Runtime 的预算。pending 轮询仅检查取消和截止时间，不因 OS 完成快慢反复扣工作量。
作用域截止时间为 30 秒；合作取消在 tick、轮询和 I/O 检查点生效，不能强行打断任意 OS 调用。
队列、任务和 I/O 返回大小有界，不表示已有通用脚本堆内存计量器。

任务 RNG 根据父 seed 与 spawn 序号确定性拆分，不抢共享 RNG；相同父 seed/提交顺序产生相同任务流。
嵌套 await 在当前任务 RNG 中顺序运行。普通同步函数原有 seed/random 行为保留。
第一阶段显式拒绝活跃 host policy 或 compute_budget 下创建 future/进入 tasks，以免绕过权限、截止时间或累计预算；
没有子任务重新读取环境策略。后续能力接入需显式预留和归还账本，当前不承诺支持。

async fn 调用的同步 helper 也处于异步限制下；只支持普通数据控制流和显式允许的标量/列表内置，
拒绝模型构造、训练、设备、Reader、同步 I/O 等资源入口。同步 helper 内 await 也拒绝，须声明 async fn。
forward/loss 不支持异步，不能用 `await net(x)` 表示模型异步执行。
CPU/GPU 模型队列、只读权重快照、训练 revision 提交及 reload/unload 与活跃任务协调均后置。

## 第一阶段 I/O

`await sleep(milliseconds)` 仅在 async fn 内可用，参数为 0..30000 的整数毫秒，返回 null；
由固定阻塞池执行可取消的等待，任务 executor 不被 sleep 阻塞。尚无原生定时器 reactor。
`await load text from PATH` 仅在 async fn 内异步读取文本，路径相对脚本根目录，返回 UTF-8 字符串。
实际读操作在固定阻塞池中执行，复用普通文件/大小检查；非普通文件、无效 UTF-8 或超限明确报错。
字节费用计入同一操作预算：先在固定池打开并检查同一文件描述符，再按 `文件大小 + 1` 原子预留共享预算，单文件最多 64 MiB；额外一字节用于增长探测。并发任务竞争同一余额，不把预算人为平分。
实际读取始终使用已打开的描述符，限制为预留长度，路径替换不会切换到新文件。读取超过原 metadata 大小时拒绝，探测到的额外字节也计费；这不承诺并发写入时的内容快照。
成功或部分读取失败均按实际追加到缓冲的字节保留费用，未用预留在工作线程结束时恰一次归还；UTF-8 解析失败仍保留已读费用。
队列拒绝或启动前取消归还全部未用预留；打开/权限/普通文件/大小检查失败发生在字节预留之前。
spawn 及 I/O 提交费用不退款。取消也等待工作线程完成计账与资源清理。64 MiB 是文件上限，本次可用读取额度还受共享剩余预算限制。
这是有界阻塞 I/O 卸载，不承诺磁盘操作本身非阻塞。JSON、HTTP、写入及 capability I/O 异步化尚未接入。
外部 I/O 已产生的效果不能因 future 取消而回滚。

## 资源释放与 GC

当前 Value 的列表/对象按值持有内容；模型等是不透明索引，由 Runtime 的资源容器持有。
变量解绑不等于模型卸载，不能把 Rust 的所有权释放称为追踪式 GC。
任务 future 在完成或取消时释放捕获数据；阻塞作业及其缓冲在真实完成后释放，scope 退出等待在途作业清理。
这里没有实现可收集任意循环对象图的追踪式垃圾回收器，也没有实现 GPU 任务生命周期协议。
