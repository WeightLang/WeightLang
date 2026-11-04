# 计算预算与原生能力（WGT 0.5，本地实验实现）

模型通过普通显式输入观察预算，通过 WGT 控制流选择能力调用。宿主是 runtime 所处的执行环境，包括虚拟机；不访问虚拟机外部的资源。加载或编译模型不调用能力，也不训练。

## 当前接口

forward 现可用 `resource_state()` 直接观察自身工作量与工作区，使用可微内存读写；
普通函数的 execution_start/resume/cancel 支持分步执行，resume 的实际工作量扣除
本预算与脚本预算，开始任务消耗一个调用名额。完整契约见[工作内存与分步执行](working-memory.md)。

所有参数必填；解释执行和生成的 Rust 可执行文件共用实现。

| 签名 | 契约 |
| --- | --- |
| `compute_budget(work,workspace_bytes,calls,deadline_ms)` | 每次运行只能设置一次。四个参数均为 [0,1000000000] 整数，calls 最大 10000。work 受当前解释器剩余操作额度约束；deadline 从此调用起计时。返回 null。 |
| `compute_status()` | 返回 work_remaining、workspace_bytes、calls_remaining、deadline_remaining_ms 对象。 |
| `compute_capabilities()` | 返回可用的版本化能力名列表：三个计算能力，以及宿主授权的 I/O 能力。不需要初始化预算。 |
| `compute_call(name,input)` | input 为列表；返回 `{ok,error,value,work}`。需要先设置预算。 |
| `compute_trace()` | 返回本次运行的调用摘要列表：capability、input_length、work、ok、error。不记录输入内容。 |

`integer_sum.v1` 接受整数列表，返回精确整数和；每个输入和每个中间和必须在 ±9007199254740991 内，空列表为 0。`sort.v1` 接受有限数值列表，返回升序列表。`find.v1` 接受 `[目标,候选0,候选1,...]`，返回首个匹配候选的零基索引，无匹配返回 null；空输入无效。查找使用 WGT 按值相等语义。

设输入元素数为 n：求和、查找扣除 max(1,n) 工作单位；排序扣除 max(1,n×整数位数(max(1,n)))。这是一致的保守工作量计价，不是精确 CPU 指令或时间。计价同时扣除脚本剩余操作预算。失败的输入验证也扣费；执行前拒绝不扣工作量，但消耗调用次数。调用次数耗尽后不再追加轨迹。

每个调用按 n×32 字节检查工作区额度。该额度仅是能力输入规模的准入估算，不包含脚本值、嵌套值、输出 JSON、轨迹或进程其他内存，不能作为内存硬限制。截止时间在能力执行前后检查，不强制抢占运行中的函数，也不限制其他脚本操作。

能力失败以 error 返回：unknown_capability、deadline_exceeded、workspace_exceeded、work_exhausted、invalid_input、calls_exhausted。错误参数类型、缺少预算、重复设置预算属于脚本错误，直接终止。成功时 error 为空字符串。三个计算能力均无外部副作用；宿主 I/O 能力另见下文。

## 示例与训练

可运行示例：[compute_policy/model.wgt](../examples/native_training/compute_policy/model.wgt)。该示例显式训练一个两动作策略，然后根据策略输出选择常量回答或求和程序。它验证接口与训练连通性，不证明复杂任务泛化、预算自适应或强化学习效果。训练本身仍受原有训练运行约束，未纳入能力工作量计价。

```wgt
wgt 0.5
fn main() {
    # 工作量与调用额度属于整个任务，不在每次调用时重置。
    compute_budget(1000, 4096, 4, 10000);
    let result = compute_call("integer_sum.v1", [10, 20, 30]);
    assert result["ok"], "host computation failed";
    return result["value"];
}
```

## 尚未实现的边界

本版是确定性计算闭环的第一步，不是完整宿主资源系统。尚无通用资源句柄、能力制品依赖协议和通用神经网络对照基准。子预算、自定义 WGT 能力和成本感知强化学习已在下文实现；进程级硬取消只支持 Linux 受限 worker。启用宿主策略后会禁止既有文件/数据集接口绕过能力授权；没有策略时仍保持原有行为。无论哪种模式，都不能据此把不可信脚本当作已隔离程序。Linux 已提供独立进程 CPU/地址空间限制；仍需训练数据流和模型保存的授权实现。

## 宿主策略与只读 I/O

宿主设置 `WGT_HOST_POLICY=/absolute/path/policy.json`，解释执行与独立 AOT 可执行文件均在执行开始时读取它；检查、编译和模型制品不携带授权。未设置时兼容原有执行方式，文件和网络能力均不可用。策略字段拼错或文件无效会拒绝启动。

```json
{
  "work": 10000,
  "workspace_bytes": 4096,
  "calls": 10,
  "deadline_ms": 10000,
  "disk_read_bytes": 4096,
  "network_body_bytes": 4096,
  "network_requests": 2,
  "files": {"data": "/absolute/path/input.txt"},
  "endpoints": {"info": "https://example.com/"}
}
```

前四项必填；三个 I/O 额度默认 0，两张授权表默认空。work、workspace_bytes、deadline_ms 最大 1000000000，calls 最大 10000；I/O 额度为非负 u64。脚本的 `compute_budget` 请求只会收紧这些上限，不能扩权。宿主截止时间在解析并验证策略后、初始化文件和设备资源前固定为绝对时刻，权限检查、HTTP 超时与资源状态共用该时刻；资源初始化不能延长授权。子预算只收紧截止时间，失败退出也恢复原来的父级时刻，不重新起算。HTTP 剩余时间不足 1 ms 时在发出请求前拒绝，不能将零超时作为无限等待。该期限在解释/AOT 操作检查点检查；超时终止脚本。它仍不能抢占长时间运行的原生函数。没有宿主策略时保留原先仅检查能力调用的截止时间行为。

`compute_host_status()` 无参数，无策略时返回 null。有策略时返回 `disk_read_bytes`、`disk_read_remaining`、`disk_written_bytes`、`disk_write_remaining`、`disk_storage_remaining`、`network_body_bytes`、`network_body_remaining`、`network_requests`、`network_requests_remaining`、`deadline_remaining_ms`、`cpu_time_us`、`cpu_limit_seconds`、`address_space_limit_bytes`、`files`、`outputs`、`endpoints`、`devices`。四个复数名称字段是授权别名列表，不暴露真实路径或 URL；CPU 时间读取失败时 `cpu_time_us` 为 null，未设置 CPU 时间或地址空间限制时对应限制字段为 null。其他计量字段为数值；WGT 使用 f64，大于 2^53 的显示值可能丢失整数精度，内部额度仍以 u64 计量。

新增能力（均通过 `compute_call(name,input)` 调用）：

| 能力 | input | value |
| --- | --- | --- |
| `file_read.v1` | `[文件别名,最大字节数]` | 从文件开头读取的 UTF-8 文本前缀 |
| `http_get.v1` | `[端点别名,最大字节数]` | `{status,body}`，状态码与 UTF-8 响应正文前缀 |

最大字节数为 1..1000000000，不能超过工作区额度或剩余读取额度；原来的列表准入估算也适用。只读取前缀，不保证完整结果、不额外读取一个字节判断 EOF；UTF-8 截断到字符中间会返回 invalid_utf8。调用者应通过数据格式判断完整性。文件必须为绝对路径的普通文件，在上下文初始化时打开并持有句柄，调用时不重新解析路径，每次读取从头开始。

HTTP 端点由宿主提供完整固定 URL，脚本只能选择别名，不能修改路径、参数或请求头。仅支持 http/https，不使用环境代理、不跟随重定向、不附加环境凭据。HTTP 非 2xx 状态也作为结果返回。请求超时受剩余截止时间限制。network_body_bytes 仅计交给读取器的正文数据，不含 TLS、头部、DNS 或网络栈预读，不能当成网卡流量硬限制。

读取按成功取得的字节块累计，即使随后读取或 UTF-8 解码失败也不会退款；请求发出前计入次数，连接失败也不退款。新增错误码：permission_denied、disk_read_exhausted、network_requests_exhausted、network_body_exhausted、network_error、io_error、invalid_utf8。失败也消耗能力调用次数与已准入的工作量。轨迹不包含 URL、路径、正文或凭据。

启用宿主策略后，包依赖解析及脚本直接 `save`、`load`、jsonl 与 hf_dataset 等接口会明确拒绝执行，需使用新的授权能力；没有策略时按当前执行契约运行。当前只读能力不能替代模型保存或训练数据流的全部功能。能力发现仅在授权表非空时列出对应 I/O 能力，发现不保证额度足够。

运行示例：[host_resources/model.wgt](../examples/native_training/host_resources/model.wgt)。将策略中的 data 指向一个已有文本文件，再运行：

```sh
WGT_HOST_POLICY=/absolute/path/policy.json wgt run examples/native_training/host_resources/model.wgt
WGT_HOST_POLICY=/absolute/path/policy.json wgt run examples/native_training/host_resources/model.wgt --compiled
```

磁盘写入和 Linux 进程上限见下文；当前不提供完整安全沙箱或恶意宿主文件系统并发修改防护。文件打开前的元数据检查不是操作系统沙箱。这套策略用于控制能力执行权限与累计消费，不能作为执行任意不可信代码的安全边界。

## Linux 进程硬限制与 CPU 计量

宿主策略可额外设置 `cpu_seconds`（1..1000000000 的整数秒）和
`address_space_bytes`（正整数，最大 i64::MAX）。未设置时不改变原有执行模式。
设置任一项时，`wgt run` 和生成的 AOT 可执行文件会启动独立工作进程。Rust 嵌入接口不因此自动切换到工作进程，硬限制行为见下文。
Linux 内核在执行前设置 CPU 的 RLIMIT_CPU 和地址空间的 RLIMIT_AS，软硬上限相同；
超过 CPU 时间会杀死工作进程，地址空间分配失败可能导致工作进程退出，父进程报告失败。
父进程还按 deadline_ms 监控并终止超时工作进程。检查与编译不会设置限制或运行训练。

CPU 时间包含整个工作进程的用户态与内核态时间及所有线程，不是墙钟时间，也不是单次能力耗时。
`compute_host_status()` 新增 cpu_time_us、cpu_limit_seconds、address_space_limit_bytes；
前者是整个进程自启动以来的 CPU 微秒数，Linux 之外目前返回 null。
CPU 限额按整秒设置；地址空间限制针对虚拟地址空间，包含代码、共享库、堆栈及映射，不是 RSS 限额。
设限低于程序启动所需内存会在启动阶段失败。硬限制只支持 Linux，其他系统显式拒绝请求。

Rust 嵌入接口不修改调用者进程的资源上限；若请求硬限制而当前进程没有相应或更严格的内核硬上限，
则拒绝执行。嵌入方需在独立 worker 中预先设限。资源隔离不等于文件系统/系统调用安全沙箱。

## 磁盘写入

Linux 宿主策略可增加 `outputs`（输出别名到绝对文件路径）、`disk_write_bytes`（累计写入额度）、
`disk_storage_bytes`（本任务新建文件内容额度），默认分别为空、0、0。
输出父目录必须已存在，执行开始时持有该目录句柄；文件只在显式调用时创建。

`compute_call("file_write.v1", [输出别名,UTF-8字符串])` 新建文件并同步内容，返回
`{bytes_written,created:true}`。不允许覆盖已有文件或符号链接；不会创建父目录、删除旧文件或隐式训练。
写入前同时检查工作区、累计写入和存储额度；每个成功写入的字节都计费，失败不退款。
此版本无删除操作，因此任务存储消费等于累计成功写入字节，不因外部删除文件退款。
新字段 disk_written_bytes、disk_write_remaining、disk_storage_remaining、outputs
可通过 compute_host_status() 查询，outputs 只显示别名。

新增错误：already_exists、disk_write_exhausted、disk_storage_exhausted、write_partial。
write_partial 表示文件已经创建，可能为空或部分写入，不能自动重试或声称回滚。
文件操作完成后即使超过协作式截止时间，也保留其真实结果，不把已发生写入伪装成未执行；
进程被硬终止时仍可能留下不完整文件，父进程失败结果不代表外部副作用回滚。
仅内容字节受此额度约束，不包含文件系统元数据/块大小开销。

可运行写入示例：[host_write/model.wgt](../examples/native_training/host_write/model.wgt)。宿主需授予 result 输出别名和至少 23 字节写入/存储额度。

## 子预算与自定义能力

`compute_scope(function,args,limits)` 在子预算内调用本编译单元的 WGT 函数；function 是函数名字符串，args 是按位置展开的实参列表。
返回 `{ok,error,value,work}`，work 是该子任务实际消耗的解释/AOT 操作数，包含嵌套能力工作量。
函数必须已经存在，不能指向内置函数；函数名、args 或 limits 的类型错误及不存在的目标函数直接报脚本错误。limits 必须为九项整数列表；列表内容或额度无效时返回失败结果：

```text
[work, workspace_bytes, calls, deadline_ms,
 disk_read_bytes, disk_write_bytes, disk_storage_bytes,
 network_body_bytes, network_requests]
```

每项 [0,1000000000]；额度自动取请求与父级余量的较小值。进入作用域消耗父级一个调用名额，
子作用域内的调用额外计入父级。最多嵌套 32 层。工作额度同时限制普通 WGT 控制流及原生能力计费。
I/O 额度不能新增授权，只收紧已有授权的消费上限；0 表示不分配该项资源。
退出时恢复父级额度上限，保留已经消耗的工作、调用和 I/O 字节。失败也按实际消费结算，不能通过反复建子预算重置额度。

子任务的普通运行错误、预算耗尽或返回非 JSON 句柄被封装为失败结果；父任务可继续。
父级全局预算耗尽、宿主截止时间到达或操作系统硬限制仍可终止整个任务。子预算不是事务，
不会回滚模型参数、文件副作用或指标。CPU 和地址空间硬上限由整个工作进程共享，不在进入子作用域时反复修改内核限制。
workspace_bytes 保留原生能力工作区准入估算语义，不是子函数堆内存硬限制。
轨迹中 scope 的 work 是 inclusive（包含子调用），不能再与其子调用记录相加作为总消费。

`compute_register(name,function,limits)` 将单参数 WGT 函数注册为当前运行内的能力，返回 null。
name 必须以 `user.` 开头，只含 ASCII 字母数字、点、下划线、连字符，最长 64 字节；建议显式以 .v1 等标示版本。
最多注册 64 项，同名不能覆盖，且只能在根作用域注册。function 是函数名字符串，目标函数必须存在并有且仅有一个参数；
limits 契约同 compute_scope。`compute_call(name,input)` 把整个 input 列表作为该函数唯一参数，
在注册的子预算内执行并返回统一结果。注册不执行函数；能力不包含 shell、Python 或动态库入口。

自定义能力不增加宿主权限；它与原生能力共享预算和资源策略。返回值需能序列化为 JSON，
不允许把模型/数据流等句柄作为能力结果跨边界返回。支持解释与 AOT，能力名会出现在 compute_capabilities()。
示例：[custom_budget/model.wgt](../examples/native_training/custom_budget/model.wgt)。

## 成本感知强化学习

[cost_aware_policy/model.wgt](../examples/native_training/cost_aware_policy/model.wgt) 使用原生 sample/reinforce
训练两动作策略，没有调用标签或 Python。观测为 `[需要精确计算,单位工作量价格]`，
奖励为 `结果正确得分 - 价格 × compute_call 返回的实际 work`，固定基线 0.2。
单步回合相当于上下文 bandit；不可微求和程序不参与反向传播，策略通过采样动作的策略梯度更新。

训练数值范围有限，评估使用不同范围的整数，精确结果由程序保证。
同时执行始终调用基线，并评估直接回答 0 的不调用基线；后者不是训练后的通用神经网络对照。
低价格时，受控评估的策略与始终调用基线均达到 100% 成功，平均工作量分别为 1 与 2；
不调用基线成功率为 50%。高价格时最优奖励动作可能是不调用而放弃正确性，
所以报告明确区分奖励最优动作比例、任务成功率和成本，不能把“省成本”当成准确率提高。
自动测试覆盖三个固定采样种子；这验证原生训练与计费闭环，不证明广泛任务泛化或最优调度。

## 验证记录

本地 Linux 原生测试通过；这些结果不代表已发布或其他操作系统已支持硬限制。

| 能力 | 验证证据 |
| --- | --- |
| CPU 计量与硬限制 | isolation 测试读取 CPU 时间；持续原生计算在 1 秒 CPU 限额被内核终止 |
| 进程内存限制 | 32 MiB 地址空间限额导致 worker 分配失败，父测试进程继续运行 |
| 磁盘写入 | 实际新建、重复写拒绝、写入/存储额度检查；AOT 写入 23 字节且计费一致 |
| 子预算 | 嵌套工作耗尽后父任务继续；实际文件 I/O 在子额度耗尽时拒绝、退出后只恢复上限 |
| 自定义能力 | WGT 求平方和返回 25；解释/AOT 均计费 21 工作单位，失败子任务计费 10 |
| 成本感知强化学习 | 三个固定采样种子的策略梯度测试；解释/AOT 评估指标一致，低价格同成功率下平均工作量 2 → 1 |

可重现检查：`cargo test -p wgtime-core --no-default-features`，
然后分别对 custom_budget/model.wgt 和 cost_aware_policy/model.wgt 执行 `wgt run` 与 `wgt run --compiled`。
本次使用 `WGT_WEIGHTFS=off cargo test --locked --offline -p wgtime-core --no-default-features`，
一次本地源码快照的运行时测试中，105 项通过，4 项 WeightFS/FUSE 实机测试按其原有标记忽略；资源相关测试均执行。测试数量会随源码变化。
随后单独使用 `WGT_WEIGHTFS=on cargo test --locked --offline -p wgtime-core --no-default-features --test weightfs -- --ignored`，
4 项 WeightFS/FUSE 实机测试通过。
共享构建产物存在并行操作时，应使用独立磁盘 CARGO_TARGET_DIR 复核。

## GPU / NPU 设备池

宿主 devices 授权现在可提供设备分配、并发/排队容量、提交次数与模型工作量额度。
WGT 使用 device_submit、device_wait、device_status；设备计算同时计入 compute_budget 与子预算。
GPU 使用原生硬件执行；NPU 需要宿主注册原生厂商后端，未注册时不回退 CPU。
完整签名、计价与显存/跨进程边界见[设备资源](accelerator-resources.md)。
