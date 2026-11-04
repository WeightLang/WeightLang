# GPU / NPU 设备资源（WGT 0.5，本地实现）

宿主为一次运行授予设备别名、精确设备 ID、并发/队列容量、提交次数和计算额度。
GPU 后端使用现有原生 Vulkan / Metal / DX12 执行器。NPU 通过宿主注册的 Rust 后端接入同一设备池；
当前仓库没有内置厂商 NPU 驱动，没有 NPU 后端时明确拒绝，不回退 CPU。

## 构建与授权

```sh
cargo build -p weightlang-wgt --features gpu --bin wgt --bin weightc
```

历史 `wgt-native --devices` 入口已移除；当前 `wgtime` 服务 CLI 不提供设备枚举命令。

wgt 的 gpu 特性默认关闭；启用后，`wgt run --compiled` 生成的可执行文件也启用 GPU。
执行核心 crate 默认启用 gpu，`--no-default-features` 构建仍可使用宿主注册的原生 NPU 后端。
检查与编译不运行模型、不创建设备任务。明确执行时才读取 WGT_HOST_POLICY；模型制品不携带授权。

示例策略（按本机 `--devices` 输出填写 device）：

```json
{
  "work": 10000,
  "workspace_bytes": 4096,
  "calls": 10,
  "deadline_ms": 30000,
  "devices": {
    "accelerator": {
      "device": "vulkan:0",
      "memory_bytes": 33554432,
      "max_parallel": 2,
      "max_queued": 2,
      "submissions": 2,
      "compute_units": 1000
    }
  }
}
```

`devices` 默认空。每个设备项全部字段必填，未知字段拒绝；device 必须是精确 GPU ID，
或与已注册宿主后端的 ID 一致。禁止同一次运行通过两个别名重复授予同一设备，避免拆分别名绕过额度。
WGT 观测使用 f64，大于 2^53 的额度显示可能失去整数精度，内部账本仍使用整数。max_parallel 为 1..32，max_queued 为 0..1024，submissions 为 0..10000；
memory_bytes 是 usize 且至少为 max_parallel，compute_units 为非负 u64。

每个运行中任务固定获得 floor(memory_bytes/max_parallel) 的受管内存额度。
原生 GPU 后端检查 CPU 会话受管区与 GPU buffer 总和，拒绝超额；不包括驱动内部显存、shader 编译缓存、
排队权重快照、结果对象或操作系统分配。它不是整卡 VRAM 硬隔离，也不是物理显存利用率上限。
需要主机进程内存硬上限时，配合[进程资源策略](compute-budget.md#linux-进程硬限制与-cpu-计量)；
GPU 驱动可能需要较大的虚拟地址空间，不能把小型 CPU 脚本的地址空间上限直接照搬。

## WGT 接口

| 签名 | 参数与结果 |
| --- | --- |
| `device_submit(model,input,alias)` | model 为原生顺序 f32 模型句柄，不支持 adaptive 句柄；input 为与输入宽度相同的有限数值列表；alias 为授权别名。返回 `{ok,job,work,error}`，成功 job 为本设备池整数 ID；准入拒绝时 job=null、work=0。 |
| `device_wait(alias,job,timeout_ms)` | job 为该池提交返回的整数，timeout_ms 为 [0,1000000000]。返回 `{ok,output,error,execution_wall_us}`；output 为 f32 推理结果列表。等待期限同时受当前计算预算与宿主剩余期限限制。 |
| `device_status(alias)` | 返回实际设备 ID、kind（Gpu/Npu）、active、ready、queued、并发/队列上限、受管内存额度、每任务额度、提交计数/余量、计算量/余量。未经授权的别名报错。 |

三项接口参数均必填；提交前必须调用 compute_budget，等待也要求该预算存在。
compute_host_status() 的 devices 字段返回授权别名列表。当前没有独立设备枚举 CLI；注册 NPU 通过授权别名的 device_status 查询。

提交时捕获模型当前权重快照，任务执行期间修改模型不会影响已提交任务；不会隐式训练或保存模型。
当前每个任务独立准备执行会话，不复用模型设备缓存，适合验证调度与授权；小模型的初始化开销可能超过计算时间。
`prepare_exact` 指定单个适配器，不进行自动选择和额外校准推理，不因设备错误偷偷运行 CPU。
后台任务只持有自包含权重；原生 Rust 提交 API 拒绝文件参数，避免设备任务通过模型路径绕过 I/O 授权。

## 队列、计费与失败

每池最多 max_parallel 个工作线程，按入队顺序取任务；完成顺序可以不同。实际硬件并行度由驱动与设备决定。
active 为已经执行的任务数；ready 是已占可运行名额、尚未被线程取出的任务数；
queued 是超过可运行名额的等待任务数，不超过 max_queued。
满队列立即拒绝提交，不隐式等待。池销毁时取消尚未开始的任务，并等待运行中的任务结束。

提交准入同时检查模型工作量、compute_budget 剩余 work、脚本剩余操作数、设备计算额度和提交次数。
通过模型和输入解析、预算存在及截止时间检查并进入准入流程的提交尝试消耗一个计算能力调用名额；
调用名额已耗尽时不能继续扣除。只有成功入队才扣除计算量和设备提交次数。
入队后发生设备失败、排队过期或关闭取消都不退款。

计算额度使用确定性模型计价：linear 为 2×in×out，加上存在 bias 时的 out；
逐元素激活/identity 每元素 1 单位；整次前向至少 1 单位。
这不是实际 FLOP 性能测量、GPU 时钟配额或设备执行毫秒数，不包含设备会话准备成本。
execution_wall_us 为工作线程执行墙钟时间，包含会话准备与同步，不是 GPU timestamp。

设备工作量也计入所在子预算与父预算。任务提交时固定其截止时间，因此子预算退出不能延长已排队任务期限。
设备池内存/并发/提交额度由整个运行共享，不能在子作用域重新授予；子预算的 workspace_bytes 不另行修改设备池显存分配。
任务返回超时后仍可能执行或完成，重复 device_wait 可以取得同一结果；等待超时不代表取消硬件 dispatch。
排队中已经过期的任务不会执行。正在运行的设备任务在完成后检查期限并报告错误；不声称可抢占 GPU/NPU 内核。

池和配额作用于本次运行，不是跨进程整机 GPU 调度器。多个独立进程需要宿主统一分配额度。
运行时不会主动改变 GPU 时钟、电源、驱动全局配置或其他用户的任务。

## NPU 宿主扩展

原生 Rust API 位于 `wgtime_core::accelerator`：

- `AcceleratorBackend`：提供 device()、kind() 和 run(model,input,memory_bytes)。
- `register_backend(Arc<dyn AcceleratorBackend>)`：宿主在创建 WGT runtime 前注册可信原生后端；同 ID 不可覆盖。
- `DevicePool::new(grant,backend)`：也可由嵌入式宿主直接创建并共享设备池。
- `submit(model,input,deadline)`、`wait(job,timeout)`、`status()`：与 WGT 接口共用调度与账本。

NPU 后端需要连接实际厂商运行库、检查其算子/shape 支持、遵守每任务受管内存额度，并在返回前同步计算。
调度器检查返回 shape 与有限数值，后端异常/失败形成任务错误，不转换为 CPU 结果。
注册接口不是 WGT 任意宿主代码执行入口；没有动态加载 Python、shell 或未经宿主注册的库。
独立 CLI 尚不包含厂商 NPU 注册器；单独写入 `npu:0` 不会自动获得 NPU 驱动。

## 示例与验证

可运行示例：[device_queue/model.wgt](../examples/native_training/device_queue/model.wgt)。

```sh
WGT_HOST_POLICY=/absolute/path/policy.json wgt run examples/native_training/device_queue/model.wgt
WGT_HOST_POLICY=/absolute/path/policy.json wgt run examples/native_training/device_queue/model.wgt --compiled
```

本机发现 NVIDIA RTX 4060 Laptop GPU（vulkan:0）与 Intel UHD（vulkan:1）。
实际验证使用 RTX 4060：设备推理与 CPU 参考一致，过小的设备内存额度返回失败；
启用 gpu 特性后，设备队列示例的解释和 AOT 执行均通过，输出误差小于 1e-6。
NPU 测试使用明确标记的模拟原生后端，覆盖注册到 WGT 的链路、并发上限、队列满、等待超时、排队过期与不退款；
不将它视为 NPU 硬件或厂商运行库验收。

测试命令：

```sh
cargo test -p wgtime-core --no-default-features --test accelerator
cargo test -p wgtime-core --test accelerator hardware_gpu_inference_and_memory_quota -- --ignored --nocapture
```

一次本地验证中，105 项无 GPU 特性的运行时测试通过（4 项既有 FUSE 测试按标记忽略）；该数量对应当时的源码快照，
另显式执行并通过真实 GPU 测试。GPU 示例在解释/AOT 模式均返回相同结果，两个任务累计扣费 66 单位；
设备内存测试使用可容纳 CPU 会话但无法容纳额外 GPU buffer 的额度，确认拒绝原因是 combined host/GPU budget exceeded。
