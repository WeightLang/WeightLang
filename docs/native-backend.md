# WGT 推理运行时架构（目标设计，尚未实现）

目标运行时命名为 `wgtime`，职责是加载权重与计算结构、接收显式张量输入并执行推理。
它将以可复用 Rust 库实现，由机器级唯一 `wgtime` 服务进程宿主；多用户共享同一服务，进程并发启动必须互斥并连接已运行服务。
WGT 编译程序只链接轻量 IPC 客户端，不嵌入推理核心、不独立加载权重、不创建 GPU context。连接失败必须明确报错，不得静默回退到进程内执行。

下述 NativeSession、IR 与资源管理章节描述当前本地实现能力，不代表上述单服务架构已落地。服务 IPC 鉴权、跨用户隔离、模型实例复用键、可变会话状态隔离、资源预算和崩溃清理仍待设计与验证。训练/优化器及 harness 的调用路径和状态归属也待确定；模型加载继续与显式训练分离，不隐式训练。
当前本地检查/原生 CLI 命令仍是 `weightc check FILE.wgt` 与 `wgt run FILE.wgt`；尚无公开 `wgtime` 进程入口，亦未完成后续去重 `wgt-native` 的合并。

```sh
# 当前本地实现的诊断入口，不代表目标 wgtime service 已实现：
cargo test -p weightlang-runtime
```

## 原生 IR v1

顶层字段：`version=1`、正整数 `input`、`parameters`、非空 `layers`。
每个参数有唯一 `name`、非空正整数 `shape` 和下列一种定义；逻辑数值为 f32：

| kind | 字段 | 契约 |
|---|---|---|
| resident | data | 扁平有限数值，长度等于 shape 元素数；会话拥有数据 |
| file | path, offset | offset 为 4 字节对齐的字节偏移；文件保存 little-endian f32 |
| resident_file | path, offset | 对齐的 little-endian f32 文件范围，准备阶段一次加载至常驻存储 |
| packed | storage, group, data, scales | storage 为 int4/int8/f16/bf16，group 为 1..4096；data 是编码字节，scales 是 F32 列表（浮点存储为空）；仅二维矩阵 |
| packed_file | storage, group, path, offset, scales_offset | 准备阶段按 shape 推导编码/scale 字节数，从文件加载紧凑常驻权重，偏移单位字节 |
| view | source, start | 引用先前参数的连续扁平区间；start 是元素偏移，区间长度由本参数 shape 决定 |
| add | base, delta | 两者都引用先前参数且完整 shape 相同；结果是表达式，不是独立参数副本 |

引用必须指向此前声明，因此自引用、循环和前向引用在 prepare 被拒绝。
表达式深度上限 32。view 可以改变形状，但只支持连续扁平片段，不支持任意步长切片。
add 不支持广播。参数加法和 view 是 native IR 能力，**尚未增加 WGT 的 delta/bind 语法**。
WGT 0.5 模型构造器当前不支持 share 声明。

linear 节点字段为 `op=linear, weight=NAME, bias=NAME|null`；bias 可以省略，表示无偏置。
权重必须为 `[out,current_width]`，bias 必须为 `[out]`。
其余节点只有 op。严格拒绝未知算子和字段。
完整的模型及断点保存流程见 [checkpoint/model.wgt](../examples/native_training/checkpoint/model.wgt)；须显式执行脚本后才会创建 model.json。

## 内存与生命周期契约

`Session::prepare(model, root, budget)` 检查契约、文件长度和预算，生成顺序执行计划，
打开参数文件并分配两个激活缓冲区和一个参数工作区。常驻参数独立保存；view/add
没有永久数据副本。文件参数每次用到时同步读取，复用工作区，不把整个文件读进堆。
组合权重临时物化在工作区中，计算结束后该空间可复用于下一层。

预算单位为字节，公式为：

```text
managed_bytes = resident_bytes + activation_bytes + workspace_bytes
activation_bytes = 2 * 最大层宽度 * sizeof(f32)
workspace_bytes = 最大单层所需工作区；直接 packed linear 只需 bias 元素数 * 4，其他为 (weight + bias) 元素数 * 4
```

预算是受管理张量存储上限，**不是进程 RSS 上限**：不包含解析期间的 JSON、IR 元数据、
OS 文件页缓存、进程/管道开销，以及 I/O 的 4 KiB 栈缓冲区。大模型控制文件应使用 file
源，避免在 JSON 中重复嵌入大量参数。会话不承诺锁定物理内存或没有缺页。
文件必须在会话期间保持不变；当前没有文件摘要锁定或磁盘写回。常驻参数的显式读写与会话 revision 见下节。
prepare 检查长度；执行时检测短读和文件中非有限值，不能静态保证磁盘永远可读。

`Session::run(&mut self, input: &[f32]) -> Result<&[f32], &'static str>` 的成功执行
不进行堆分配，也不依赖垃圾回收。结果借用会话存储；Rust 借用规则阻止仍使用旧结果时
再次可变借用会话。drop 确定性释放权重、缓冲区和文件。
测试通过全局分配计数器验证内存和文件参数路径，并非仅观察 RSS 或运行一次 warmup。

## GPU 存储和能力边界

CPU 的预算公式保持不变。GPU 总预算为 `host_managed_bytes + gpu_buffer_bytes`，
后者包含不同 weight/bias 组合的常驻有效权重、文件表达式共享暂存区（至少 4 字节）、
两个激活缓冲、回读缓冲和每算子 32 字节控制参数。纯常驻 view/add 上传一次，
显式修改参数后刷新缓存；依赖文件的表达式每次读取并上传，提交顺序防止暂存覆盖。
GPU 分配失败、超出设备 buffer/dispatch 限制均拒绝候选。

核显/统一内存设备可执行，但当前仍创建独立主机和 GPU 缓冲，预算分别计费；
设备 memory_model 只是报告信息，不代表已实现共享物理分配或零拷贝。
预算不包含驱动、编译、校准输入、上传暂存和系统开销，不能作为硬性 RSS/显存峰值上限。

本文描述的 NativeSession 推理入口不提供自动求导、优化器、KV cache、通用图、GPU batch 或异步文件预取。
CPU 常驻 linear 使用 AVX2/FMA（不支持时标量回退），直接读取常驻权重免去逐次矩阵拷贝。
GPU linear 使用工作组并行归约，文件 I/O 尚未与计算重叠。
不能把此阶段描述为完整 PyTorch 替代。所有能力为本地源码实现，尚未发布。

验证原生运行时：`cargo test -p weightlang-runtime`。GPU 能力须在目标设备单独验证。
