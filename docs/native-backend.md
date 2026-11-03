# Rust 原生执行后端（本地实验版本）

`crates/wgt-runtime` 提供独立 Rust CPU/GPU f32 推理核心与 `wgt-native`。
通过 `weightlang.native.NativeSession` 显式使用，不替换默认 Torch 模块加载。
Python 控制会话，Rust 执行前向；设备选择支持 auto/cpu/gpu，严格模式仅 CPU。
原生参数更新接口支持自定义算法，不内置自动求导或优化器。

## 构建和运行

```sh
cargo build -p weightlang-runtime --release
# 静态检查、预算验证和会话准备；不执行前向。
target/release/wgt-native examples/native_backend/model.json 64 --device cpu --check
# 第一行输出内存报告，随后每个输入向量对应一行输出。
echo '[0.1,0.2]' | target/release/wgt-native examples/native_backend/model.json 64
WEIGHTLANG_NATIVE=target/release/wgt-native PYTHONPATH=src python examples/native_backend/run.py
```

Rust CLI 和执行核心无需 Python、Torch。Python 包目前仍有 Torch 安装依赖；
不能将原生核心的独立性说成 Python 包已完全脱离 Torch。
示例是显式给定参数的结构演示，不是训练结果。

## Python 接口

`weightlang.native.lower(source, parameters, *, config=None) -> dict`
解析当前 WGT 源码，生成 native IR v1。仅接受单输入、单输出、静态 f32 向量的
顺序链（包括可展开的 seq）；支持 linear、relu、tanh、sigmoid、identity。linear 使用
`[out,in]` 权重，bias 默认为有，`bias=0` 时不需要偏置。
明确引用前一层的 `from` 与隐式顺序等价；分支、非顺序 block、循环、额外结果、其他算子、
多端口、动态维度、权重类型与 adapter 均拒绝，不会丢弃语义后继续执行。

`parameters` 是保持插入顺序的映射：参数路径映射到扁平数值列表，或者下节的
存储/表达式字典（不含 name；层参数可以省略 shape）。辅助参数必须给 shape。
每个所需参数都必须提供值；已有 `parameter` / `share` 声明转成共享视图，
共享目的参数不可再单独赋值。此入口不执行源码 init、import、derive 或训练；
有 import/derive 的源码明确拒绝。所有原生参数当前仅用于推理，freeze 不引入训练。
返回的 IR 仍须经过 Rust prepare 的引用、形状、文件和预算验证才可执行。

`NativeSession(model, *, budget_bytes, executable=None, device="auto", strict_zero_alloc=False)`
读取已保存的 native IR JSON。budget_bytes 必填，是非负整数；设备选择规则见下节。
executable 依次取显式值、`WEIGHTLANG_NATIVE`、PATH 中的 `wgt-native`、源码目录的 `target/release/wgt-native`、`target/debug/wgt-native`，未找到即报错，不隐式编译。
模型文件的父目录是相对参数文件路径的基准。

`NativeSession.from_source(source, parameters, *, budget_bytes, executable=None, config=None, device="auto", strict_zero_alloc=False)`
组合 lower 与准备阶段，不访问训练数据，不保存模型制品。控制文件临时放在
`/tmp/weightlang/`，准备完成后删除。参数文件路径必须绝对；控制 JSON 上限 8 MiB，
大权重应使用磁盘文件。已有 Torch 参数可显式用 `detach().flatten().tolist()`
传入用于小模型对照，但原生前向不调用 Torch。

`session.report` 返回预算报告；`session(values)` 接受一个扁平输入向量并返回列表。
使用 `with` 或显式 `close()` 释放子进程、管道和原生会话。
错误输入或文件 I/O 错误可能终止 CLI 会话；调用方需新建会话。
JSON 通信、Python 对象和准备阶段会分配内存，不属于零分配保证。

## 设备选择

默认 GPU feature 经 wgpu 提供 Vulkan、Metal、DX12；纯 CPU 构建使用
`cargo build -p weightlang-runtime --release --no-default-features`。
`wgt-native --devices` 或 `available_devices(*, executable=None)` 返回设备清单。
本机验证了 Linux Vulkan 下 NVIDIA 独显、Intel 核显；Metal/DX12 路径尚未在对应机器验收。
没有直接 CUDA、ROCm、MPS 或 NPU 驱动接口。可选 device：

- `auto`：在 CPU 和可用 GPU 间按当前模型实测选择；失败候选记录原因，允许选 CPU。
- `cpu`：直接使用 CPU，不进行校准。
- `gpu`：实测选择可用 GPU；没有符合条件的设备则报错。
- `vulkan:N`、`metal:N`、`dx12:N`：选择清单中的设备；失败报错，不回退 CPU。

候选须通过设备限制、预算、两个确定性非零输入与 CPU 对照的正确性检查
（绝对容差 1e-4，相对容差 1e-3），再预热两次、计时五次取中位数。
计时包含主机输入、上传、计算、同步回读以及需要的文件读取，不包含准备/编译时间。
每次只准备一个 GPU 候选。报告包含候选耗时、拒绝原因和实际设备。
这是固定形状、合成输入、当前机器负载下的选择，不是对所有负载的全局最优保证；
每次新建会话会校准，随后复用选定设备。`auto --check` 同样会执行校准；
只需准备检查且不执行前向时使用 `--device cpu --check`。

`strict_zero_alloc=True` / `--strict-zero-alloc` 将 auto 限定为 CPU；与显式 GPU 冲突时报错。
保证仅覆盖 Rust 成功执行热路径，不覆盖 Python、JSON、准备、GPU 驱动/命令分配。
Rust 不依赖 GC 不等于整个应用零分配。

## 原生 IR v1

顶层字段：`version=1`、正整数 `input`、`parameters`、非空 `layers`。
每个参数有唯一 `name`、非空正整数 `shape` 和下列一种定义，所有数值均为 f32：

| kind | 字段 | 契约 |
|---|---|---|
| resident | data | 扁平有限数值，长度等于 shape 元素数；会话拥有数据 |
| file | path, offset | offset 为 4 字节对齐的字节偏移；文件保存 little-endian f32 |
| view | source, start | 引用先前参数的连续扁平区间；start 是元素偏移，区间长度由本参数 shape 决定 |
| add | base, delta | 两者都引用先前参数且完整 shape 相同；结果是表达式，不是独立参数副本 |

引用必须指向此前声明，因此自引用、循环和前向引用在 prepare 被拒绝。
表达式深度上限 32。view 可以改变形状，但只支持连续扁平片段，不支持任意步长切片。
add 不支持广播。参数加法和 view 是 native IR 能力，**尚未增加 WGT 的 delta/bind 语法**。
WGT 中既有 share 在原生执行中保留同源关系，并且不会重复计算常驻存储预算。

linear 节点字段为 `op=linear, weight=NAME, bias=NAME|null`；bias 可以省略，表示无偏置。
权重必须为 `[out,current_width]`，bias 必须为 `[out]`。
其余节点只有 op。严格拒绝未知算子和字段。
完整例子见 [model.json](../examples/native_backend/model.json) 和
[WGT 模型](../examples/native_backend/model.wgt)。

## 内存与生命周期契约

`Session::prepare(model, root, budget)` 检查契约、文件长度和预算，生成顺序执行计划，
打开参数文件并分配两个激活缓冲区和一个参数工作区。常驻参数独立保存；view/add
没有永久数据副本。文件参数每次用到时同步读取，复用工作区，不把整个文件读进堆。
组合权重临时物化在工作区中，计算结束后该空间可复用于下一层。

预算单位为字节，公式为：

```text
managed_bytes = resident_bytes + activation_bytes + workspace_bytes
activation_bytes = 2 * 最大层宽度 * sizeof(f32)
workspace_bytes = 最大单层(weight 元素数 + bias 元素数) * sizeof(f32)
```

预算是受管理张量存储上限，**不是进程 RSS 上限**：不包含解析期间的 JSON、IR 元数据、
OS 文件页缓存、进程/管道开销，以及 I/O 的 4 KiB 栈缓冲区。大模型控制文件应使用 file
源，避免在 JSON 中重复嵌入大量参数。会话不承诺锁定物理内存或没有缺页。
文件必须在会话期间保持不变；当前没有文件摘要锁定或磁盘写回。常驻参数的显式读写与会话 revision 见下节。
prepare 检查长度；执行时检测短读和文件中非有限值，不能静态保证磁盘永远可读。

`Session::run(&mut self, input: &[f32]) -> Result<&[f32], &'static str>` 的成功执行
不进行堆分配，也不依赖垃圾回收。结果借用会话存储；Rust 借用规则阻止仍使用旧结果时
再次可变借用会话。drop 确定性释放权重、缓冲区和文件。Rust 核心生产代码无 unsafe。
测试通过全局分配计数器验证内存和文件参数路径，并非仅观察 RSS 或运行一次 warmup。

## GPU 存储和能力边界

CPU 的预算公式保持不变。GPU 总预算为 `host_managed_bytes + gpu_buffer_bytes`，
后者包含不同 weight/bias 组合的常驻有效权重、文件表达式共享暂存区（至少 4 字节）、
两个激活缓冲、回读缓冲和每算子 16 字节控制参数。纯常驻 view/add 上传一次，
显式修改参数后刷新缓存；依赖文件的表达式每次读取并上传，提交顺序防止暂存覆盖。
GPU 分配失败、超出设备 buffer/dispatch 限制均拒绝候选。

核显/统一内存设备可执行，但当前仍创建独立主机和 GPU 缓冲，预算分别计费；
设备 memory_model 只是报告信息，不代表已实现共享物理分配或零拷贝。
预算不包含驱动、编译、校准输入、上传暂存和系统开销，不能作为硬性 RSS/显存峰值上限。

当前没有原生自动求导、优化器、KV cache、通用图、GPU batch 或异步文件预取。
CPU linear 为标量 Rust 实现；GPU linear 使用工作组并行归约，文件 I/O 尚未与计算重叠。
不能把此阶段描述为完整 PyTorch 替代。所有能力为本地源码实现，尚未发布。

验证命令：`cargo test -p weightlang-runtime`、
`cargo clippy -p weightlang-runtime --all-targets --all-features -- -D warnings`、
`python -m pytest tests/test_native_backend.py tests/test_native_devices.py`
（先构建二进制；设置 `WEIGHTLANG_REQUIRE_GPU=1` 可要求实际 GPU 测试不能跳过）。

## 显式参数读写增量（0.3，本地未发布）

NativeSession 新增 parameter_info/read_parameter/write_parameters，Rust Session 与
ExecutionSession 提供对应原语；resident/view 可写，file/add 只读。
完整批次先检查，再更新共享底层存储；GPU 缓存在下一次执行前刷新。
IR 参数可选 frozen=false，冻结表达式会保护其全部存储依赖。
完整签名、默认值、形状与扁平区间规则、错误恢复、revision 和可运行例子见
[参数更新](parameter-updates.md)。该增量不增加自动求导或内置优化器。
