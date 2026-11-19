# 有限算力下的执行与权重复用：2026-11-20

本轮减少重复参数复制、重复元数据解析和矩阵访问中的索引计算。实测覆盖 Python 权重访问与计划构造，以及 Rust 解释器的矩阵内核；这些结果还不能量化架构搜索质量、重训练轮数或模型端到端速度。原始样本与范围说明见[便携证据摘要](benchmarks/efficiency-2026-11-20.json)。

## 四项机制及用途

| 机制 | 实现与用途 | 成本边界 |
| --- | --- | --- |
| 只备份可写参数 | 预处理收集 `Instruction::Write` 的目标，事务开始与回滚只复制这些参数。适合大权重、少量可变状态的反复评估。临时参数绑定先检查额外数值缓冲区预算，退出时恢复原参数。见[事务实现](../crates/wgtime-core/src/adaptive.rs)、[临时绑定](../crates/wgtime-core/src/adaptive/autodiff.rs)。 | 受管预算计量 `f32` 存储，不代表进程总内存。训练更新仍有独立候选值和优化器状态；本报告未单独测量此机制的壁钟收益。 |
| 按步长访问矩阵 | 在矩阵行与批次边界计算偏移，以步长递增访问元素，并同时累加四列，减少视图每次读取时的除法、取余和重复左值读取。见[生产内核](../crates/wgtime-core/src/adaptive/matmul.rs)。 | 本次测量只覆盖 Rust 解释路径使用的内核，未测量生成的 C 路径或完整模型。 |
| Rust 不可变数值共享 | `Tensor::clone` / `reshape`、`WeightSet::clone`、`Plan::from_base` / `copy` 共享 `Arc<Vec<f32>>`，便于复用候选的未修改权重。见[权重实现](../crates/wgt-model/src/weights.rs)及[共享契约](weights.md)。 | 形状、名称和映射仍有维护成本；算术变换与对外 `bindings` 独立分配。`Plan::build` 仍完整序列化并生成新快照，候选构建不是零成本。 |
| Python 只索引元数据、只复制所取张量 | 完整验证后建立私有只读的名字到形状与字节区间索引；引用查找不再反复解析整个 header，读取只复制目标区间。返回的 Torch tensor 可修改，并独立于快照及其他读取。见[Python 实现](../python/weightlang/weights.py)。 | 索引不缓存解码 tensor；保留多个返回值仍需相应数据内存。全量 `tensors()` 与 `Plan.build()` 仍处理完整权重。 |

## Python 权重访问与计划构造

测试数据为 64 个 `256 × 256` 的 F32 参数，共 16,777,216 字节数值（16 MiB），Torch 为 `2.15.0`，单线程。`read` 每轮遍历全部 64 个固定引用、逐个读取并累计校验值；它不保留全部返回 tensor。`plan` 每轮从原快照构造完整计划并构建快照，检查 payload revision 一致。定义见[基准脚本](../python/benchmarks/weight_snapshots.py)。

每个版本、每个案例分别运行 3 个新进程，每进程计时 4 轮；表中时间是三个进程的“总计时 / 4”的中位数。计时时关闭 `tracemalloc`，随后另运行一轮记录 Python 可追踪分配峰值。基线与最终索引版分批取得，属于非配对样本；加速比是两组中位数之比，按未舍入数值计算。

| 操作 | 基线时间（ms/轮） | 最终时间（ms/轮） | 中位数比 | 基线追踪峰值（B） | 最终追踪峰值（B） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 逐个遍历 64 个权重引用 | 94.152 | 1.469 | 64.105× | 16,835,520 | 525,498 |
| 完整 Plan 构建 | 230.219 | 96.036 | 2.397× | 33,801,068 | 34,055,958 |

逐个读取的追踪峰值下降约 96.88%；这个指标只覆盖 `tracemalloc` 可见分配。Plan 的追踪峰值没有改善，与其仍需全量物化、校验和序列化的行为一致。原始进程 RSS 包含导入、夹具与运行时开销：读取中位数为 343,688 → 344,788 KiB，Plan 为 368,720 → 367,228 KiB，不能据此宣称整体 RSS 改善。

基线取自 `weight-snapshots-comparison.json` 的 `phase=before` 行，最终结果只取 `weight-snapshots-indexed-after.json`。前者的中间 `after` 行以及更早的独立 before/after 文件没有进入本报告。每组仅 3 个进程，数值用于描述这台机器上的固定夹具，不作为普遍加速保证。

## Rust 解释内核矩阵微基准

每个布局运行 9 组旧实现与新内核的配对样本，按组交替先后顺序；每次样本执行 8 个矩阵乘积，再取每个乘积耗时的中位数。计时前逐个比较输出 F32 位模式，五个夹具全部与旧实现一致。机器为 Intel Core i9-12900HX，可用 CPU 亲和集合为 24 个逻辑 CPU；编译器为 `rustc 1.100.0 (0ef1bc386 2026-11-09)`，参数 `--edition=2024 -O`。

| 布局 | 批数 × 行 × 内维 × 列 | 旧实现（μs/乘积） | 新内核（μs/乘积） | 中位数比 |
| --- | --- | ---: | ---: | ---: |
| `rank2-owned` | 1 × 64 × 96 × 48 | 449.496 | 220.466 | 2.039× |
| `rank2-reshape-view` | 1 × 64 × 96 × 48 | 2662.026 | 212.281 | 12.540× |
| `rank2-transposed` | 1 × 64 × 96 × 48 | 2925.445 | 271.587 | 10.772× |
| `rank2-offset-tail` | 1 × 31 × 65 × 33 | 548.482 | 43.221 | 12.690× |
| `rank3-permuted-batch` | 4 × 32 × 64 × 40 | 4073.714 | 207.599 | 19.623× |

[基准脚本](../scripts/bench_strided_matmul.rs)直接引入生产内核，旧实现保留逐元素的视图索引计算。表中较大的比值主要出现在这些视图夹具，不能外推到生成的 C、GPU、训练、完整模型或架构搜索。原始记录只保留中位数，没有保留各组耗时，因而不能重算配对差值分布或置信区间；本次也没有测量内核的 RSS 收益。

## 复现与证据来源

在仓库根目录使用装有 Torch `2.15.0`、`safetensors` 的 Python 环境。以下循环为每种操作启动 3 个独立进程；脚本内部负责 4 轮计时和单独一轮分配追踪。

```bash
for trial in 1 2 3; do
  PYTHONPATH=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python python/benchmarks/weight_snapshots.py --case read --parameters 64 --side 256 --iterations 4
  PYTHONPATH=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python python/benchmarks/weight_snapshots.py --case plan --parameters 64 --side 256 --iterations 4
done
```

基线模块固定为提交 `2e08ef04dd5c5d7f27d2e670d264f4fa02b971bb` 的 `python/weightlang/weights.py`，使用同一基准脚本与依赖。证据 JSON 的 `reproduction.python_baseline_loader` 保存可传给 `python -c` 的加载器：它从 Git 读取旧模块，再替换基准中的 `Plan` 与 `WeightSet`，无需修改工作树；其余参数与上述命令一致。最终索引实现对应提交 `fb7d395`。

矩阵微基准无需重建 Cargo 工作区：

```bash
mkdir -p /tmp/weightlang
rustc --edition=2024 -O scripts/bench_strided_matmul.rs -o /tmp/weightlang/bench-strided-matmul
/tmp/weightlang/bench-strided-matmul
```

开发机原始记录位于 `weights/efficiency-20261120/`：`weight-snapshots-comparison.json`、`weight-snapshots-indexed-after.json` 与 `strided-matmul.json`。便携 JSON 保留选中的 12 条 Python 样本、5 个矩阵案例、聚合方式、来源文件 SHA-256 和矩阵源码 SHA-256，省略临时可执行文件绝对路径及重复 stdout。矩阵记录的 UTC 时间为 `2026-11-19T23:05:45.924189+00:00`。

## 集成验证与传播

本轮完成 123 项自动化测试，另完成独立 wheel 验收、两个应用的编译检查，以及 benchmark 的 6 次原生单步运行。所有下列命令均退出 0；按本地验证策略记录，不以未运行的全仓或远端检查补足结论。

| 验证范围 | 实际结果 | 覆盖与证据 |
| --- | --- | --- |
| 核心运行时 | 9 suites、71 passed、0 ignored | 状态回滚、编译与解释路径、梯度、视图、动态几何、工作区；完整命令和日志摘要见证据 JSON |
| Rust 权重库 | 9 passed | 共享地址、生命周期、形状检查、快照隔离与迁移 |
| 默认和显式导出目标 | 1 passed | 默认 WGT IR，显式 Torch 与 Rust |
| Python 权重与模型对象 | 32 passed | `test_weights.py`、`test_model.py` 全部用例；包括 Rust/Python roundtrip 和真实服务 |
| 独立安装的 wheel | 通过 | 0.5.4；22 个包文件与源码一致，编译、梯度、无编译器恢复通过 |
| 编译器、服务与权重桥 | 构建成功 | Python 与 LightNest 的编译器/服务 SHA-256 一致；benchmark 也固定相同服务摘要 |
| iris、gates | 两个工作区均编译通过 | `cargo test --locked --offline` 退出 0；各自当前 0 个测试，不计入 123 项 |
| LightNest | 2 passed | 循环模型梯度与训练、两组 routed attention 几何及参数更新 |
| benchmark | Rust 4 passed、Python 4 passed；6 次原生执行通过 | ResNet × 2 个任务 × 3 种初始化，每次 1 个 Adam 更新、batch=1 |

核心运行时命令为 `cargo test --locked -p wgtime-core --no-default-features -j 2 --test selective_rollback --test strided_matmul --test adaptive --test adaptive_autodiff --test tensor_contracts --test dynamic_geometry --test compiled_semantics --test workspace --test custom_optimization`。Python 命令为 `python -m pytest -q python/tests/test_weights.py python/tests/test_model.py`，显式指定本轮构建的 `WGT_COMPILER`、`WGT_WEIGHT_BRIDGE` 与 `WGTIME_TEST_DAEMON`。

本地原始回执保存在 `weights/efficiency-20261120/` 的 `core-validation.json`、`build-validation.json`、`python-validation.json`、`installed-wheel-validation.json` 及其日志。便携 JSON 保存对应摘要、实际命令、文件哈希与产物哈希；其中本机路径用命名根路径替代，原始回执保留展开后的绝对路径。

### 固定源码与下游回执

- 功能源码固定为 [`072582a03a123b7d95c815d563689383033019d4`](https://github.com/WeightLang/WeightLang/commit/072582a03a123b7d95c815d563689383033019d4)。Python、核心运行时、Rust 权重共享分成三批提交并验证。
- benchmark 已更新 pin、Cargo.lock、源码归档和运行清单，提交为 [`319e9ab515adfa069e28a7f6dc95b4ede2bc3cb4`](https://github.com/WeightLang/benchmark/commit/319e9ab515adfa069e28a7f6dc95b4ede2bc3cb4)。[本轮验收](https://github.com/WeightLang/benchmark/blob/319e9ab515adfa069e28a7f6dc95b4ede2bc3cb4/validation/sdk-refresh-20261120.json)与[6 次原始运行](https://github.com/WeightLang/benchmark/blob/319e9ab515adfa069e28a7f6dc95b4ede2bc3cb4/validation/sdk-refresh-20261120-smoke.json)分别保存。此次使用 Rust 1.100.0；没有把旧 Harbor 得分归给新源码。
- LightNest 保持指向本地 0.5.4 SDK 的 editable 来源，使用同一批新编译器与服务验收；记录提交为 [`a9c4e5732b4ce9168875669de321104a3f4f278d`](https://github.com/LIghtJUNction/lightnest/commit/a9c4e5732b4ce9168875669de321104a3f4f278d)，[范围与复现命令](https://github.com/LIghtJUNction/lightnest/blob/a9c4e5732b4ce9168875669de321104a3f4f278d/docs/weightlang-efficiency-2026-11-20.md)包含两项真实服务测试。

完整依赖边和责任见[依赖传播清单](dependency-propagation.md)。Project Z 的维护方收到独立证据与候选补丁后，明确由其按实际 Z 库关联决定纳入范围；本轮在 WeightLang 内维护生态图，没有把候选登记当作已生效。

本轮不改变 WGT 语法或源码版本。仍需重点处理的成本是完整快照的验证与序列化、训练候选与 tape 的独立数据，以及尚未改变的 generated-C 矩阵内核。现有数据证明了所测操作的开销下降，没有证明架构搜索质量提升、重训练轮数下降或 RSI 已实现。
