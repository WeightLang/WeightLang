# Python SDK：减少运行时检查开销

本轮优化减少了动态执行器中重复的张量检查，并在无梯度推理时省去反向诊断包装。三个 CPU 单样本微基准的推理中位耗时降低约 70%–74%；数学算子、参数快照、工作量预算及事务提交规则继续使用已有实现。

对照版本为 [8fbeca0](https://github.com/WeightLang/WeightLang/commit/8fbeca0e67c3e166fc6771a2f7ba21be0e7e63bc)，优化实现为 [14e806d](https://github.com/WeightLang/WeightLang/commit/14e806d693e44a73d93c47244495343942e37080)。测量使用这两个版本的实际 Python 包，无运行时替换。

## 改动与校验边界

- 将前向有限值校验与反向梯度诊断分开；动态执行器完成同步检查后，仅挂接所需的梯度诊断。
- 关闭梯度时跳过 `autograd.Function.apply`。如果返回值来自需要梯度的输入或参数，则先 `detach()`，保持原有推理输出语义。
- 参数在调用入口及每次普通写入时验证。后续读取复用已验证候选；typed operations 继续在事务提交前校验最终候选。

前向的非有限值检查、训练时的内部梯度诊断，以及失败回滚仍保留。此次修改集中在 `ops.py` 和 `program.py` 的检查调度。

## 实测结果

环境：Linux x86_64、Python 3.12.14、PyTorch 2.15.0+cpu，float32，计算线程和 interop 线程均为 1。输入为单个标量 `[1.0]`。普通循环的全部迭代共享一个初始值为 1.001 的可训练 gain；可学习深度初值为 4、上限为 8，gain 初值为 1.01。这些参数用于测量计算开销，未经过训练。

使用 `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.2)`，每项先预热 5 次。编译和模型初始化不计时。表中单位为微秒，括号为四分位距 IQR；加速比为旧版中位数除以优化版中位数。

| 模型 | 模式 | 旧版中位数（IQR） | 优化版中位数（IQR） | 中位数加速比 |
| --- | --- | ---: | ---: | ---: |
| 循环 3 次 | 无梯度前向 | 836.267（181.989） | 253.416（38.746） | 3.30× |
| 循环 32 次 | 无梯度前向 | 6514.145（1615.679） | 1692.936（244.286） | 3.85× |
| 可学习深度 4/8 | 无梯度前向 | 7123.409（1060.804） | 1954.630（292.651） | 3.64× |
| 循环 3 次 | 开启梯度的前向 | 1001.674（193.308） | 705.940（228.631） | 1.42× |
| 循环 32 次 | 开启梯度的前向 | 6536.307（1768.254） | 5377.901（1824.814） | 1.22× |
| 可学习深度 4/8 | 开启梯度的前向 | 7939.458（2209.496） | 5820.922（1354.063） | 1.36× |

本机测量存在明显波动，部分 IQR 约为中位数的 15%–34%。开启梯度的测量只包含构建计算图的前向，不包含 backward 或 optimizer.step；表中的比值不能当作完整训练加速比。结果限定于这些解释执行开销较高的 CPU 微基准，大规模张量、GPU 和端到端训练需另行测量。

## 复现

使用仓库中的 [runtime_checks.py](../python/benchmarks/runtime_checks.py)。已有编译器时，在仓库根目录运行：

```sh
PYTHONPATH=python .venv/bin/python python/benchmarks/runtime_checks.py \
  --compiler target/debug/wgt --label current \
  --min-run-time 0.2 --output weights/runtime-checks/current.json
```

对照旧版时，将旧提交的 `python/weightlang` 目录解出，并让 `PYTHONPATH` 指向其父目录，继续运行同一份基准脚本和同一个编译器。两个版本分别启动进程，依次测量。提高 `--min-run-time` 可增加每项测量时间。输出文件使用独占创建，已有文件会被拒绝覆盖。

结果 JSON 记录实际包路径、包源码 SHA-256、各 WGT 工作负载 SHA-256、环境、median、IQR 和采样次数。本次包源码指纹为：

```text
旧版  27eec052a8b9c7ceaefdea9b93e85da9f29919d3ac98735b9057f6b1e86e44aa
优化  fcbfc4a1e22a8ef4573f86c419aee179f8f82b85bc2872a19fb9402be88f031c
```

## 正确性验收

完整 Python 套件通过 143 项测试，包含原生执行结果与梯度对照、控制流、工作内存、预算、约束投影、事务回滚、并发更新和模型保存。新增 4 个回归覆盖 `no_grad`／`inference_mode` 下直接返回输入或参数的场景，保证推理结果不携带原计算图，同时后续训练仍能正常反向传播。

运行验证：

```sh
OMP_NUM_THREADS=1 \
WGT_COMPILER="$PWD/target/debug/wgt" \
WGT_REFERENCE="$PWD/target/debug/examples/torch_reference" \
.venv/bin/python -m pytest python/tests -q
```

