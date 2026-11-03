# 原生扩展与性能

权重保存现在调用已有的 `safetensors` Rust 扩展进行二进制序列化，Python 负责
准备 CPU 连续张量、组织制品元数据和调用扩展。无需新增 Rust 编译工具链或依赖。
这是已在本地验证的源码优化，尚未发布新版本；不是整个项目已经改成 Rust。

原实现 `bytes(t.clone().untyped_storage())` 会在 Python 层逐字节读取张量。
对 256 KiB 权重进行一次独立 CPU profile，观察到 262,144 次 storage
`__getitem__` 调用。新实现通过公开的 `safetensors.torch.save` 接口调用其原生
序列化路径，移除这段逐字节循环。张量计算仍由 PyTorch 原生后端执行；语法解析、
编排、训练循环等其余 Python 路径没有在本次迁移。

## 保持文件一致

WeightLang 用文件哈希标识权重，不能直接改用默认 safetensors 文件头，因为第三方
元数据字段顺序不保证固定。Python 继续生成原来的规范化文件头，只采用 Rust 编码
得到的张量数据部分；每次处理一个张量，避免一次性保留所有已编码的二进制块。

验证范围包括：字段顺序、Unicode 名称和元数据、非连续张量、存储偏移、共享存储、
标量、空张量、负零、无穷和 NaN 的原始位模式。保存前后文件逐字节一致，所以模型
哈希和制品版本不变。此变更没有增加语言语法或训练选项。

## 本地测量

脚本：[benchmarks/serialization.py](../benchmarks/serialization.py)。
对照为提交 `548a0b4` 中的旧保存函数，两边在同一进程内交替执行。分别预热一次，
随后各运行五次。CPU 单线程，Python 3.14.9、PyTorch 2.14.1+cu130（本基准使用 CPU）、
safetensors 0.7.0。机器有其他任务，结束时 24 个逻辑 CPU 的一分钟负载约 9.2。

| 工作负载 | 旧实现中位数（范围） | 原生路径中位数（范围） |
| --- | --- | --- |
| 保存四个 128×128 float32 张量，共 256 KiB 数据 | 599.6 ms（501.5–802.4） | 0.493 ms（0.452–0.604） |
| 完整编译两个 256×256 linear 层并保存制品 | 1012.9 ms（902.4–1644.9） | 3.028 ms（2.853–7.523） |

这些数字衡量移除低效字节循环的收益，不代表 Rust 普遍比 Python 快多少倍。
完整编译测量包含解析、初始化、元数据生成及保存，但不含 Python 进程启动或训练。
文件写入使用热文件系统缓存，未执行 fsync，不是持久化磁盘吞吐量或推理性能。
所有输出都实际写入、加载，并确认两种路径的 SHA-256 相同；没有错误或跳过计算。
细粒度极小张量可能更受扩展调用和临时文件头成本影响，未声称所有模型同等提速。

原始报告保留在 `weights/native_serialization_v1/report.json`，不会作为发布权重上传。
重跑时指定新报告路径：

```sh
PYTHONPATH=src .venv/bin/python benchmarks/serialization.py \
  --baseline-ref 548a0b4 --output weights/native_serialization_new/report.json
```

基准临时制品存放在磁盘上的 `.cache/weightlang/native-perf/`，结束自动删除；报告
保留。累计写入超过 100 MB 前按项目约定检查空间。运行环境和负载变化时应重新测量，
不能把这里的编译收益套用到训练或 Responses 生成速度上。
