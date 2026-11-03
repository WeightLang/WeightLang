# 原生模型测试与评估

在仓库根目录运行（保存目标必须不存在）：

```sh
cargo build -p weightlang-wgt --features gpu
mkdir -p weights/native_evaluation
./target/debug/wgt run examples/native_evaluation/model.wgt
./target/debug/wgt -C examples/native_evaluation test
./target/debug/wgt -C examples/native_evaluation test --filter dataset
./target/debug/wgt -C examples/native_evaluation/gpu test
```

`model.wgt` 同时定义单特征线性模型和显式训练 main。前三个训练点学习 y=2x+1，
四个 JSONL 评估点检验区间内插值，不代表真实任务泛化能力。测试本身用 WGT 函数表达，
无需固定输入输出 JSON；`tests/fixed.json` 只演示旧式 fixture 与函数测试共存。
数据文件是输入数据而非测试协议，可换成现有流式数据接口。

模型、优化器断点保存在仓库 `weights/native_evaluation/`，不会自动覆盖或重新训练。
失败测试、CPU 训练计时的参数更新均不保存；每个测试使用独立运行时。
GPU 是显式独立套件，无 GPU 或构建未启用 gpu 特性时失败，不回退 CPU。
单维模型只演示计时口径，不能据此宣称 GPU 比 CPU 快。

协议和限制见[原生测试](../../docs/native-testing.md)。`wgt test --compiled` 暂不支持并报错；
普通 `wgt run --compiled` 的评估/benchmark 内置与解释器共用 Rust 实现。
