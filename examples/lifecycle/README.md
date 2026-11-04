# 原生模型生命周期回归测试

本例学习单特征回归 `y = 2x + 1`。`model.wgt` 通过 JSONL 流分批读取五行数据，按显式 `split` 字段划分三个训练样本和两个评估样本。每个样本输入和输出均为 `f32[1]`；训练批次为 `[3,1]`。模型以 seed 42 初始化，Adam 的全部更新共享同一组线性参数。

`run.jsonl` 控制本次执行的训练步数、是否恢复断点以及是否执行最终质量断言。`build` 为 false 时返回可序列化的预测值；为 true 时返回 `weightc build` 所需模型句柄。默认执行 300 步；脚本断言训练 MSE 至少下降 99%，完整训练后的 held-out MSE 小于 `1e-6`。这里是确定性的合成回归检查，不代表真实数据质量。

`crates/wgt/tests/lifecycle.rs` 会把示例复制到测试独占的磁盘临时目录，创建 `out/`，并调用真实的 `wgt` 和 `weightc` 二进制。无需手动创建仓库 `weights/`，也不应直接在本目录执行训练以保留测试输出。

```sh
cargo test -p weightlang-wgt --test lifecycle -- --nocapture
cargo test -p weightlang-wgt --test lifecycle --features gpu -- --nocapture
WGT_REQUIRE_GPU=1 cargo test -p weightlang-wgt --test lifecycle --features gpu -- --nocapture
```

测试设计覆盖以下行为，实际通过状态以执行结果为准：

- `weightc check` 解析全部三个 WGT 文件，检查本身不写模型。
- 300 步连续训练，与独立进程执行 100 步、保存、恢复后执行 200 步，比较完整模型和断点文件字节；检查 Adam 步数、一阶矩和二阶矩。
- `wgt test` 执行 `tests/evaluation.wgt`，验证 held-out MSE、CPU 前向和 CPU 训练 benchmark 的设备及计时字段。
- `weightc build` 生成完整制品，检查模型及断点与训练时一致；从导出的推理 IR 重载预测。
- `wgt release-assets` 在独立的临时 Git 仓库打包，核对每个发布文件的 SHA256 和大小，再从发布模型文件加载推理。只创建本地发布材料，不上传任何内容。
- `weightc compile` 编译后删除源码，再用空工具 PATH 执行独立二进制，对比解释执行报告及完整模型、断点；编译阶段不执行训练。
- 重复保存、构建和打包必须拒绝已有输出，文件内容保持不变；推理和评估前后比较目录快照，检测隐式训练或制品创建；损坏断点格式必须报 `unsupported checkpoint format`。
- GPU benchmark 只在 gpu feature 和 `WGT_REQUIRE_GPU=1` 同时启用时执行，设备不可为 CPU，不允许静默回退。

解释执行、构建、评估和独立 AOT 执行都将 PATH 设为 `/nonexistent`。发布包命令只暴露 Git。AOT 编译阶段需要本机 Cargo、rustc 和链接器，使用正常工具链 PATH；本测试不提供 Python 回调或 Python 实现。

## 原生制品与支持边界

测试最初在提交 `5d2bb29` 实测发现 `wgt package/export` 进入已移除的 Python CLI。当前已新增[原生制品命令](../../docs/native-artifact-commands.md)：package 接受完整原生构建目录并输出目录包，export 接受原生构建目录或推理 JSON，输出不含优化器的推理 JSON。生命周期测试真实执行两个命令并验证往返推理相等，不再忽略这两条路径。旧 ZIP、源码初始化导出、外部权重引用仍不支持，明确拒绝。

`weightc build`、package/export 和脚本保存接口拒绝覆盖，不支持 `--force`；测试检验不带覆盖选项时的保护。

该 HEAD 有 Rust GPU trainer，但尚无 WGT `training_device`/`training_status` 接入，因此本例只验证 GPU 前向 benchmark，不宣称完成 WGT GPU 训练。AOT 编译器在该 HEAD 不继承 gpu feature，本例 AOT 只比较 CPU 生命周期。

断点只恢复模型与优化器；本例每步使用相同的固定训练批次，不需要恢复随机采样器或数据游标。它不证明任意数据流和环境状态能够自动续接。
