# 模型测试与旧脚本测试边界

以 `docs/native-testing.md`、`crates/wgt/src/main.rs::Cmd`、core Suite 实现及真实服务测试为准。当前 `wgt` 没有内置 run、script、test；不要生成目录发现、CLI filter、JSON fixture 汇总或 `wgt test --compiled` 的使用承诺。

## 当前模型路径

新增测试由 Cargo 运行 Rust 主程序。定义解析与服务实例化分离：先检查 `ModelDefinition` / 宏，再显式连接服务、初始化实例、推理或训练并验证结果。API 以 `docs/native-backend.md` 为准。

`crates/wgt-model/tests/service.rs` 的纯定义测试默认运行；真实服务用例需要先构建 `wgtime-service --bin wgtime`，设置 `WGTIME_TEST_DAEMON` 并使用 `--include-ignored`。具体命令见 `docs/native-testing.md`，共享服务启动与回收 helper 在 `crates/wgt-model/tests/support/mod.rs`。不要用尚未执行的 ignored 用例作为验收证据。

状态回归必须区分普通调用与类别采样：服务 Adaptive JSON 会话的 `Request::Predict` 通过 `run` 成功提交 forward 写入，失败回滚；旧 core `sample_model_result` 使用 `evaluate`，保留模型与优化器状态，但仍消耗 RNG 和操作预算。对应回归为 `crates/wgtime-service/src/main.rs` 的 `adaptive_predict_commits_writes_and_rolls_back_failed_output` 与 core `sample_from.rs` 的 `sampling_discards_forward_writes_and_keeps_optimizer_state`。`InitializeDefinition` 的 `CompiledForward` 仍拒绝 `Write`，不从兼容路径推断该入口支持可写 forward；本次不新增协议或 ABI。

## core 兼容层

`wgtime_core::script::testing::Suite` 只从传入源码发现普通 `test_*` 函数，函数名排序，每次显式调用独立运行且不执行 main。调用方提供 root、操作预算和依赖；错误保留源码行号，后续调用及汇总由 Rust 调用方决定。带参测试调用失败，构造器不是测试函数。它不扫描文件系统或汇总 JSON fixtures。

- `crates/wgtime-core/tests/native_testing.rs`：发现、排序、独立运行、预算恢复、构造器排除，以及 benchmark 契约。
- `crates/wgtime-core/tests/assert_syntax.rs`：严格布尔、消息惰性和条件单次求值；旧调用形式的拒绝见 `legacy_syntax.rs`。
- `crates/wgtime-core/tests/model_calls.rs`：句柄与函数绑定、参数数量、shape/行号错误及模型/checkpoint 回载。
- `crates/wgtime-core/tests/sample_record.rs` 与 `sample_from.rs`：采样概率、RNG、输入单次求值、错误绑定和不提交 forward 写入。
- `crates/wgtime-core/tests/categorical.rs`、`ppo_typed.rs`、`ppo_rollout.rs`、`ppo_records.rs`：类别契约、typed PPO 数值、首次 rollout 对照及完整记录批次训练。
- `crates/wgtime-core/tests/slicing.rs` 与 `postfix_mean.rs`：列表/张量行为、梯度、训练和错误输入。

保留 `weightc compile` 的客户端边界：`script_compiler` 当前转到 `client_compiler`，不支持的 `SampleFrom`、`LetSample`、`Slice` 在构建前拒绝。旧 core codegen 中存在对应实现，不代表当前 compiler 已迁移。删除旧 AOT 成功预期前，保留其尚未被等价覆盖的解释器数值/状态断言，并单独验证编译拒绝。

驻留策略要求单一源码单元；含 import 且所选入口可达静态策略调用的组合在客户端生成前拒绝，使用 `client_compiler_boundaries.rs` 检查不产生可执行文件。`script_compiler.rs` 覆盖命名入口及客户端/策略共享辅助函数，`async_aot.rs` 覆盖实际生成程序中的嵌套任务、索引、绑定与布尔错误；两者使用隔离服务 fixture。核心同步/异步绑定对照见 `wgtime-core/tests/async_tasks.rs`，驻留调用和短路对照见 `wgtime-service/tests/strategy.rs`。外部服务启动失败应明确报告，不添加环境条件将它变成通过。

benchmark 属于旧脚本兼容层：mode 默认为 cpu，也支持 gpu/适配器 ID 与 train:cpu。训练 workload 为 `[inputs,targets,rate,"mse"]`，warmup/repeats 均更新模型；GPU 测量同步上传到回读的主机耗时，无设备报错，不回退 CPU。它不是 Rust facade 的 benchmark API，GPU 训练计时未接入。
