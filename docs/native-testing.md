# Rust 模型测试与旧脚本兼容性测试

当前 `wgt` 内置命令负责包管理、源码检查与制品处理，**没有 `run`、`script` 或 `test` 子命令**。命令列表以 [CLI 定义](../crates/wgt/src/main.rs) 为准；同名 PATH 扩展不属于内置测试入口。`wgt check model.wgt` 只检查语法和原生模型 lowering，不执行测试或训练，具体范围见[检查实现](../crates/wgt/src/source_check.rs)。

## Rust 主程序与真实服务

新增模型测试使用 Rust 测试框架。定义解析、宏展开检查与模型执行分别验证：`ModelDefinition::parse`、`ModelDefinition::from_file`、`model!` 和 `include_model!` 构造定义；执行测试再显式连接 `wgtime`、初始化实例并断言数值结果。训练循环、数据切分、指标归约与文件 I/O 由 Rust 主程序负责，API 见 [Rust 后端契约](native-backend.md)。

[facade 服务测试](../crates/wgt-model/tests/service.rs) 覆盖定义入口、前向、训练、权重保存与加载、失败后的状态保持和资源释放。它也编译实际 Rust consumer，删除源码后以不含构建工具链的 PATH 运行；模型仍由真实服务执行。服务协议层的隔离、原子训练与 checkpoint 续训由[服务集成测试](../crates/wgtime-service/tests/service.rs)覆盖；协议内部的 checkpoint 覆盖不代表 facade 已提供公共 checkpoint API。

只验证纯定义可以运行：

```sh
cargo test --locked -p wgt-model --test service definitions_are_pure_and_macros_use_actual_facade
```

真实服务测试需要先构建 daemon，并显式包含默认忽略的服务用例：

```sh
cargo build --locked -p wgtime-service --bin wgtime
WGTIME_TEST_DAEMON="$(pwd)/target/debug/wgtime" \
  cargo test --locked -p wgt-model --test service -- --include-ignored
```

测试 helper 启动隔离服务、等待协议握手并在结束时回收进程；实现见[服务测试 helper](../crates/wgt-model/tests/support/mod.rs)。仅运行默认测试集或成功解析定义，不能计作真实服务执行通过。

### 普通调用与类别采样的状态规则

普通 Rust `Model::forward` 对应 `Request::Predict`，使用 `AdaptiveSession::run`：成功完成才提交 forward 内的 `Write`，执行失败保留调用前状态。[服务内回归](../crates/wgtime-service/src/main.rs) `adaptive_predict_commits_writes_and_rolls_back_failed_output` 同时检查共享定义编译入口与兼容 Adaptive JSON 入口的连续调用提交和写后失败回滚。

旧 core 的类别采样 `sample_model_result` 单独使用 `evaluate`，不提交 forward 写入，也不改变优化器状态；它仍消耗采样 RNG 和操作预算。“只读”指模型状态，不表示没有采样开销。该行为由[采样回归](../crates/wgtime-core/tests/sample_from.rs) `sampling_discards_forward_writes_and_keeps_optimizer_state` 验证。

`InitializeDefinition` 启用 `CompiledForward`，普通写入与工作内存均由原生编译路径执行。[Rust SDK 状态用例](../crates/wgt-model/tests/native_state.rs)通过真实服务检查提交、回滚、内存生命周期、自定义 loss 梯度和权重回载；[完整指令用例](../crates/wgtime-core/tests/compiled_instruction_coverage.rs)与[梯度和状态用例](../crates/wgtime-core/tests/compiled_semantics.rs)检查 C 原生入口与参考执行的一致性。覆盖范围见 [Rust SDK](rust-sdk.md#当前模型覆盖范围)。

## 保留的 core Suite API

`wgtime_core::script::testing::Suite` 是旧普通脚本运行时的库级兼容接口。它接收调用方提供的源码，不扫描 `tests/` 目录，也不提供 CLI filter、JSON fixture 汇总或退出码协议。其[实现](../crates/wgtime-core/src/script/testing.rs)与[行为测试](../crates/wgtime-core/tests/native_testing.rs)界定了以下范围：

- `Suite::parse` / `parse_with_imports` 发现以 `test_` 开头的普通函数，按函数名排序；模型构造器不作为测试函数。
- `Suite::run(name, root, max_operations, dependencies)` 执行指定函数，不执行 `main()`；无参函数可运行，带参数的测试在调用时失败。
- 每次调用建立独立运行时，隔离模型、随机数、指标和操作预算。预算与相对路径根目录由 Rust 调用方显式提供。
- 断言和运行时错误保留源码行号。一次调用失败后，调用方仍可执行其他测试；汇总策略由调用方决定。磁盘、网络等外部副作用不随运行时销毁回滚。

对应维护检查：

```sh
cargo test --locked -p wgtime-core --test native_testing --test assert_syntax
```

纯测试编译单元可以没有 `main`；这不提供可执行文件入口。旧 CLI 的目录发现、`--filter`、`--compiled`、`tests/*.json` 共存及统计输出已不再是当前命令契约，也没有恢复旧 Python revision 执行路径。

### 兼容层行为覆盖

维护旧测试时，以实际行为用例区分已有覆盖与已退役的入口承诺；下列 core 测试已通过定向验证。客户端编译成功与真实服务执行仍是独立验证范围。

| core 测试 | 保留的行为 |
| --- | --- |
| [native_testing](../crates/wgtime-core/tests/native_testing.rs) / [assert_syntax](../crates/wgtime-core/tests/assert_syntax.rs) | 测试函数隔离、预算、行号、严格布尔及惰性断言消息 |
| [model_calls](../crates/wgtime-core/tests/model_calls.rs) | 模型句柄绑定、函数作用域、参数数量、shape 错误与 checkpoint 字节一致性 |
| [categorical](../crates/wgtime-core/tests/categorical.rs) | 类型沿函数传递、输出类型序列化、概率验证、RNG 与转换计费 |
| [sample_record](../crates/wgtime-core/tests/sample_record.rs) / [sample_from](../crates/wgtime-core/tests/sample_from.rs) | action/log_prob、单次输入求值、RNG、错误绑定、forward 写入丢弃及优化器状态保持 |
| [ppo_typed](../crates/wgtime-core/tests/ppo_typed.rs) / [ppo_rollout](../crates/wgtime-core/tests/ppo_rollout.rs) / [ppo_records](../crates/wgtime-core/tests/ppo_records.rs) | typed PPO 概率公式、首次采样行对照、完整记录批次训练更新与 RNG 一致性 |
| [slicing](../crates/wgtime-core/tests/slicing.rs) / [postfix_mean](../crates/wgtime-core/tests/postfix_mean.rs) | 列表与张量数值、梯度、训练和错误输入 |

类型序列化测试不自动覆盖 typed checkpoint 文件回载；一次 PPO 更新也不等于长期训练阈值验收。迁移旧 AOT 测试时，仍需保留这些组合场景中独立的解释器断言。

## 旧脚本编译与评估边界

`weightc compile` 和 `weightc compile-client` 当前都调用[client compiler](../crates/wgt/src/script_compiler.rs)，只接受已迁移的构造，并在生成可执行文件时要求显式入口。编译不会执行训练；客户端模型操作需要服务。旧 core 的解释器或 `script::codegen` 能处理一种语法，不代表当前 client compiler 已支持它。

例如类别采样的 `sample from`、结构化采样绑定与普通列表切片仍有 core 解释器回归，但[当前 client compiler](../crates/wgt/src/client_compiler.rs)没有对应 `SampleFrom`、`LetSample`、`Slice` 分支，会在 Cargo 构建前明确拒绝。维护测试应分别验证解释器行为与编译边界，不恢复旧 CLI 或将编译拒绝改成跳过检查。

core 兼容层仍可由 Rust 调用方通过 `Suite` 或 `script::run` 执行旧评估函数：显式加载模型、使用 `jsonl` / `batch` / `close` 读取数据，再归约 MSE、准确率等指标并断言阈值。相对路径使用调用方传入的 root；空数据集、模型状态、类别编码和数据切分均需显式处理。这些能力不等于 Rust facade 提供了对应脚本内置函数，也不使已移除的 `examples/native_evaluation` 重新成为可运行项目。

## benchmark

本节描述保留中的 core 脚本兼容接口。

`benchmark(model,workload,warmup,repeats,name[,mode])` 返回指标对象并登记到 metric。
mode 默认 `"cpu"`；warmup 为 0..10000，repeats 为 1..100000 整数，其他参数必填。
仅顺序模型，不支持 forward 程序模型。计时单位 us，mean 算术平均；p50/p95 为排序后
`floor((N-1)*q)` 项（q=0.5/0.95），无插值。排除脚本 dispatch、数据解析、模型/设备初始化。

| mode | workload | 测量范围 |
| --- | --- | --- |
| `"cpu"` | 单样本数值向量 | CPU 前向；warmup 前额外验证一次 |
| `"gpu"` 或精确适配器 ID（如 `"vulkan:0"`） | 单样本数值向量 | 第一台硬件 GPU / 指定 GPU 的同步 host-to-host 前向 |
| `"train:cpu"` | `[inputs,targets,rate,"mse"]` | CPU 前向、批均值 MSE、反向、优化器更新 |

GPU 需要 gpu 构建特性；无设备、内存/算子不支持即失败，不回退 CPU。
GPU 私有队列在计时前完成一次不计时的同步前向，即使 warmup=0；每次 run 等待本次提交
并映射回读，因此计时包括输入/必要权重上传、编码提交、设备执行、等待和输出回读，
不是 GPU 内核时间，也不是跨队列全局设备屏障。设备张量预算固定 256 MiB。
宿主策略/compute_budget 下拒绝直接 GPU benchmark，沿用直接 GPU 调用权限边界。

训练计时只支持 CPU MSE，rate 为正有限 f32，inputs=[样本,特征]，targets=[样本,输出]。
SGD/Adam 沿用模型优化器；warmup 与 repeats 均真实更新同一模型，不重置或保存。
GPU trainer 不允许作为 train:cpu 计时；GPU 训练步计时尚未接入。
