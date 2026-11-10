# WGT 到 Rust：模型导出与原生执行

WGT 描述模型参数、内部计算、分支循环和状态；Rust 应用负责输入数据、训练循环、文件和调度。`wgt compile --target rust` 把一份经过共享前端检查的模型导出为 Rust 模块。Rust SDK 用这个定义向 `wgtime` 服务创建实例，服务编译并执行模型。[导出实现](../crates/wgt/src/rust_export.rs)、[Rust SDK](../crates/wgt-model/src/lib.rs)和[服务入口](../crates/wgtime-service/src/main.rs)分别负责这三步。

## 快速开始

在模型目录中创建 `model.wgt`：

```wgt
wgt 0.5
model Scale
input x: f32[2]
output y: f32[2]
# 每个输入特征有一个独立增益；声明值是初始化值。
parameter gain: f32[2] = [2,3]
fn forward(x) { return x * gain; }
```

在 Rust 应用的 `Cargo.toml` 添加依赖，调整路径到当前 WeightLang 源码目录：

```toml
[dependencies]
wgt-model = { path = "../WeightLang/crates/wgt-model" }
```

生成模块，然后在 `src/main.rs` 使用它：

```sh
wgt compile model.wgt --target rust -o src/model.rs
```

```rust
mod architecture { include!("model.rs"); }

fn main() -> wgt::Result<()> {
    let definition = architecture::definition()?;
    let client = wgt::Client::connect()?;
    let model = client.initialize(&definition, wgt::InstanceOptions::default())?;
    assert_eq!(model.backend(), "native-cpu");
    assert_eq!(model.forward(&[1.0, 2.0])?, vec![2.0, 6.0]);
    model.close()?;
    Ok(())
}
```

执行前需要启动匹配版本的 `wgtime` 服务；[服务启动说明](../README.md#use-wgt-from-rust)所在章节上方给出了构建与启动命令。SDK 使用 `WGTIME_ENDPOINT` 指定的 Unix socket，未设置时连接 `/run/wgtime/service.sock`。定义构造不会启动服务或执行模型；初始化默认使用 64 MiB 托管内存预算、无优化器。

## 导出入口

```sh
# 默认源文件 model.wgt，省略 -o 时输出到 stdout。
wgt compile --target rust

# 从 stdin 读取；标准纯计算模块仍可使用。
wgt compile - --target rust < model.wgt

# Cargo 依赖改名时，调整生成代码引用的 crate 路径。
wgt compile model.wgt --target rust --crate-path renamed_wgt -o src/model.rs
```

生成文件公开 `SOURCE: &str` 和 `definition() -> wgt::Result<wgt::ModelDefinition>`。文件输入会解析并嵌入纯计算 imports 的完整闭包，运行时无需原始 WGT 文件。输出文件已存在时拒绝覆盖。`--crate-path` 只用于 Rust 目标，默认 `wgt`；它必须是有效的 Rust 模块路径。[CLI 验收](../crates/wgt/tests/rust_export.rs)覆盖输出、错误输入、导入、依赖改名和生成文件的实际编译。

导出物保存自包含架构源码；调用 `definition()` 时仍由共享前端解析定义。声明的初始化方式和 seed 随架构保留，服务初始化才创建实际参数。导出不生成训练权重，也不执行 forward。架构变更后需重新导出并重新编译 Rust 应用。

0.5.2 增加 `wgt::Dimensions`（`BTreeMap<String, usize>`）及 `ModelDefinition::parse_with_dimensions(source, &dimensions)` / `from_file_with_dimensions(path, &dimensions)`，用于在检查前绑定源码顶层 `dim`。默认入口使用声明默认值；显式绑定不初始化或训练。完整限制和可训练签名见[模型语言](model-language.md)。

直接嵌入模型的既有方式仍可用：`wgt::model!("...")`、`wgt::include_model!("model.wgt")`、`ModelDefinition::parse(source)` 和 `ModelDefinition::from_file(path)`。它们和 Rust 导出使用同一个模型前端，详细签名见[后端契约](native-backend.md#三种-rust-入口)。

## 当前模型覆盖范围

覆盖基准为 WGT 0.5 共享前端产生的三种模型计划：顺序 `Graph`、内部程序 `Forward` 和具名 `Operations`。内部程序的完整集合由 [tensor_contracts.json](../language/tensor_contracts.json)列出的 **48 个 IR 指令**定义。编译器对枚举逐项处理；[指令差分测试](../crates/wgtime-core/tests/compiled_instruction_coverage.rs)对照这份目录，防止新增指令后遗漏编译验收。状态子模型、显式 FF 更新与 `wgt::evolution` 原生 GA 的接口见[无反传学习](local-learning.md)。

| 指令类别 | 当前覆盖 |
| --- | --- |
| 线性计算与卷积 | Linear、Conv、Matmul、BatchMatmul |
| 形状与索引 | Reshape、Transpose、Permute、Slice |
| 逐元素与常量 | Add、Mul、Scale、Minimum、Clip、Neg、Exp、Log、Tanh、Sigmoid、Relu、Detach、Constant |
| 归约、规范化和池化 | Sum、Mean、Softmax、LogSoftmax、SoftmaxAxis、LayerNorm、MaxPool2d、AvgPool2d、ValidateProbabilities |
| 路由与控制流 | Gate、Route、InferenceSkip、Label、Branch、Jump、Return |
| 参数读写 | Read、Write |
| 调用内工作内存 | MemoryAlloc、MemoryFree、MemoryRead、MemoryWrite、ResourceState |

源码中的 helper、子模型、共享或独立重复层、learn 参数和纯模块导入继续使用共享前端的组合规则；它们最终转换为上述指令。typed operations 保留自身合同：具名零输入或多输入入口、状态与参数事务、返回向量或 void；它们不支持自动微分、loss 或子模型，详见[模型操作](model-operations.md)。

服务初始化将模型指令生成为 C，再由本机 `cc` 编译共享库；Rust 应用通过 SDK 调用服务。因此这个 Rust 导出模块需要 SDK 和服务，服务当前需要 Linux 与 C 编译器。推理失败、编译失败或预算不足会返回错误。服务定义入口不回退到解释器。[编译器](../crates/wgtime-core/src/compiled_forward.rs)和[运行时](../crates/wgtime-core/src/adaptive.rs)保留同一执行与错误边界。

## 写入、内存与训练

`Model::forward` 对应一次可提交的普通模型调用。forward 中的 `write` 先写候选值，后续读取能看到它；只有整个调用成功才提交，出现非有限数、非法地址、预算耗尽等错误时保留调用前参数。typed `Model::call` 同样在成功后原子提交显式状态及参数写入。[服务回归](../crates/wgt-model/tests/native_state.rs)检查连续调用、实例隔离、写后失败和权重回载。

`memory_alloc` 创建当前调用的临时 bank。`memory_write` 使用每个 slot 的地址权重更新内容，`memory_read` 按权重加权读取；`memory_free` 释放 bank。跨调用不会保存 bank；需要持久值时显式写入参数或 typed state。`resource_state()` 返回剩余 work、剩余托管内存字节、当前存活 bank 字节。编译路径的固定工作区也占用预算，所以它和解释路径可能报告不同的剩余内存；该值不是整个进程的 RSS。

带梯度的前向使用相同原生入口，回调只记录实际执行的张量及版本关系；反向继续使用 Rust 梯度 tape。参数重绑定、工作内存多次覆盖和释放后的梯度连接保留其原有规则。训练对调用前参数计算更新，不提交训练前向的临时写入。core 内部 `forward/evaluate` 的快照行为和 `run` 的提交行为由[状态与梯度测试](../crates/wgtime-core/tests/compiled_semantics.rs)分别验证。

```rust
let model = client.initialize(&definition, wgt::InstanceOptions {
    optimizer: Some(wgt::Optimizer::Adam),
    ..Default::default()
})?;
let loss = model.train(
    &[vec![1.0, 2.0]],
    wgt::Targets::Values(&[vec![0.0, 0.0]]),
    wgt::TrainOptions { rate: 0.01, loss: wgt::Loss::Mse, clip: Some(1.0) },
)?;
```

支持 MSE、交叉熵和 WGT 声明的自定义 loss。learn 浮点范围、正数、bool 和 int 继续遵循原生读取和更新投影规则；需要明确离散梯度策略时使用 `train_with_policy`。完整训练合同见[原生后端 API](native-backend.md#显式生命周期与训练)。

## 其他 Rust SDK 功能

| 接口 | 用途 |
| --- | --- |
| `definition.interface/parameters/states/operations` | 检查输入输出和存储 schema，无需服务实例 |
| `model.call(name, inputs)` | 调用 typed 模型的具名入口 |
| `model.parameters()` / `model.state()` | 读取实例参数或 typed state |
| `model.update_parameters(candidates)` | 原子校验并替换候选参数，供外部搜索使用 |
| `model.save_weights(path)` / `load_weights(path, optimizer)` | 显式保存、绑定 safetensors 参数；保存拒绝覆盖，加载失败保持实例 |
| `model.backend()` / `client.stats()` | 查询执行后端和服务资源信息 |
| `model.close()` | 释放实例；连接和句柄失效会明确报错 |

普通权重只保存参数，typed state 与优化器 checkpoint 有独立合同；当前 Rust facade 没有公共训练 checkpoint API。客户端不自动重建失联实例。

## 验证与边界

```sh
cargo test --locked --no-default-features -p wgtime-core \
  --test compiled_instruction_coverage --test compiled_semantics \
  --test compiled_forward --test compiled_training --test compiled_control_flow
cargo test --locked -p weightlang-wgt --test rust_export
cargo build --locked -p wgtime-service --bin wgtime
WGTIME_TEST_DAEMON="$(pwd)/target/debug/wgtime" \
  cargo test --locked -p wgt-model --tests -- --include-ignored
```

上述检查分别证明当前指令的编译执行、状态与梯度合同、生成 Rust 代码可编译、SDK 连接真实服务。全指令覆盖不意味着任意 PyTorch 模型都可表达：原生数据合同仍是固定 shape 的 f32，普通 forward 单向量，typed 操作按声明接收多个向量；动态 shape、任意 dtype、GPU 内核和编译后的 backward 不属于这个目标。旧脚本宿主控制与分段 `start_execution/resume_execution` 也不属于模型导出；compiled 实例会明确拒绝旧分段执行接口。
