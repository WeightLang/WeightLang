# WeightLang

[English](README.md)

> **优雅地描述神经网络的内部架构。**

WGT 描述模型内部的参数、状态，以及计算和更新它们的操作。Rust 负责数据准备、I/O、调度以及训练或搜索循环；唯一的 `wgtime` 服务拥有可变模型实例。可复用算法通过标准库显式导入。

前向计算、loss 与 backprop 是一种可选的模型路径。模型也可以使用有状态操作，或接收外部生成的参数更新。操作文档所列的 core 状态回滚、无梯度搜索、真实 service 操作和 CNN 回归范围已通过本地验收。typed 操作不支持自动微分（AD）；既有 forward 路径保留显式训练策略。这些能力已在本地实现，但尚未发布；边界与验收范围见[多范式模型操作](docs/model-operations.md)。

**可学习架构参数（Learnable Architecture Parameters）**让模型在明确约束下声明数值与结构开关。以下声明已在本地实现。旧 `train` 路径对整数和布尔读取使用 STE（直通估计器）；既有 forward 路径保留显式训练策略，`learn` 本身不绑定到某种策略：

```wgt
learn depth: int in 1..=12 = 6;
learn use_skip: bool = true;
```

可通过 `depth_gate` 或有界 `repeat` 表达受约束的计算，也可用布尔参数控制残差连接。`repeat` 已在文档所述边界内本地实现并验证；`choice` 和动态形状变化仍属于设计方向。完整浮点约束、离散训练规则及权重保存方式见[可学习架构参数与已测示例](docs/learnable-architecture-parameters.md)。

参见[设计理念](docs/design-philosophy.md)了解动机，参见[架构覆盖矩阵](docs/architecture-coverage.md)了解验证边界。

## 构建并启动服务

从 workspace 构建服务程序：

```sh
cargo build --locked -p wgtime-service
```

默认情况下，`wgtime` 使用 `/run/wgtime`，该目录必须预先存在、归服务 UID 所有且权限为 `0700`。开发时可创建私有目录并显式指定：

```sh
install -d -m 700 /tmp/weightlang/wgtime
./target/debug/wgtime --test-directory /tmp/weightlang/wgtime
```

让服务在此终端持续运行。另开一个终端，将 Rust 客户端指向 Unix socket `service.sock`：

```sh
export WGTIME_ENDPOINT=/tmp/weightlang/wgtime/service.sock
```

服务只接受可选参数 `--test-directory DIRECTORY`。Rust 客户端连接现有服务，不会自动启动另一个模型执行核心。未设置 `WGTIME_ENDPOINT` 时，facade 使用默认地址 `/run/wgtime/service.sock`。

## 在 Rust 中使用 WGT

完整代码与中文注释导览见 [WGT 语法总览](language/syntax-tour/model.wgt)，其中明确区分现行语法与设计提案。

workspace package 名为 `wgt-model`，其 Rust library crate 路径为 `wgt`。它提供三种模型源码入口：`model!` 接收内联 Rust 字符串，`include_model!` 在构建时嵌入文件，`ModelDefinition::from_file` 在运行时解析已部署的源码文件。宏会在编译期校验 WGT 并嵌入源码；运行时还会重新解析，不会嵌入静态 IR。模型导入支持标准模块和文件模型的嵌套本地模块，详见[标准库](docs/standard-library.md)与[源码包说明](docs/packages.md)。

```toml
[dependencies]
wgt-model = { path = "../WeightLang/crates/wgt-model" }
```

```rust
use wgt::{model, Client, InstanceOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let definition = model!(r#"wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
"#);
    let client = Client::connect()?;
    let model = client.initialize(&definition, InstanceOptions::default())?;
    assert_eq!(model.forward(&[3.0])?, vec![6.0]);
    Ok(())
}
```

解析定义不会初始化模型、获取权重、训练或创建制品。初始化需要显式调用，默认不启用优化器。权重绑定、训练、保存和服务启动也都是独立的显式操作。Linux 上服务实例化模型目前需要可用的 `cc` 编译器来构建本机共享库；尚不能承诺服务部署无需编译器。当前限制见 [Rust 后端契约](docs/native-backend.md) 和 [WGT 源码包说明](docs/packages.md)。

## 项目范围

源码语言为 WGT 0.5。参见 [语言参考](docs/reference.zh-CN.md)、[原生训练](docs/native-training.md)、[标准库](docs/standard-library.md)、[存储约定](docs/storage.md)和[命令文档索引](docs/wgt.md)。`std/` 提供可组合的激活与损失计算模块；各模块的形状和梯度契约见标准库文档。完整模型和保留的实验结果放在 `weights/`。

[MIT 许可证](LICENSE)。
