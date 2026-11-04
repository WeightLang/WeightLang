# WeightLang

[English](README.md)

> **优雅地描述神经网络的内部架构。**

WGT 描述模型的内部架构与计算，让张量流、参数共享、模块组合和状态转移清晰可见。Rust 控制数据准备、I/O、调度与训练循环。模型执行以同一份 WGT 定义和 `wgtime` 服务为目标。

**可学习架构参数（Learnable Architecture Parameters）**让模型在明确约束下学习数值与结构开关。以下声明已在本地实现：整数和布尔参数在前向使用离散值，训练时通过 STE（直通估计器）传递代理梯度。

```wgt
learn depth: int in 1..=12 = 6;
learn use_skip: bool = true;
```

通过 `depth_gate` 与显式有界计算组合，可以训练有效深度；布尔参数可以控制残差连接的门控。自动 `repeat`、改变张量形状和 `choice` 仍属于设计方向。完整浮点约束、离散训练规则及权重保存方式见[可学习架构参数与已测示例](docs/learnable-architecture-parameters.md)。

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
