# WeightLang

[English](README.md)

> **优雅地描述神经网络的内部架构。**

## 安装

在仓库根目录一次安装 uv、Rust/Cargo、`wgt` 和 `weightc`：

```sh
sh install.sh
. "${XDG_CONFIG_HOME:-$HOME/.config}/weightlang/env"
```

Windows 运行 `powershell -ExecutionPolicy Bypass -File .\install.ps1`，然后打开新终端。
安装器复用已有的可用工具，从官方来源补齐缺失工具，并配置用户 PATH；全新 Rust 安装使用 stable 最小工具链。
源码编译还需要系统链接器：macOS 使用 Xcode Command Line Tools，Linux 使用 C/C++ 构建工具链，Windows 使用包含 C++ 工作负载和 Windows SDK 的 Visual Studio Build Tools。
自定义路径、版本 tag 与 shell 配置见[安装说明](docs/weightc-migration.md#安装)。

## 使用 cargo generate 创建项目

```sh
cargo install --locked cargo-generate
cargo generate --git https://github.com/WeightLang/WeightLang.git
```

交互选择 `native`、`mlp` 或 `tiny-lm`，也可以直接指定：

```sh
cargo generate --git https://github.com/WeightLang/WeightLang.git templates/tiny-lm --name my-model
cd my-model
wgt train
wgt infer --prompt "Hello"
```

生成项目只需要 cargo-generate；训练前需安装 `wgt` 和 uv。`native` 生成用于 Rust/service 集成的 WGT 模型定义。离线生成及维护方法见[项目快速上手](docs/project-quickstart.md#cargo-generate)。

## 在 PyTorch 中导入 WGT 架构

用 `model.wgt` 描述参数和前向计算，Python 加载后得到标准 `torch.nn.Module`。SDK 复用 Rust 编译器、PyTorch FX、自动求导、优化器及模型查看工具。

```sh
uv venv
uv pip install -e './python[visualize]'
uv run --no-project python python/examples/mlp/train.py
```

```python
import torch
import weightlang as wgt

model = wgt.load("model.wgt", device="auto")
input_tensor = torch.zeros(1, *model.input_shape, device=next(model.parameters()).device)
prediction = model(input_tensor)
print(wgt.summary(model))
wgt.visualize(model, "architecture.svg")
wgt.save(model, "weights/demo/model.pt")
```

Python 负责数据、训练循环、推理和可视化；已保存的 SDK 模型或预编译 JSON 可在无需编译器、无需模型服务的环境加载。SDK 支持张量程序、控制流、可学习架构参数与事务式 typed 操作，保留原生初始化和子模型参数身份，并使用原生后端做差分验收。SVG/PNG 需要系统 Graphviz，Mermaid/DOT 仅需 PyTorch。当前版本从源码安装，完整用法见 [Python SDK 指南](docs/python-sdk.md)。原生 Rust/service 路径见下文。

SDK 加载使用 host 传入的模型路径和设备参数，不会自动读取项目的 `train`、`infer` 或 `runtime` 设置。需要使用 `wgt.toml` 的 host 应自行读取配置，再将相应参数传给 SDK 和 PyTorch。

## 开始训练第一个模型

选一个模板，修改 `wgt.toml` 参数，就可以训练并试用模型。Python 环境和依赖交给 uv，优化计算复用 PyTorch，小型语言模型复用 Transformers。这些模板通过 TOML 与 Python 构造模型；WGT 源码加载使用上面的 SDK 入口。

完成上面的安装后，创建并训练项目：

```sh
wgt init my-model --template tiny-lm
cd my-model
wgt train
wgt infer --prompt "Hello"
```

模板自带文本数据，默认自动选择可用的 GPU 加速，无可用 GPU 时使用 CPU。首次训练由 uv 选择兼容的 PyTorch 构建并准备 `.venv` 和 `uv.lock`，无需手动激活环境。训练输出保存在 `weights/model/`。

模型尺寸、训练、推理和设备参数集中在 `wgt.toml`。Python 版本、库依赖和 uv 设置保留在 `pyproject.toml`；Rust 应用使用 `Cargo.toml` 管理 Rust 依赖。两个 PyTorch 模板会生成 WGT 和 Python 各自的配置文件，调整模板提供的模型参数时只需编辑 `wgt.toml`。

| 模板 | 用途 | 修改位置 |
| --- | --- | --- |
| `tiny-lm` | 用本地文本训练一个小型 GPT 风格语言模型 | `wgt.toml`、`data.txt` |
| `mlp` | 从小型 CSV 数据集学习数值关系 | `wgt.toml`、`data.csv` |
| `native` | 为 Rust/service API 定义 WGT 模型 | `model.wgt`、`wgt.toml` |

`wgt templates` 列出模板，`wgt init --interactive` 提供交互选择；不带选项的 `wgt init` 默认使用原生模板。语言模型模板用于学习完整流程，实际语言能力取决于数据和训练投入。配置方法、环境管理和执行边界见[项目快速上手](docs/project-quickstart.md)。

旧项目仍可整份使用 `pyproject.toml` 中的 `[tool.wgt]` 配置。改用 `wgt.toml` 时，按[迁移步骤](docs/project-quickstart.md#6-迁移旧项目配置)移动配置并更新 runner 的读取逻辑。

## 原生 WGT 模型

WGT 描述模型内部的参数、状态，以及计算和更新它们的操作。Rust 负责数据准备、I/O、调度以及训练或搜索循环；唯一的 `wgtime` 服务拥有可变模型实例。可复用算法通过标准库显式导入。

前向计算、loss 与 backprop 是一种可选的模型路径。模型也可以使用有状态操作，或接收外部生成的参数更新。操作文档所列的 core 状态回滚、无梯度搜索、真实 service 操作和 CNN 回归范围已通过本地验收。typed 操作不支持自动微分（AD）；既有 forward 路径保留显式训练策略。这些能力已在本地实现，但尚未发布；边界与验收范围见[多范式模型操作](docs/model-operations.md)。

**可学习架构参数（Learnable Architecture Parameters）**让模型在明确约束下声明数值与结构开关。以下声明已在本地实现。旧 `train` 路径对整数和布尔读取使用 STE（直通估计器）；既有 forward 路径保留显式训练策略，`learn` 本身不绑定到某种策略：

```wgt
learn depth: int in 1..=12 = 6;
learn use_skip: bool = true;
```

可通过 `depth_gate` 或有界 `repeat` 表达受约束的计算，也可用布尔参数控制残差连接。`repeat` 已在文档所述边界内本地实现并验证；`choice` 和动态形状变化仍属于设计方向。完整浮点约束、离散训练规则及权重保存方式见[可学习架构参数与已测示例](docs/learnable-architecture-parameters.md)。

参见[设计理念](docs/design-philosophy.md)了解动机，参见[架构覆盖矩阵](docs/architecture-coverage.md)了解验证边界；[训练策略与模型生命周期](docs/training-strategies-and-lifecycle.md)说明自定义权重更新、STE、Checkpoint 和工程化路线。

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

在 Rust 应用的 `Cargo.toml` 中声明 crate 依赖：

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
