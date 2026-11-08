# 项目快速上手

WeightLang 的模板把创建项目、配置模型、训练和推理连成一个流程。先用自带数据跑通，再修改架构参数或换成自己的数据。

## 1. 选择模板并训练

本页对应当前仓库源码。在仓库根目录运行以下安装命令；Python 模板还需要安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```sh
cargo install --locked --force --path crates/wgt --bin wgt --bin weightc
wgt templates
wgt init my-model --template tiny-lm
cd my-model
wgt train
wgt infer --prompt "Hello"
```

首次 `wgt train` 会让 uv 选择 Python、创建项目内的 `.venv`、安装依赖并生成 `uv.lock`，随后开始训练。无需手动激活虚拟环境。模板包含演示数据，训练过程使用本地文件；安装 Python 和依赖时需要网络。

| 模板 | 模型与数据 | 默认运行方式 |
| --- | --- | --- |
| `tiny-lm` | 小型 GPT 风格语言模型，附 `data.txt` | PyTorch + Transformers，CPU |
| `mlp` | 多层感知机，附数值回归数据 `data.csv` | PyTorch，CPU |
| `native` | 纯模型定义 `model.wgt` | Rust API + `wgtime` 服务 |

`wgt init --interactive` 会询问模板；直接按回车选择小型语言模型。`wgt init` 或 `wgt new NAME` 不带模板选项时仍创建原生项目。`init` 可初始化现有目录，遇到将要写入的同名文件会停止；`new` 要求目标目录不存在。

如果希望创建项目后立即准备环境，可运行 `wgt init my-model --template tiny-lm --sync`。只执行 `init` 或 `wgt check` 时不会安装依赖、初始化模型或开始训练。

## 2. 在一份 TOML 中修改架构和训练参数

Python 模板的配置文件是标准 `pyproject.toml`。uv 读取 `[project]` 与 `[tool.uv]`，WeightLang 读取 `[tool.wgt]`；依赖解析、Python 安装和锁文件都由 uv 负责。

下面是小型语言模型的主要配置。编辑已有表中的值，不要重复添加同名表。

```toml
[tool.wgt.model]
kind = "tiny-lm"
hidden_size = 64
layers = 2
heads = 4
context_length = 128
dropout = 0.0

[tool.wgt.runtime]
device = "cpu"

[tool.wgt.train]
data = "data.txt"
steps = 100
batch_size = 4
learning_rate = 0.003
weight_decay = 0.01
seed = 42
output = "weights/model"
log_every = 10

[tool.wgt.infer]
checkpoint = "weights/model"
prompt = "Hello"
max_new_tokens = 32
temperature = 0.8
top_k = 40
seed = 42
```

| 想改变什么 | 修改哪个参数 | 含义 |
| --- | --- | --- |
| 模型有多少层 | `model.layers` | Transformer 块的数量 |
| 每层有多宽 | `model.hidden_size` | 每个 token 的内部向量长度 |
| 同时关注多少组信息 | `model.heads` | 注意力头数；必须整除 `hidden_size` |
| 一次能看多少内容 | `model.context_length` | 最大上下文 token 数 |
| 训练多久 | `train.steps` | 优化器更新次数 |
| 一步使用多少样本 | `train.batch_size` | 每次更新的样本数 |
| 每次更新幅度 | `train.learning_rate` | 优化器学习率 |
| 使用自己的文本 | `train.data` | 项目中的 UTF-8 文本文件 |
| 生成多少内容 | `infer.max_new_tokens` | 本次生成的最大 token 数 |

小型语言模型按 UTF-8 字节编码文本，因此这里的一个 token 是一个字节；中文字符会占多个 token。模型、词表和演示文本都从本地创建，训练流程无需下载预训练模型或 tokenizer。

运行 `wgt check` 可先检查 TOML 配置。短期试验也可以直接覆盖参数：

```sh
wgt train --steps 20 --device cpu
wgt infer --prompt "Hello" --max-new-tokens 16 --temperature 0.7
```

增大层数、宽度、上下文和批量都会增加计算或内存需求。先使用默认 CPU 配置完成一次训练，再逐项调整。模板演示的是可运行的架构和训练流程；高质量语言能力还需要合适的数据、模型规模与训练投入。

## 3. 使用 uv 管理 Python 和库

Python 库直接使用 uv 原生的依赖格式。已有的 `[project]` 表包含项目名、版本、`requires-python` 和 `dependencies`，可通过命令修改：

```sh
uv add numpy datasets
uv remove datasets
uv python pin 3.12
wgt sync
wgt run -- python -c "import torch; print(torch.__version__)"
```

`uv python pin` 写入 `.python-version`，uv 在运行项目时使用这个版本。也可在创建时用 `wgt init my-model --template tiny-lm --python 3.12` 选择版本。需要提前安装解释器时，直接运行 `uv python install 3.12`。版本选择、下载和环境创建遵循 [uv 的 Python 管理规则](https://docs.astral.sh/uv/concepts/python-versions/)。

`wgt sync` 复用 `uv sync`，`wgt train`、`wgt infer` 和 `wgt run` 复用 `uv run`。`uv add` 等命令可直接使用同一个项目；没有另一套 Python 依赖清单或解析器。`[tool.wgt.python].runner` 指向模板中的 `workflow.py`，其中训练和推理代码可按需求修改。

提交 `pyproject.toml`、`uv.lock` 和 `.python-version`，让其他机器使用相同的依赖与 Python 选择。环境目录 `.venv/` 已被忽略。环境准备完成后，可使用以下命令要求严格沿用锁文件：

```sh
wgt --locked sync
wgt --offline --locked sync
```

离线模式要求 Python 和依赖已在本机可用；`--offline` 限制包管理网络访问，用户自己的 Python 程序仍自行决定是否联网。锁文件语义遵循 [uv 的锁定和同步规则](https://docs.astral.sh/uv/concepts/projects/sync/)。

模板默认使用 CPU 版 PyTorch 索引。需要 CUDA 等加速时，按 [uv 官方 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/) 修改 `[tool.uv.sources]` 与 `[[tool.uv.index]]`，同步后再把 `runtime.device` 改为相应设备。环境管理复用 uv；设备可用性由 PyTorch 和本机驱动决定。

## 4. 保存模型并再次使用

训练默认写入 `weights/model/`，包含模型权重、架构、训练摘要和当次 TOML 配置。`wgt infer` 从 `infer.checkpoint` 读取保存的模型，架构以保存时的配置为准。后来修改 TOML 的层数或宽度，会影响下一次新训练。

默认训练会拒绝覆盖已有输出。比较多次实验时，修改 `train.output` 并把 `infer.checkpoint` 指向对应目录；确实要替换已识别的模板输出时，显式执行：

```sh
wgt train --force
```

这些输出保存推理所需的模型，没有保存优化器和训练进度；当前模板从头开始新训练。完整模型与保留的实验结果继续放在 `weights/`，按[存储约定](storage.md)保存。

数值模型也使用同一流程：

```sh
wgt init numbers --template mlp
cd numbers
wgt train
wgt infer --inputs 0.5
```

MLP 的 CSV 不含表头，每行先写 `model.input_size` 个输入值，再写 `model.output_size` 个目标值。默认示例学习 `y = 2x + 1`，可先输入 `0.5` 查看模型接近 `2.0` 的程度。

## 5. 使用原生 WGT 模型

原生模板保留 `model.wgt` 作为真实模型源码，`wgt.toml` 保存项目和模型包配置：

```sh
wgt init native-model --template native
cd native-model
wgt check
```

原生模型的训练和执行使用 [Rust API 与 `wgtime` 服务](../README.zh-CN.md#在-rust-中使用-wgt)。Python 模板使用 TOML 配置驱动的 PyTorch 架构；[模板实现](../crates/wgt/templates/workflow.py)中的 `tiny-lm` 复用 Transformers 的 GPT-2 实现，`mlp` 复用 `torch.nn.Sequential`。这一工作流尚未提供任意 WGT 源码到 PyTorch 的转换；修改 Python 模板的架构应编辑 TOML 或其 `workflow.py`。

需要直接把 `model.wgt` 导入 PyTorch 时，使用 [Python SDK](python-sdk.md) 的 `weightlang.load`。SDK 复用 Rust 前端导出纯前向计算图，得到标准 PyTorch 模型；上面的 `tiny-lm` 与 `mlp` 入门脚本仍按自己的 TOML/Python 架构运行。

Python 库使用 `[project].dependencies` 与 `uv.lock`；WGT 模型包使用 `[tool.wgt.dependencies]` 与 `wgt.sum`。两类依赖的命令和格式见[包管理说明](packages.md#项目配置与-python-依赖)。
