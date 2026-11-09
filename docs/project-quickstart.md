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
| `tiny-lm` | 小型 GPT 风格语言模型，附 `data.txt` | PyTorch + Transformers，自动选择设备 |
| `mlp` | 多层感知机，附数值回归数据 `data.csv` | PyTorch，自动选择设备 |
| `native` | 纯模型定义 `model.wgt` | Rust API + `wgtime` 服务 |

`wgt init --interactive` 会询问模板；直接按回车选择小型语言模型。`wgt init` 或 `wgt new NAME` 不带模板选项时仍创建原生项目。`init` 可初始化现有目录，遇到将要写入的同名文件会停止；`new` 要求目标目录不存在。

如果希望创建项目后立即准备环境，可运行 `wgt init my-model --template tiny-lm --sync`。只执行 `init` 或 `wgt check` 时不会安装依赖、初始化模型或开始训练。

## 2. 在 wgt.toml 中修改架构和训练参数

`wgt.toml` 集中保存项目、模型配方、运行设备、训练和推理参数。Python 模板同时生成 `pyproject.toml`，让 uv 继续使用标准 Python 项目配置；Rust 应用则继续使用 Cargo。

| 要配置什么 | 编辑哪个文件 |
| --- | --- |
| WGT 项目、模型配方、设备、训练、推理、Python runner、WGT 模型包依赖 | `wgt.toml` |
| Python 版本要求、Python 库依赖、uv 安装源 | `pyproject.toml` |
| Rust crate 依赖和构建设置 | Rust 应用的 `Cargo.toml` |

下面是小型语言模型的 `wgt.toml` 示例。编辑模板已有表中的值，不要重复添加同名表。

```toml
[project]
module = "local/my-model"
edition = "0.5"

[python]
runner = "workflow.py"

[model]
kind = "tiny-lm"
hidden_size = 64
layers = 2
heads = 4
context_length = 128
dropout = 0.0

[runtime]
device = "auto"

[train]
data = "data.txt"
steps = 100
batch_size = 4
learning_rate = 0.003
weight_decay = 0.01
seed = 42
output = "weights/model"
log_every = 10

[infer]
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
| 使用自己的文本 | `train.data` | UTF-8 文本文件；相对路径从配置目录开始 |
| 生成多少内容 | `infer.max_new_tokens` | 本次生成的最大 token 数 |

小型语言模型按 UTF-8 字节编码文本，因此这里的一个 token 是一个字节；中文字符会占多个 token。模型、词表和演示文本都从本地创建，训练流程无需下载预训练模型或 tokenizer。

运行 `wgt check` 可先检查 TOML 配置。参数取值顺序为显式命令行参数、当前 WGT 配置、模板对可选项的默认值；必填配置仍须保留。短期试验可以直接覆盖参数：

```sh
wgt train --steps 20 --device cpu
wgt infer --prompt "Hello" --max-new-tokens 16 --temperature 0.7
```

`python.runner`、`train.data`、`train.output` 和 `infer.checkpoint` 中的相对路径都以所选配置文件所在目录为基准。从其他目录执行 `wgt -C my-model train` 也会读取该项目的数据，并将结果写入该项目的输出目录。

增大层数、宽度、上下文和批量都会增加计算或内存需求。先使用默认配置完成一次训练，再逐项调整。模板演示的是可运行的架构和训练流程；高质量语言能力还需要合适的数据、模型规模与训练投入。

## 3. 使用 uv 管理 Python 和库

Python 库直接使用 uv 原生的依赖格式。`pyproject.toml` 的 `[project]` 表包含 Python 项目名、版本、`requires-python` 和 `dependencies`，可通过命令修改：

```sh
uv add numpy datasets
uv remove datasets
uv python pin 3.12
wgt sync
wgt run -- python -c "import torch; print(torch.__version__)"
```

`uv python pin` 写入 `.python-version`，uv 在运行项目时使用这个版本。也可在创建时用 `wgt init my-model --template tiny-lm --python 3.12` 选择版本。需要提前安装解释器时，直接运行 `uv python install 3.12`。版本选择、下载和环境创建遵循 [uv 的 Python 管理规则](https://docs.astral.sh/uv/concepts/python-versions/)。

`wgt sync` 复用 `uv sync`，`wgt train`、`wgt infer` 和 `wgt run` 复用 `uv run`。`uv add` 等命令可直接使用同一个项目。`wgt.toml` 中的 `[python].runner` 指向模板中的 `workflow.py`，其中训练和推理代码可按需求修改。

提交 `wgt.toml`、`pyproject.toml`、`uv.lock` 和 `.python-version`，保存模型设置、依赖锁和 Python 选择。环境目录 `.venv/` 已被忽略。环境准备完成后，可使用以下命令要求严格沿用锁文件：

```sh
wgt --locked sync
wgt --offline --locked sync
```

离线模式要求 Python 和依赖已在本机可用；`--offline` 限制包管理网络访问，用户自己的 Python 程序仍自行决定是否联网。锁文件语义遵循 [uv 的锁定和同步规则](https://docs.astral.sh/uv/concepts/projects/sync/)。

模板默认在 `pyproject.toml` 中启用 `[tool.uv].torch-backend = "auto"`，在 `wgt.toml` 中启用 `[runtime].device = "auto"`。首次使用 `wgt sync` 或 `wgt train` 准备环境时，WeightLang 复用 [uv 的 PyTorch 后端探测](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)，把兼容的安装源记录到 `pyproject.toml` 的 `[tool.uv.sources]` 和 `[[tool.uv.index]]`，随后由 uv 正常锁定和安装依赖。之后仍可直接使用 uv 的项目命令；只有 `uv pip` 提供后端探测，直接执行首次 `uv sync` 不会进行这一步。

训练和推理使用 PyTorch 检查设备，优先选择 CUDA／ROCm、其次 XPU、再其次 MPS，均不可用时使用 CPU。实际选择写入终端的诊断输出，推理结果仍单独输出。这里按兼容性和可用性选择加速设备，不会先运行性能竞赛；小模型未必在 GPU 上更快。省略 `runtime` 表或 `device` 字段也采用 `auto`。

需要固定运算设备时，设置 `runtime.device`，或使用 `--device cpu`、`--device cuda:0`、`--device xpu:0`、`--device mps` 覆盖。显式设备不可用或编号越界会报错。需要固定安装构建时，设置 `[tool.uv].torch-backend`，例如 `"cpu"`，再运行 `wgt sync`；uv 的 `UV_TORCH_BACKEND` 环境变量同样有效。用户手动配置的包来源会被保留。

更换机器或 GPU 后运行 `wgt sync` 可重新选择自动管理的安装源；`--locked` 和已有源的离线模式沿用已记录的选择。旧模板若写死了 CPU，要同时移除原 `pytorch-cpu` 来源配置、启用上述两个 `auto` 设置，再运行 `wgt sync`。设备加速需要本机有兼容驱动；目前实际验收使用 CPU，GPU 分支通过模拟可用性检查验证。

## 4. 保存模型并再次使用

训练默认写入 `weights/model/`，包含模型权重、架构、训练摘要和当次配置快照。新项目保存 `wgt.toml` 与 `pyproject.toml`，旧格式项目保存其 `pyproject.toml`；`weightlang.json` 记录命令行覆盖后实际使用的模型、训练和设备参数。

`wgt infer` 从 `infer.checkpoint` 读取保存的模型，架构以保存时的配置为准。后来修改 `wgt.toml` 的层数或宽度，会影响下一次新训练。

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

原生模型的训练和执行使用 [Rust API 与 `wgtime` 服务](../README.zh-CN.md#在-rust-中使用-wgt)。需要在 Python 中运行 `model.wgt` 时，使用 [Python SDK](python-sdk.md) 的 `weightlang.load("model.wgt", device="auto")`。SDK 复用 Rust 前端导出受支持的纯前向计算图，构造标准 PyTorch 模型；数据、损失函数和训练循环由 Python host 提供。

SDK 使用调用时传入的模型路径和设备参数，不隐式读取项目的 `train`、`infer`、`runtime` 配置或 `WGT_CONFIG_FILE`。自定义 host 若使用 `wgt.toml`，需要自行读取配置，并将相应值传给 SDK、优化器和推理代码。

`tiny-lm` 和 `mlp` 入门模板仍使用 `wgt.toml` 中的模型配方与 Python 代码。[模板实现](../crates/wgt/templates/workflow.py)分别复用 Transformers 的 GPT-2 和 `torch.nn.Sequential`，没有切换为 SDK 源码编译流程。修改这些模板的架构应编辑 `wgt.toml` 或其 `workflow.py`。当前 `wgt train` 和 `wgt infer` 调用 Python runner；统一的 Rust/Python 训练调度尚未实现。

Python 库使用 `pyproject.toml` 的 `[project].dependencies` 与 `uv.lock`；WGT 模型包使用 `wgt.toml` 的 `[dependencies]` 与 `wgt.sum`。两类依赖的命令和格式见[包管理说明](packages.md#项目配置与-python-依赖)。

## 6. 迁移旧项目配置

旧项目可以继续使用 `pyproject.toml` 中完整的 `[tool.wgt]` 配置，无需立即迁移。配置按整份文件选择：`wgt.toml`、旧 `wgt.mod`、带 `[tool.wgt]` 的 `pyproject.toml` 只能存在一个 WGT 配置来源；同时提供两个会报错。普通 Python/uv `pyproject.toml` 可与 `wgt.toml` 共存。

希望改用统一的 `wgt.toml` 时，手动完成以下步骤：

1. 把整个 `[tool.wgt]` 配置移到同目录的 `wgt.toml`，去掉表名的 `tool.wgt.` 前缀。例如 `[tool.wgt.project]` 改为 `[project]`，`[tool.wgt.model]` 改为 `[model]`，`[tool.wgt.python]` 改为 `[python]`；训练、推理、运行设备、模型包依赖及其他 WGT 子表一并移动。原 `[tool.wgt]` 表自身的键放在 `wgt.toml` 根部。
2. 从 `pyproject.toml` 删除已移动的 `[tool.wgt]` 及其子表，保留原 Python `[project]`、`[tool.uv]`、构建与其他 Python 工具设置。已有 `uv.lock`、数据和模型输出继续保留。
3. 更新 runner 的配置读取逻辑。旧模板或自定义 runner 若硬编码读取 `pyproject.toml` 的 `[tool.wgt]`，只搬配置会使它无法读取新设置。可在另一个目录生成同类新模板，参考其中 `workflow.py` 的配置读取部分；自定义训练代码继续保留。
4. 执行 `wgt check` 检查配置。需要开始新实验时，先选择新的 `train.output`，再运行 `wgt train`；使用新结果推理时相应更新 `infer.checkpoint`。

CLI 通过 `WGT_CONFIG_FILE` 向 `wgt train`、`wgt infer` 和 `wgt run` 启动的进程传入所选配置的绝对路径。runner 应读取该文件：`wgt.toml` 直接读取根表，旧 `pyproject.toml` 读取 `[tool.wgt]`，相对数据和输出路径从该配置目录解析。新版模板直接通过 Python 运行时，从脚本目录向上寻找最近的唯一 WGT 配置；从其他工作目录启动不会改变配置来源。
