<p align="center">
  <img src="https://raw.githubusercontent.com/WeightLang/WeightLang/main/assets/logo.png" alt="WeightLang" width="148">
</p>

<h1 align="center">WeightLang</h1>

<p align="center">
  <b>用一门语言构建神经网络，并分发、复用它们的权重。</b>
</p>

<p align="center">
  <a href="https://github.com/WeightLang/WeightLang/releases/latest"><img alt="release" src="https://img.shields.io/github/v/release/WeightLang/WeightLang?label=weightc"></a>
  <a href="https://pypi.org/project/weightlang/"><img alt="PyPI" src="https://img.shields.io/pypi/v/weightlang"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="backend" src="https://img.shields.io/badge/backend-PyTorch-ee4c2c">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-green"></a>
  <img alt="status" src="https://img.shields.io/badge/status-0.2%20prototype-orange">
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#语言">语言</a> ·
  <a href="#标准库">标准库</a> ·
  <a href="#复用权重">复用权重</a> ·
  <a href="#包管理">包管理</a> ·
  <a href="#文档">文档</a> ·
  <a href="#限制">限制</a>
</p>

---

在 `.wgt` 文件里描述网络，`weightc` 用 PyTorch 构建、初始化、训练或导入权重，
最后生成一个**模型制品（artifact）**目录。它包含权重，以及加载和运行模型所需的
网络结构、输入输出约定、版本、哈希和每个张量的来源。制品可以发布为包，
像 Go 模块一样通过 GitHub 路径引用，让其他程序复用其中的权重。

WeightLang 主要面向 AI 编程模型，因此语法小而严格，以行为单位，也方便人审阅。

**Bootstrap 目标**：用 WeightLang 训练能理解、编写和调试本语言的专用语言模型，
保留现代指令、多轮对话和工具调用能力；对外只要求 Open Responses 协议（兼容 OpenAI Responses API）。
这是待完成的目标，能力与协议验收见 [bootstrap 约定](docs/bootstrap.md)。
首个[本地训练与文本接口实验](examples/bootstrap/README.md) 已跑通，但尚未通过语言能力验收。

```text
 xor.wgt ──weightc legacy-build --no-store──▶ weights/xor_mlp/ ──weightc run──▶ {"prob": [[0.00003], [0.9999], ...]}
                              ├─ manifest.json        结构、输入输出、来源、哈希
                              ├─ weights.safetensors  float32 张量，不使用 pickle
                              └─ source.wgt           构建时的原始源码
```

## 安装

**Linux / macOS**

```sh
curl -fsSL https://github.com/WeightLang/WeightLang/releases/latest/download/install.sh | sh
```

**Windows（PowerShell）**

```powershell
irm https://github.com/WeightLang/WeightLang/releases/latest/download/install.ps1 | iex
```

两个脚本都会从 GitHub Release 下载 `weightlang` wheel，并根据 `SHA256SUMS`
校验哈希。如果已安装 [uv](https://docs.astral.sh/uv/)，就使用 `uv tool` 安装；
否则创建独立虚拟环境，不需要 sudo 或管理员权限。Linux 默认安装 CPU 版 PyTorch，
设置 `WEIGHTLANG_TORCH=default` 可使用 PyPI 上的 CUDA 版本。
用 `WEIGHTLANG_VERSION=v0.2.0` 固定版本。执行前可以先阅读脚本：
[install.sh](install.sh)、[install.ps1](install.ps1)。

<details>
<summary><b>其他安装方式</b></summary>

```sh
# 用 uv 直接安装 Release 中的 wheel
uv tool install https://github.com/WeightLang/WeightLang/releases/download/v0.2.0/weightlang-0.2.0-py3-none-any.whl

# 从 PyPI 安装
uv tool install weightlang        # 也可以用 pipx install weightlang
# Linux 上安装 CPU 版 PyTorch，下载量更小
uv tool install --index https://download.pytorch.org/whl/cpu weightlang

# 从源码安装，用于开发
git clone https://github.com/WeightLang/WeightLang && cd WeightLang
python -m venv .venv && .venv/bin/pip install -e '.[test]'
```

如果系统已经安装 PyTorch，可以复用：

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install --no-build-isolation --no-deps -e .
```

</details>

依赖：Python ≥ 3.10、PyTorch ≥ 2.2、safetensors ≥ 0.4。

## 快速开始

源码版默认记录权重版本：`weightc legacy-build model.wgt -m "说明"`。下列独立模型示例使用
`--no-store` 保留原有输出目录；详见 [权重版本控制](docs/weight-version-control.md)。

以下命令在仓库根目录执行：

```sh
weightc check   examples/xor.wgt               # 解析和类型检查，不构建
weightc legacy-build --no-store   examples/xor.wgt -o weights      # 初始化 → 训练 → weights/xor_mlp/
weightc run     weights/xor_mlp --input '[[0,0],[0,1],[1,0],[1,1]]'
weightc inspect weights/xor_mlp                  # 查看输入输出和每个张量的来源
weightc verify  weights/xor_mlp                  # 重新校验哈希、形状和可加载性
```

参考输出：

```text
built weights/xor_mlp
  weights sha256 7969227a9c0e3528317e1cd22b8da766e5b1a7a705e0ba96078a86cde8296ba2
  tensors trained=4
  training loss 0.7066 -> 0.0001 on 4 rows (xor), train accuracy 1.00
```

在 Python 中使用：

```python
from weightlang import load_artifact

model = load_artifact("weights/xor_mlp")      # 校验哈希和形状
model.predict([[0, 1], [1, 1]])             # tensor([[0.9999], [0.0001]])
model.module                               # 普通的 torch.nn.Sequential
```

不安装 WeightLang，也可以只用 `torch` 和 `safetensors` 加载制品，
见 [制品格式文档](docs/artifact.md)。

也可以在 Python 中训练 `model.module`，再用
`model.save("weights/finetuned", training={"method": "Adam"})`
保存到新制品目录，保留权重来源并更新哈希。训练记录由调用者提供；
不提供记录时，变化的权重标为 `edited`。详见 [Python 互操作](docs/python.md)。

| 命令 | 用途 |
|---|---|
| `weightc check FILE` | 解析并检查模型或库文件 |
| `weightc legacy-build --no-store FILE -o DIR` | 构建制品，输出到 `DIR/<model>/` |
| `weightc run ARTIFACT --input JSON` | 运行推理，输入输出均为 JSON |
| `weightc inspect ARTIFACT [--json]` | 查看接口、网络层和张量来源 |
| `weightc std` | 列出内置标准库的 block 名称和参数 |
| `weightc audit ARTIFACT` | 检查标准权重的准入条件 |
| `weightc verify ARTIFACT` | 校验哈希、张量形状和可加载性 |
| `weightc get [DIR]` | 下载、校验并锁定 `wgt.mod` 中的依赖 |
| `weightc publish PKG --version vX.Y.Z` | 构建包的发布文件 |

## 语言

```wgt
wgt 0.2
use std/nn
use std/heads
model std_mlp
input x: f32[2]
output logits: f32[3]
layer body: nn.mlp(in=2, hidden=8, out=4)
layer head: heads.classifier(in=4, classes=3)
data train:
  [-2, 0] -> [0]
  [2, 0] -> [1]
  [0, 2] -> [2]
init seed=42
train data=train loss=cross_entropy optimizer=adam lr=0.03 epochs=200
```

- 每行一条语句，以关键字开头，用 `key=value` 指定选项，没有通用表达式语法。
- 运行前检查张量形状，报错包含文件名和行号，例如 `xor.wgt:9: shape mismatch: ...`。
- **每个张量的来源都记录在 manifest 中**：

| origin | 含义 |
|---|---|
| `random-init` | 使用固定种子初始化，从未训练；不能把依赖它的输出当成有效预测 |
| `trained` | 初始化后，由当前程序训练 |
| `imported` | 从其他制品原样复制，原来的来源记录在 `upstream_origin` 中 |
| `imported+trained` | 导入后继续训练 |

没有 `train` 的程序也能构建；如果包含未训练权重，CLI 和 Python 加载器会提示。
导入未训练张量不会让它变成已训练张量；导入已训练权重则可以直接用于推理。
指定 `validate=` 后，训练集与验证集指标分开记录。

完整语法见 [语言文档](docs/language.md)。

当前源码的 `wgt 0.3` 支持命名轴、`linear(out=...)` 输入宽度推导、
`rearrange` 重排与 `reduce` 归约。见 [带注释的维度教程](docs/dimensions.md)，
运行 `weightc check examples/dimensions.wgt --shapes` 可逐层查看形状。

## 标准库

`use std/nn`、`use std/heads` 和 `use std/blocks` 提供多层感知机（MLP）、
分类或回归输出层，以及先做归一化再计算的残差块。残差块将输入与内部网络的输出相加。

这些 block 在编译时展开为具名网络层，**不包含预训练权重**。
运行 `weightc std` 查看名称和参数。
示例见 [examples/residual.wgt](examples/residual.wgt)，
详细说明见 [标准库文档](docs/stdlib.md)。

## 复用权重

从其他制品导入一个网络层，可以固定权重哈希，也可以冻结该层，让它不参与后续训练：

```wgt
init   seed=7
import hidden from "../weights/xor_mlp" layer=hidden freeze
train  data=xnor loss=bce optimizer=adam lr=0.05 epochs=300
```

[examples/xnor_reuse.wgt](examples/xnor_reuse.wgt) 导入 `xor_mlp` 已训练的隐藏层，
冻结它，只训练新的 XNOR 输出层。测试会检查导入张量与源张量逐位相同、
训练期间保持冻结、来源哈希正确记录，以及 `sha256=` 固定哈希和形状检查确实生效。

这个例子展示的是复用机制。只有四行训练数据，不能证明复用能提高学习效果；
从头训练一个网络同样可以拟合 XNOR。

## 包管理

包由 GitHub 路径标识，版本对应 Git tag，权重作为 GitHub Release 附件分发，
不写入 Git 历史。

```text
wgt.mod   require github.com/WeightLang/WeightLang/packages/gates v0.1.0
xnor.wgt  import hidden from "github.com/WeightLang/WeightLang/packages/gates/xor_mlp" layer=hidden freeze
wgt.sum   提交号，以及索引、manifest、权重和源码的 SHA256；应提交此锁文件
```

```sh
weightc get   examples/remote_reuse                  # 联网下载、校验并写入 wgt.sum
weightc legacy-build --no-store examples/remote_reuse/xnor.wgt -o weights # 离线构建，根据 wgt.sum 校验缓存
```

`weightc get` 将 tag 解析为提交号，根据 Release 的 `wgt.index.json` 检查各文件大小
和 SHA256，并要求制品中的 `source.wgt` 与仓库**对应提交**中的源码一致，
然后验证制品。如果以后 tag 被移动或附件被替换，与 `wgt.sum` 不一致就会报错。

现有包：

- [gates v0.2.0](https://github.com/WeightLang/WeightLang/releases/tag/packages/gates/v0.2.0)：
  六种双输入逻辑门，在各自完整的四行真值表上均预测正确。
- [iris v0.1.0](https://github.com/WeightLang/WeightLang/releases/tag/packages/iris/v0.1.0)：
  小型鸢尾花分类器，独立测试集上答对 29/30。
  数据、许可证、指标更正和限制见 [packages/iris](packages/iris)。
- [standard_reuse](examples/standard_reuse) 演示新包的导入，包含已提交的锁文件。
- 上面的 `remote_reuse` 示例仍固定使用 gates v0.1.0，用于验证兼容性。

发布自己的包：

```sh
weightc publish path/to/pkg --version v0.1.0 -o dist   # 构建文件并打印 tag、Release 发布命令
```

设计、锁文件格式和已知限制见 [包管理文档](docs/packages.md)。

## 文档

- [工程层级与目录规范](docs/project-layout.md)：工程、构件库、构件、模型、层、模型制品和权重包。
- [完整工程示例](examples/complete_project/README.md)：拆分源码、数据、训练、推理与行为测试。

以下详细文档目前为英文：

| 文档 | 内容 |
|---|---|
| [docs/language.md](docs/language.md) | 语法、检查规则、权重来源 |
| [docs/artifact.md](docs/artifact.md) | 制品文件、manifest 字段、校验、不依赖 WeightLang 的加载方式 |
| [docs/packages.md](docs/packages.md) | 模块路径、wgt.mod、wgt.sum、发布、下载和待定设计 |
| [docs/stdlib.md](docs/stdlib.md) | 内置算子、std block、标准权重和路线图 |

## 开发

```sh
.venv/bin/python -m pytest                                # 不联网
WEIGHTLANG_NETWORK_TESTS=1 .venv/bin/python -m pytest     # 额外验证真实 gates Release 的下载
ruff check src tests && ruff format --check src tests
```

测试覆盖语法解析和错误行号、XOR 训练、保存与重载后输出逐位一致、
重复构建的字节一致性、篡改检测、纯 PyTorch 加载、未训练权重标记、
标准库展开、残差计算、CSV 数据、训练与验证指标，以及权重导入和冻结。
包管理测试用模拟 GitHub 验证发布、下载、离线构建、附件篡改、tag 移动和源码不匹配；
联网测试则访问真实 Release。

### 发布状态

| 渠道 | 状态 |
|---|---|
| GitHub Release `v0.2.0`（wheel、源码包、SHA256SUMS、安装脚本） | 已发布 |
| [PyPI weightlang](https://pypi.org/project/weightlang/) | 通过手动触发的 [publish-pypi](.github/workflows/publish-pypi.yml) 工作流发布，文件与 GitHub Release 一致 |

## 限制

以下描述已发布的 0.2；main 分支的新架构支持见后文。

- **语言规模小**：支持嵌套顺序网络和残差网络，单个一维 f32 输入与输出；
  不支持 embedding、i64、卷积、注意力或通用计算图。
- **导入粒度**：只能导入完整网络层或嵌套 block，不能导入张量切片。
  包管理只支持直接依赖和 GitHub Release；首次获取时信任并记录哈希，
  尚无签名验证。使用者下载的是发布者预先构建的权重。
- **可复现性**：同一 PyTorch 版本和 CPU 环境下可重复得到相同字节；
  跨版本或硬件的结果可能不同，应通过哈希核对。
- **运行时**：目前仅支持 PyTorch，没有 ONNX 等导出格式。
- **稳定性**：仍处于 0.x 阶段，语法和格式可能变化。Windows 安装脚本尚未在 Windows 上实测。

## 致谢与引用

感谢 **Rikiya Takehi、Jayden Teoh、Alex Zhang 和 Omar Khattab** 的文章
[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)。
WeightLang 明确引用这篇文章，作为权重读取、差异比较和评估工作流的灵感来源。
本项目独立开发；目前 gates 和 Iris 示例仍使用梯度训练，尚未复现文章中的 LLM 直接编辑权重实验。

## main 分支新增：Rust 工具与权重检查

**以下功能尚未包含在已发布的 weightc 0.2.0 wheel 中。**
独立 Rust 工具 `wgt` 提供类似 Cargo 的项目和包管理流程：
`new/init`、`add/remove/update`、`fetch`、`tree/metadata`、
`weights/run/test/check`、`verify`、`package/publish`、
`graph`、`cache`、`doctor`，并支持 PATH 中的 `wgt-子命令` 插件。

```sh
cargo install --path crates/wgt --locked
wgt new demo
wgt -C demo check
wgt -C demo build
wgt -C demo train --fresh
wgt -C demo test
(cd demo && weightc revision checkout HEAD -o weights/demo)
wgt -C demo graph weights/demo --open
```

包管理、制品校验和架构渲染用 Rust 实现，不依赖 Python。
`weights/run/test/check` 调用现有的 `weightc`，编译与推理仍需要 Python/PyTorch。
架构图支持 HTML、SVG、Markdown、JSON、TOON、Mermaid、DOT，
保留嵌套 block、残差跳连、尺寸、参数量和权重来源。
安装、依赖管理和扩展方式见 [wgt 文档](docs/wgt.md)。

此外新增 `weightc weights`、`weightc diff` 和 `weightc evaluate --compare`：
导出可读权重、精确统计改动的参数，在同一份有哈希的数据上比较候选与基线。
详见 [权重实验流程及引用说明](docs/weight-space.md)。

### main 上新增 Transformer 和 CNN

尚未发布的 **wgt 0.3** 支持固定长度序列、整数 token、Embedding、可学习位置编码、
多头自注意力（双向或因果）、卷积和池化。`std/transformer` 和 `std/vision`
加入后，共有 **5 个库、21 个 block**。它们是架构积木，不是预训练模型。

```sh
# 需要安装源码版编译器；已发布的 0.2.0 wheel 尚不支持这些示例
python -m pip install -e .
weightc legacy-build --no-store examples/transformer.wgt -o weights
wgt graph weights/tiny_transformer --open
weightc legacy-build --no-store examples/cnn.wgt -o weights
wgt graph weights/tiny_cnn --open
```

两个示例都在四行合成数据上达到 100% **训练集**准确率，仅用于验证流程，
不代表真实任务或验证集效果。当前没有 tokenizer、预训练模型导入、KV cache、
padding mask 或交叉注意力。形状及算子规则见 [语言文档](docs/language.md)。

main 已支持直接编辑权重：weightc patch-template 生成绑定基线哈希的 JSON，
weightc patch 写出独立候选制品，记录 edited 来源并清除失效指标。
运行 [完整示例](examples/weight_space/README.md)，即可体验不训练、直接写入
y = 2x + 1 的参数，再比较和评估结果。详见 [补丁指南](docs/weight-space.md)。

## 许可证

代码采用 [MIT](LICENSE) 许可证。Iris 数据和训练权重的 CC BY 4.0 许可及署名要求，
见 [packages/iris/README.md](packages/iris/README.md)。

## 输出目录

模型默认保存到 `weights/<模型名>/`，用 `-o ROOT` 修改输出根目录。已有模型默认不覆盖，重建需显式传 `--force`。大型临时工作放磁盘上的 `.cache/weightlang/`，小型预览放 `/tmp/weightlang/`。详见[目录规范](docs/storage.md)。

## 消融实验

源码版支持 `weightc ablate weights/my_model --suite ablation.json`。
可屏蔽层输出、跳过层或清零指定权重；每个变体从同一基线独立运行，报告损失、准确率及变化量，保留配置和数据哈希，不改原模型。
目前是推理时消融，不包含重新训练。配置见[消融实验](docs/ablation.md)。

## Git 包注册表

源码版 `wgt` 支持从 [WeightLang/registry](https://github.com/WeightLang/registry) 添加已审核的模型包：

```sh
cargo install --git https://github.com/WeightLang/WeightLang --locked weightlang-wgt
wgt add weightlang/gates
```

默认选择最新收录版本，随后锁定精确版本、来源和文件哈希。内置库可直接
`import std/nn`（wgt 0.3），无需 add。发布者运行
`wgt publish --index dist/wgt.index.json --license MIT --edition 0.3`，上传制品后
自动向注册表提交 PR；维护者审核后才可被其他用户发现。详见[注册表文档](docs/registry-design.md)。

## Source bundles and explicit training (current source)

`wgt build` creates one `.wgtpkg` ZIP without training or initializing weights.
`wgt train` continues the current weight revision; `--fresh` explicitly starts from initialization.
`wgt package --weights model.wgt=DIR` embeds selected weights; `--reference` records hashes only.
Metadata lives in `[package]` in `wgt.toml`. See [bundle and migration specification](docs/bundles.md).
Older build examples below/above use the explicit compatibility command `weightc legacy-build`;
standalone initialization is `wgt export`. These changes are not yet released.

### 一键升级

```sh
wgt update --all   # 升级所有模型包依赖，重新锁定精确版本和哈希
wgt self update   # 下载、校验并切换到本平台最新稳定版 wgt
```

原有 `wgt update` 自更新用法继续兼容。批量升级保留各包的注册来源，全部校验
成功后才修改项目文件；不更改 Python 环境。详见[升级规则](docs/toolchains.md)。
