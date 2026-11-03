# 源码分发包与显式续训（当前源码，尚未发布）

`build` 只检查并打包源码，不分配模型权重、不访问训练数据、不执行 Python 回调。
`train` 默认延续权重管理系统 `weights/.revisions` 的 HEAD，并提交新版本；没有基线就报错。
`train --fresh` 才显式初始化。继续权重不等于恢复优化器：默认不恢复 Adam 动量、
学习率调度器、随机状态或数据进度；这些训练检查点由 Python 任务管理。

## 命令与迁移

| 命令 | 当前职责 |
| --- | --- |
| `wgt check` | 静态检查；不训练 |
| `wgt build [-o dist/model.wgtpkg] [--entry model.wgt] [--force]` | 生成源码 ZIP 包 |
| `wgt train [model.wgt] [--fresh] [--store PATH] [--from REV]` | 从当前/指定权重版本继续训练；显式 fresh 才初始化 |
| `wgt run / wgt test` | 读取当前权重 HEAD；缺失或源码不匹配时报错，不隐式训练 |
| `wgt export [model.wgt]` | 显式构建初始化/迁移后的独立模型目录，不训练 |
| `weightc export model.wgt` | 显式初始化/迁移并提交权重版本；`--no-store` 选择独立目录 |
| `wgt package --weights model.wgt=DIR` | 打包指定的完整模型制品 |
| `wgt package --reference model.wgt=DIR` | 只记录该制品的文件大小和 SHA256，不复制权重 |
| `wgt install dist/model.wgtpkg -o installed/model` | 校验并解包到新目录；不执行代码、不下载权重 |
| `wgt release-assets --version vX.Y.Z` | 旧版 GitHub 多资产发布协议，保持与现有 registry 兼容 |

`weightc build/package [PROJECT]` 与原生命令使用相同 ZIP 实现；默认 PROJECT 为当前目录。
权重选择可重复，每项使用包内相对源码路径与完整制品目录的映射；同一模型不能同时
选择 embedded 和 external。`build` 不读取 `[model].output` 作为源码包输出位置。
`export` 的模型目录覆盖仍要求 `--force`；安装目标必须不存在。

旧 `build` 的训练行为移至显式 `weightc legacy-build`，执行时打印弃用提示。
这个兼容入口保留旧选项和旧语义，不能用作新项目默认路径。
已有脚本将旧命令迁移到 `train`、`export` 或 `build`；旧选项传给新 build 会报错，
不会静默忽略。旧 `wgt package --version ...` 改用 `wgt release-assets`。
旧发布协议不因增加 ZIP 分发而自动支持上传/下载 `.wgtpkg`。

## 元数据标准 v1

人工描述放在 `wgt.toml` 的 `[package]`，模型入口放在 `[model].source`。
训练配置属于 `[train]`，不属于模型描述，且不会随加载隐式执行。

```toml
[project]
module = "local/demo"
edition = "0.3"

[model]
source = "model.wgt"

[package]
name = "demo"
version = "0.1.0"
description = "示例模型架构"
license = "MIT"
authors = ["Example"]
tags = ["classifier"]

[package.extensions."org.example"]
purpose = "experiment"

[train]
python = "train.py:train"

[train.config]
epochs = 10
```

`name/version/description/license` 为可选字符串，`authors/tags` 为可选字符串数组。
`extensions` 是表，其键必须含点，供组织命名空间扩展使用。未知标准字段、错误类型
会拒绝。未提供字段不自动推断为已训练或已验证。
`format_version=1` 是包格式版本，与模型的 WGT 语言版本、包版本独立。

包内布局：

```text
model.wgtpkg                     # 标准 ZIP，支持 ZIP64
  WGT-INFO/metadata.json         # 格式版本、入口、模型映射、描述、资产大小/摘要
  sources/model.wgt
  sources/lib/layers.wgt        # 保留目录，便于 use 的相对引用
  weights/<source-id>/          # 仅显式 --weights 才有
    manifest.json
    source.wgt
    weights.safetensors
```

所有项目内 `.wgt` 默认递归收集，排除隐藏目录、weights/checkpoints、缓存、虚拟环境、
target 和 dist；不扫描权重库，不自动收集 Python 文件、外部数据集、凭证或优化器检查点。已有内联 data 行仍属于源码内容。
有多个模型时要求显式入口或 `[model].source`；只有库的包可以无入口。
外部本地源码依赖必须先纳入项目树。当前不自动打包远程包依赖或动态 Python 后端。
需要构建时权重 import/inherit/derive 的模型必须显式选择已有制品；不能假装是独立源码包。
包记录当前标准库文件摘要，加载时拒绝标准库漂移，需用目标运行时重新打包。

元数据 `files` 为归档成员路径到 `{sha256,size}` 的映射，不包含元数据文件自身；
`models` 为源码相对路径到模型名的映射；`entry` 为默认源码相对路径或 null；
`weights` 按源码路径记录 `mode=embedded|external`、包内目标路径及三件套文件摘要。
external 不记录本机绝对路径，消费者显式提供制品目录，不自动联网。
包内摘要用于完整性检查，不构成作者签名或可信来源证明。
训练状态、来源与冻结规则以权重制品 manifest 为准；人工描述不能覆盖这些事实。

safetensors 使用 ZIP STORED，不重复压缩；其他内容使用 Deflate。复制与校验按块读取，
不把整个权重载入内存。安装检查重复成员、路径穿越、符号链接、库存、大小及摘要，
失败不会留下半安装目录。写入前检查剩余空间；操作者仍需确认目标位于磁盘而非 tmpfs。
打包和安装会临时占用接近一份目标产物的空间；embedded 安装同时保留压缩包与解包权重。

## 加载

```python
import weightlang

# 先 wgt install，源码包得到初始化模型；不会训练或保存权重。
model = weightlang.load("installed/model")
# 多入口包可以显式选择。
model = weightlang.load("installed/model", entry="model.wgt")
# 外部引用包必须提供已匹配的完整制品，缺失或摘要变化就失败。
model = weightlang.load("installed/model", external_weights="weights/demo")
```

`load(source, *, device="cpu", placements=None, entry=None, external_weights=None)`：
后两个参数只适用于安装后的分发包；单独 `.wgt` 和原生模型目录保留原有加载语义。
当前不直接加载压缩文件，不自动解压/下载，不支持 pip 安装 `.wgtpkg`。

## Python 训练任务

`[train].python` 使用相对于项目根目录的 `FILE.py:FUNCTION`（默认函数 train）。
工具先从权重库装载基线并验证模型/形状/权重兼容性，然后调用 `train(ctx)`。
`ctx.model` 是可训练模块；`ctx.source` 是源码绝对路径；`ctx.revision` 是基线版本
（fresh 时为 None）；`ctx.config` 是 `[train.config]` 字典。
返回指标字典或 None；返回后校验结构、冻结权重与数值，再提交到现有版本管理系统。
回调失败不推进 HEAD；并发修改 HEAD 会拒绝提交。默认保留策略继续生效。

```python
import torch

def train(ctx):
    # 示例针对 f32[2] -> f32[1]；batch 在 Python 侧，权重由 ctx 提供。
    x = torch.tensor([[0., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.]])
    device = next(ctx.model.parameters()).device
    x, y = x.to(device), y.to(device)
    optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=0.01)
    for _ in range(ctx.config.get("epochs", 10)):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(ctx.model(x), y)
        loss.backward()
        optimizer.step()
    return {"loss": loss.item()}
```

新任务不需要 WGT `data/train`。旧声明暂保留解析兼容；没有 `[train]` 时，显式
`train` 可以运行旧声明。`load`、`build`、`export` 均不执行它们。
续训拒绝架构/接口变化、替换权重的源码指令和静默解冻；先显式迁移，再继续训练。
`--from` 选择训练基线，不切换权重库 HEAD；新提交父版本仍是运行前的当前 HEAD。
`--fresh` 不清空历史、不覆盖旧制品；它明确创建随机初始化基础上的新版本。
