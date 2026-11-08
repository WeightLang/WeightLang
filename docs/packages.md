# 包与 GitHub Release 附件

包的核心是源码，权重可选。每个模型的入口名为 `model.wgt`，多个模型使用子目录区分。
当前本地实现使用 `weightlang.package-index` v2，将源码和独立权重附件列在同一份
`wgt.index.json` 中；实际分发只使用 GitHub Releases 附件，不需要新的 registry。
本页描述本地实现，不表示某个版本已经发布，也不表示新的 Rust crate、网络 API 或下载流程已经发布或经过网络验证。

## 项目配置与 Python 依赖

原生项目使用 `wgt.toml`，兼容已有 `wgt.mod`。可选的 PyTorch 模板使用标准
`pyproject.toml`：Python 版本要求和 Python 库放在 `[project]`，WeightLang 的模型、
训练、推理及模型包依赖配置放在 `[tool.wgt]`。同一项目只使用一份 WeightLang manifest；
不要同时放置 `wgt.toml`、`wgt.mod` 或带 `[tool.wgt]` 的 `pyproject.toml`。

Python 库由 `uv add`、`uv remove` 和 `uv sync` 管理，版本锁定在 `uv.lock`；
WGT 模型包由 `wgt add`、`wgt remove` 和 `wgt fetch` 管理，锁定在 `wgt.sum`。
在 PyTorch 项目中，模型包依赖写入 `[tool.wgt.dependencies]`，不会改写
`[project].dependencies`。`wgt sync` 同步这两类依赖，`wgt train`、`wgt infer`
和 `wgt run` 通过 uv 使用项目 Python 环境。完整例子见[项目快速上手](project-quickstart.md)。

## Rust 应用与 WGT 模型的入口

Rust 主程序可通过 `wgt-model` facade crate 使用 WGT 模型定义，crate 名为 `wgt-model`，Rust 路径为 `wgt`。`model!` / `include_model!` 字符串宏会在编译期解析并校验模型源码，再将源码嵌入产物；运行时源码加载入口执行纯解析。编译期嵌入和运行时加载属于三种入口中的两种。宏嵌入的是 WGT 源码，不是静态 IR；加载或解析架构不会自动获取权重、训练或创建训练产物。

首期实现的三种入口都统一拒绝 WGT `import`。Rust crate 依赖是应用的编译依赖，与 WGT 模型源码包及其模块依赖不同；声明 crate 依赖不会自动取得模型源码包。未来若支持 WGT 模块导入，运行时加载模式才需要部署入口源码及其导入模块；当前 API 不加载带有 WGT imports 的源码包。模型需要的外部数据仍须由使用者另行准备。

权重仍是可选的独立附件，须由调用方明确提供并按运行时契约绑定；训练须显式执行。Rust API 与对应发布流程仍在实现中；以下 `wgt release-assets`、`wgt fetch` 和 `wgt verify` 章节描述现有本地 WGT 包工具的行为，不表示新 Rust crate 已发布或网络端到端验证已经完成。

## 本地准备制品

在已提交、工作目录干净的包目录执行：

```sh
wgt release-assets --version v0.5.0 --output dist
```

命令只准备本地文件，不创建 Release、不上传、不执行源码或训练。包的 module 必须是
`github.com/OWNER/REPO[/SUBDIR]`，并匹配仓库中的目录。目标目录须不存在或为空。
源码包可完全没有权重；它不声明模型已训练。旧的 `--allow-untrained` 会给出迁移诊断。

打包器从 Git 跟踪文件中寻找 `model.wgt` 和 `operators NAME` 库，递归收集相对导入的
模块，保留完整相对路径，并包含受跟踪的 `wgt.toml` 或 `wgt.mod` 及 `wgt.sum`。
相对模块导入必须以 `./` 开头、留在入口目录下且已跟踪；缺失模块、符号链接和越界路径报错。
不自动收集数据、credentials、训练输出，也不进入 `weights/`、`.cache/`、`target/` 或 `dist/`。
源码中读取的外部数据须由使用者另行准备。包依赖通过 manifest 和锁声明，打包不下载依赖。

## 显式准备权重附件

`--artifacts DIR` 显式指定附件输入目录。该目录必须包含 `weights.json`，以变体名映射到
safetensors 文件及其关联的模型源码路径：

```json
{
  "baseline": {
    "file": "baseline.safetensors",
    "model": "classifier/model.wgt"
  }
}
```

```sh
wgt release-assets --version v0.5.0 --artifacts weights/release-input --output dist
```

变体名使用字母、数字和下划线，首字符不能是数字。`file` 是该输入目录内的相对路径；
`model` 必须指向本次包内的 `model.wgt`。只读取清单指定的文件，不扫描其他模型结果。
目前单附件大小上限为 256 MiB，参数必须是有限 F32；打包调用 runtime 的 safetensors
检查器读取排序后的参数名、shape 和 dtype，不执行模型。超过上限明确报错，不写入 tmpfs。

索引记录每个附件的文件名、字节数、SHA256、参数契约及关联的源码路径。
参数契约不包含注释或源文件字节哈希，因此修改源码注释不会改变该契约；源码自身仍按字节锁定。
此处尚无架构兼容哈希，`compatibility: parameter-schema-only` 明确表示只记录张量契约。
相同参数形状不能证明计算图、轴含义或模型质量相同；实际绑定必须由 runtime 对显式模型验证。
清单使用 `training: unspecified`，不从文件名或存在权重推断训练已完成。

权重文件不包含图结构、优化器状态和训练进度。`model.json` 是推理模型 IR，
`checkpoint.json` 是训练断点，均不能作为此处的 safetensors 附件。
本地完整模型和训练断点继续按[存储规范](storage.md)保留，不能因打包被当作缓存清理。

## 获取、锁定与离线校验

```sh
# 默认仅获取源码
wgt fetch
# 仅选择指定依赖的一个明确权重变体，可重复 --weights
wgt fetch --weights github.com/OWNER/REPO/SUBDIR=baseline
# 复用已锁定的源码与显式选择过的权重
wgt --offline --locked fetch
```

版本必须精确为 `vMAJOR.MINOR.PATCH`。下载来源为对应 GitHub Release 的
`wgt.index.json` 和附件；索引、tag commit、源码及所选权重 SHA256 进入 `wgt.sum`。
指定不存在的变体或依赖会报错，不会自动选择其他变体。后续 fetch 保留锁中已有的权重选择；
`--locked` 禁止新增或更改锁项，离线选择要求对应权重已锁定并存在于缓存。

下载先在磁盘包缓存的独立目录中验证大小、SHA256、模块路径与参数契约，再整体安装；
失败保留已有缓存。`wgt verify` 校验锁与缓存，字节哈希只能检测内容变化，不能证明语义兼容、
训练过程或质量。检查与编译不隐式下载或训练；源码包的安装也不运行其中的 main。

`dependency(module)` 返回包缓存根目录，源码位于索引保留的相对路径，已选择的权重位于
`weights/VARIANT.safetensors`。算子库仍通过 `import alias from "MODULE/LibraryName"`
显式导入，详见[原生自定义算子](custom-operators.md)。模型加载与显式绑定权重的契约由
[原生训练](native-training.md)及[语言参考](reference.zh-CN.md)定义。
