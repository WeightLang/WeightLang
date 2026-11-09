# 原生工具链与可选 Python SDK

`wgt` 和 `weightc` 使用 Rust 原生工具链，源码检查、编译和原生执行保持独立于 Python。
可选的 [Python SDK](python-sdk.md) 位于 `python/`，通过 `wgt compile --target torch`
复用原生前端，将 WGT 模型架构加载为 PyTorch 模型。SDK 提供普通 Python API，
训练循环与数据处理由 Python 宿主负责。`wgt init` 的 PyTorch 模板另由 uv 管理环境。

## 安装

```sh
cargo install --locked --force --path crates/wgt --bin weightc --bin wgt
# 等价本地安装/更新：
WEIGHTLANG_SOURCE="$PWD" sh install.sh
```

Windows 使用 `install.ps1`。两个安装器需要 Rust/Cargo，默认安装到 Cargo bin 目录，
遵循 Cargo 自身的安装路径配置。仓库外默认从 GitHub main 构建，
`WEIGHTLANG_VERSION` 指定 Git tag，必须包含原生工具源码。
旧 Python/Torch/uv 安装选项已删除。重复安装显式替换这两个程序，不修改模型目录。

旧虚拟环境、缓存和 PATH 中的 Python 启动器不会被扫描删除；若还残留旧启动器，
用 `command -v weightc` 或 `Get-Command weightc` 找到它，并将 Cargo bin 排在前面。
不要误删权重和实验结果。

## 更新与工具链锁定

`wgt update [--version X.Y.Z]`、`wgt self update` 下载并验证成对原生程序后，
切换最近的 `.wgt-version`，没有项目锁时更新缓存内的默认选择。
`wgt download X.Y.Z` 只缓存，不切换。离线更新必须指定已完整缓存的版本。
`weightc` 与 `wgt` 读取同一个选择；版本探测不会递归调度。

发行版 tag 必须为 `wgt-vX.Y.Z`，资产包括：

- `wgt-<os>-<arch>` 与 `weightc-<os>-<arch>`；Windows 追加 `.exe`。
- `SHA256SUMS` 中每个资产必须恰好有一个有效校验项。
- `--version` 必须分别报告 `wgt X.Y.Z` 和 `weightc X.Y.Z`。

只包含旧 wheel、单个 wgt、预发布或草稿的发行版不进入 latest 候选。
下载、校验或版本探测失败不更新选择。安装先暂存成对程序，再发布缓存目录。
已有损坏/不完整缓存不会被静默覆盖，须明确处理对应缓存目录后重装。
`wgt toolchain X.Y.Z --path /path/to/wgt` 要求旁边存在同版本 `weightc`。

## 发布打包

```sh
cargo build --locked --release -p weightlang-wgt --bins
sh scripts/package_wgt.sh --binary target/release/wgt --output dist/native-release
```

可用 `--weightc PATH` 指定编译器。Windows 对应
`scripts/package_wgt.ps1 -Binary target/release/wgt.exe -Output dist/native-release`，
可选 `-Weightc PATH`。拒绝覆盖已有输出，验证版本一致，写入两份程序和 SHA256SUMS。
这些脚本只打包，不创建 Git tag、不上传、不发布。

## 当前执行入口

| 入口 | 行为 |
| --- | --- |
| `weightc check SOURCE` | 解析当前原生 `wgt 0.5` 脚本，不执行，不做完整类型检查 |
| `weightc compile SOURCE -o EXECUTABLE` | 通过 Rust/Cargo 生成本机程序，不自动训练或保存权重 |
| `weightc legacy-build SOURCE [-o DIRECTORY] [--max-operations N]` | main 返回模型句柄，保存完整原生模型；N 默认 10000000 且须正数 |
| `weightc build` | 原生 legacy-build 别名，不是历史源码 ZIP 打包命令 |
| `wgt compile SOURCE --target torch [-o JSON]` | 为 Python SDK 导出顺序图、forward 或 typed operations 的模型 IR，不执行计算 |
| `wgt run -- COMMAND ARGS` | 在项目的 uv 环境中运行宿主命令 |
| `wgt sync` | 同步模型包与已配置的 uv 环境；`--no-env` 仅同步模型包 |

原生构建默认输出 `weights/<source stem>/`，已有目录和符号链接一律拒绝覆盖。
制品包含 `model.json`、`checkpoint.json`、`source.wgt`、`manifest.json`。
`wgt package --entry DIRECTORY -o NEW_DIRECTORY` 校验并复制原生目录制品；
`wgt export DIRECTORY_OR_MODEL_JSON -o NEW_MODEL_JSON` 导出推理模型，不执行源码或训练。
这两个命令不支持旧 ZIP/源码初始化语义，详见[原生制品命令](native-artifact-commands.md)。
`wgt check` 检查源码或项目配置，`wgt train`、`wgt infer` 显式运行已配置的 Python
host。其他扩展按当前 `wgt --help` 与 PATH 插件规则处理；原生模型测试使用 Cargo，
详见[原生测试](native-testing.md)。当前 SDK 的接口与验收边界以
[Python SDK 指南](python-sdk.md)和对应测试为准。
