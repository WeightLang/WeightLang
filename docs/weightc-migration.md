# 原生工具链与可选 Python SDK

`wgt` 和 `weightc` 使用 Rust 原生工具链，源码检查、编译和原生执行保持独立于 Python。
可选的 [Python SDK](python-sdk.md) 位于 `python/`，通过 `wgt compile --target torch`
复用原生前端，将 WGT 模型架构加载为 PyTorch 模型。SDK 提供普通 Python API，
训练循环与数据处理由 Python 宿主负责。`wgt init` 的 PyTorch 模板另由 uv 管理环境。

## 安装

在仓库根目录运行一次安装器，准备 uv、Rust/Cargo、`weightc` 和 `wgt`：

```sh
sh install.sh
. "${XDG_CONFIG_HOME:-$HOME/.config}/weightlang/env"
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

脚本会检查工具能否实际运行，复用已有工具；缺少 uv 时使用 [uv 官方安装器](https://docs.astral.sh/uv/reference/installer/)，缺少 Rust/Cargo 时使用 [rustup](https://rust-lang.github.io/rustup/installation/other.html) 安装 stable 最小工具链。仅有 rustup、尚未设置默认工具链时会补齐默认值。安装完成会输出 uv、Cargo、rustc、wgt 和 weightc 的版本；下载、构建或版本检查失败时返回非零状态。

Linux/macOS 的共享环境文件为 `${XDG_CONFIG_HOME:-$HOME/.config}/weightlang/env`。安装器在 `.profile`、`.bashrc`、已有的 Bash 登录配置和 `${ZDOTDIR:-$HOME}/.zshenv` 中加入一次加载语句，并生成 fish 的 `conf.d/weightlang.fish`。重复运行不会重复添加配置。当前 POSIX 终端执行上面的 `.` 命令即可生效；fish 打开新终端，或执行 `source ~/.config/fish/conf.d/weightlang.fish`（设置 XDG_CONFIG_HOME 时使用对应路径）。Windows 更新当前安装进程和用户环境变量，运行上述子进程命令后打开新终端；直接在当前 PowerShell 执行 `& .\install.ps1` 时，该会话的 PATH 也会更新。

| 环境变量 | 作用与默认值 |
| --- | --- |
| `CARGO_HOME` | Cargo 工具与缓存目录，默认 `~/.cargo` |
| `RUSTUP_HOME` | Rust 工具链目录，默认 `~/.rustup` |
| `CARGO_INSTALL_ROOT` | `wgt`、`weightc` 安装根目录，默认使用 `CARGO_HOME`；安装器显式传入 `--root`，并把其 `bin` 目录加入 PATH |
| `UV_INSTALL_DIR` | 新装 uv 的目录；默认 `~/.local/bin`，Linux/macOS 也遵循 `XDG_BIN_HOME`、`XDG_DATA_HOME` |
| `WEIGHTLANG_SOURCE` | 优先从指定本地源码目录的 `crates/wgt` 构建 |
| `WEIGHTLANG_VERSION` | 在未选择本地源码时使用指定 Git tag，例如 `wgt-v0.5.0` |

未设置 `WEIGHTLANG_SOURCE` 时，安装器优先使用当前目录下的源码；仓库外默认从 GitHub main 构建。重复安装会更新两个原生命令。编译缓存放在磁盘上的用户缓存目录，也可用 `CARGO_TARGET_DIR` 指定已有缓存。

源码构建需要系统链接器：Linux 使用 C/C++ 构建工具链，macOS 使用 Xcode Command Line Tools，Windows 使用包含 C++ 工作负载和 Windows SDK 的 Visual Studio Build Tools。已有 Rust 过旧时，可先运行 `rustup update stable`，并检查 `rustup show` 的实际工具链选择。uv 管理项目所需的 Python 和依赖；Python SDK 按[SDK 安装步骤](python-sdk.md#安装)装入项目环境。

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
