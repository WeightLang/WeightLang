# 发布原生版本

[Release 工作流](../.github/workflows/release.yml)负责选择版本、构建 Linux 原生工具、提交版本变更并创建 GitHub Release。发布标签为 `wgt-vX.Y.Z`，与 `wgt self update` 使用的[原生工具链约定](toolchains.md#native-release-contract)一致。

## 从 GitHub 发布

1. 打开仓库 **Actions → Release → Run workflow**，选择 `main`。
2. 选择 `bump`，或在 `version` 填入目标版本。
3. 保持 `publish` 勾选并运行。需要先验证构建时，取消勾选；此时制品保存在该次运行的 Artifacts 中。
4. 查看各任务结果。发布完成后，运行摘要会给出 Release 链接。

| 输入 | 含义 | 以 `0.1.2` 为起点的示例 |
| --- | --- | --- |
| `bump: patch` | 末段加一；默认选项 | `0.1.3` |
| `bump: minor` | 中段加一，末段归零 | `0.2.0` |
| `bump: major` | 首段加一，其余归零 | `1.0.0` |
| `bump: none` | 使用当前版本；发布时要求标签尚不存在 | `0.1.2` |
| `version` | 填写后覆盖 `bump` 的选择 | 填 `0.3.0` 即发布 `wgt-v0.3.0` |
| `publish` | 默认勾选；取消后只构建、测试和保存制品 | 不写远端分支、标签或 Release |

版本必须是完整的 `X.Y.Z`，不带 `v` 前缀、前导零或 `-rc.1` 等后缀，也不能低于当前版本。[版本脚本](../scripts/release_version.cjs)只更新 `crates/wgt/Cargo.toml` 的 `weightlang-wgt` 版本和 `Cargo.lock` 中对应记录；`wgt`、`weightc` 从同一个 package 获取版本。WGT 语言版本、服务 crate、编辑器和模型包的版本独立管理。

工作流使用 GitHub 自动提供的 `GITHUB_TOKEN`，仅发布任务声明 `contents: write`，无需额外配置 PAT。仓库或组织的 Actions 策略、默认分支规则和标签规则需要允许该机器人提交版本并创建标签。

## 构建与发布顺序

候选版本先在构建工作区生成。两个架构使用同一份候选 manifest 和锁文件，并核对 Git 源码树；工具链安装测试、服务启动测试、release 构建和打包全部成功后，发布任务才把版本提交与标签一起原子推送到远端。若 `bump: none` 且版本未变，则直接为测试过的提交创建标签。

发布任务先上传完整制品到草稿 Release，再公开发布并设为 Latest。源码归档包含 bump 后的文件，与二进制对应的源码树一致；发布说明记录最终提交 SHA。整个过程在同一次工作流中完成。

修改发布工作流、版本脚本或原生打包脚本的 `main` push 和 pull request 会自动执行构建验证。这些自动运行只生成候选制品；只有手动运行且勾选 `publish` 才会发布。手动取消 `publish` 时也可从其他分支验证。

## 制品与运行环境

工作流在 Ubuntu 24.04 的两个原生 runner 上构建，生成以下文件：

| 架构 | 可执行文件 |
| --- | --- |
| Linux x86_64 | `wgt-linux-x86_64`、`weightc-linux-x86_64`、`wgtime-linux-x86_64` |
| Linux ARM64 | `wgt-linux-aarch64`、`weightc-linux-aarch64`、`wgtime-linux-aarch64` |

每个 Release 还包含 `weightlang-X.Y.Z-source.tar.gz` 和覆盖全部制品的 `SHA256SUMS`。仅构建运行的完整制品位于 `native-release-X.Y.Z` artifact 中。

当前发布的 `wgtime` 为 CPU 服务。源码中的客户端和服务依赖 Linux 接口，工作流尚不构建 macOS 或 Windows。`wgtime` 需要显式启动；模型初始化需要系统提供 `cc`，详见 [Rust 后端契约](native-backend.md)。使用现有 `weightc compile` 命令时，还需要 Rust、Cargo、链接器、匹配版本源码及依赖缓存，路径选择见[编译器实现](../crates/wgt/src/script_compiler.rs)。

### 下载后使用

从同一个已发布 Release 下载本机架构的三个可执行文件、源码包和 `SHA256SUMS`，放到同一目录。下例中的 `0.1.3` 应替换为实际下载的版本：

```sh
release_version=0.1.3
release_arch=$(uname -m) # x86_64 或 aarch64
sha256sum --ignore-missing --check SHA256SUMS
mkdir -p "$HOME/.local/bin"
for tool in wgt weightc wgtime; do
    install -m 755 "$tool-linux-$release_arch" "$HOME/.local/bin/$tool"
done
export PATH="$HOME/.local/bin:$PATH"

tar -xzf "weightlang-$release_version-source.tar.gz"
cd "weightlang-$release_version"
cargo fetch --locked
export WGT_CONTROL_SOURCE="$PWD/crates/wgtime-control"
```

保留解压后的源码目录。`WGT_CONTROL_SOURCE` 应指向其中的 `crates/wgtime-control`，因为下载的编译器无法继续使用构建 runner 上的源码路径。

启动服务时使用归当前用户所有、权限为 `0700` 的私有目录：

```sh
install -d -m 700 /tmp/weightlang/wgtime
wgtime --test-directory /tmp/weightlang/wgtime
```

让服务在该终端持续运行。在运行 Rust 客户端的另一个终端中设置：

```sh
export WGTIME_ENDPOINT=/tmp/weightlang/wgtime/service.sock
```

服务只接受可选的 `--test-directory DIRECTORY`，不提供 `--version` 或 `--help`。Rust 使用示例见[项目说明](../README.zh-CN.md#在-rust-中使用-wgt)。

## 失败后处理

| 失败情况 | 处理方法 |
| --- | --- |
| 版本格式无效、版本倒退或 manifest/lock 不一致 | 修正输入或仓库版本记录后重新运行。 |
| 目标标签已存在 | 选择一个新的版本；已有正式发布不会被覆盖。 |
| 测试、构建或打包失败 | 修复失败项后重新运行；此时远端版本和标签尚未修改。 |
| 构建期间默认分支有新提交 | 从最新 `main` 启动一次新的工作流。 |
| 分支或标签规则拒绝推送 | 按仓库规则处理发布权限；原子推送不会只更新其中一个 ref。 |
| 标签已推送，但上传或公开 Release 失败 | 对原运行选择 **Re-run failed jobs**；发布任务核对已有标签的源码树，再补全草稿。已公开的 Release 保留原制品。 |

重跑发布任务需要原运行的 artifacts 仍然可用。若标签已创建，不要用同一版本重新启动完整工作流；准备任务会拒绝占用已有标签。
