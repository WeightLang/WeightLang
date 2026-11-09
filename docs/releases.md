# 同步发布原生工具与 SDK

[Release 工作流](../.github/workflows/release.yml)从同一份源码树发布原生工具、Python/PyTorch SDK 和 Rust SDK。统一版本写在根 `Cargo.toml` 的 `[workspace.package]`，发布标签保留 `wgt-vX.Y.Z`，与 `wgt self update` 的[工具链约定](toolchains.md#native-release-contract)一致。

## 发布入口

1. 打开仓库 **Actions → Release → Run workflow**，选择 `main`。
2. 选择 `bump`，或在 `version` 填写完整目标版本。
3. 需要预演时取消勾选 `publish`；保持勾选会在全部验证通过后发布到 crates.io、PyPI 和 GitHub Releases。
4. 等待最后的 `publish` 任务成功，运行摘要会给出公开 Release 链接。

| 输入 | 含义 | 以 `0.2.0` 为起点 |
| --- | --- | --- |
| `bump: patch` | 末段加一，默认值 | `0.2.1` |
| `bump: minor` | 中段加一，末段归零 | `0.3.0` |
| `bump: major` | 首段加一，其余归零 | `1.0.0` |
| `bump: none` | 使用当前版本 | `0.2.0`；需要未使用的发布版本 |
| `version` | 指定版本，覆盖 `bump` | `0.3.2` → `wgt-v0.3.2` |
| `publish` | 默认勾选；取消后仅构建、测试、保存制品 | 不修改远端分支、标签或包仓库 |

版本必须是稳定的 `X.Y.Z`，不带 `v`、前导零或预发布后缀，且不能低于当前版本。版本脚本会同时核对全部 workspace crates、内部依赖、`Cargo.lock`、Python manifest 与运行时 `__version__`。发现不一致会在写入前退出。一次 bump 只修改四个文件：根 `Cargo.toml`、`Cargo.lock`、`python/pyproject.toml`、`python/weightlang/__init__.py`。WGT 源语言声明和编辑器版本仍按各自契约管理。

旧 Python `0.2.0` 已经发布；本次统一以 `0.2.0` 为源码基线，下一次默认发布为 `0.2.1`。不要用相同版本替换旧 Python 包。

## 首次配置包仓库

GitHub 发布使用自动提供的 `GITHUB_TOKEN`，无需 PAT。仓库分支与标签规则需要允许 Actions 机器人创建版本提交和标签。两个包仓库使用独立的 Actions environment，以便配置发布身份。

| 目的地 | 发布身份 | 配置值 |
| --- | --- | --- |
| PyPI `weightlang` | Trusted Publisher，或 `PYPI_API_TOKEN` secret | owner=`WeightLang`，repository=`WeightLang`，workflow=`release.yml`，environment=`pypi` |
| crates.io 的四个 SDK crates | 首次发布使用 `CARGO_REGISTRY_TOKEN` secret；已有包也可使用 Trusted Publishing | owner=`WeightLang`，repository=`WeightLang`，workflow=`release.yml`，environment=`crates-io` |

PyPI token 和 crates.io token 可配置在对应 environment 或仓库 secrets 中。不要把 token 放进源码、运行输入或日志。没有显式 token 时，工作流通过 OIDC 获取临时发布身份；两个任务都只在手动勾选 `publish` 时申请 `id-token: write`。

crates.io 的首次发布需要 API token，之后可为 `wgtime`、`wgt-language`、`wgt-model-macros`、`wgt-model` 分别登记 Trusted Publisher，再移除长期 token。账号必须拥有对应包名的发布权。配置步骤见 [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing)、[PyPI GitHub Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)、[uv 发布指南](https://docs.astral.sh/uv/guides/publish/)。

首次发布 Rust SDK 时，先完成 crates.io 账号的邮箱验证，创建允许发布上述四个新 crate 的 token（`publish-new` 权限）。在 GitHub **Settings → Environments → crates-io → Environment secrets** 中添加 `CARGO_REGISTRY_TOKEN`，也可使用仓库级同名 secret。token 只填入 GitHub secret；不写进运行输入、源码或问题讨论。四个 crate 首次发布成功后，才能逐个配置表中的 Trusted Publisher 身份。

PyPI 的 Trusted Publisher 匹配工作流文件名。旧 `publish-pypi.yml` 的发布身份不会自动授权 `release.yml`；需要在现有 `weightlang` 项目中添加表中的身份，或在 `pypi` environment 配置 `PYPI_API_TOKEN`。

## 发布门槛

准备任务先在工作区生成候选版本并记录 Git tree。原生构建、两个 SDK 工作流及 Rust 模型语义测试均下载这份候选文件，并核对源码树，避免不同任务构建不同版本。

手动发布通过默认分支、版本和标签检查后，`check-rust-publishing` 与构建并行执行。缺少 token 且任一 crate 尚不存在时，预检列出这些包并要求首次发布凭据；已有 crate 则实际尝试 OIDC 认证。网络和注册表错误会使预检失败。预检通过才允许 `stage-release` 写入版本提交、标签和草稿，已启动的构建仍可完成验证并保存制品。普通 push、PR 和 `publish=false` 不申请发布身份，也不依赖该预检。

配置了 token 时，预检只确认凭据存在，实际权限由 crates.io 在发布时校验。OIDC 预检获取的短期 token 随任务结束撤销；正式发布重新申请身份，不跨任务传递或保存 token。

| 验证 | 实际执行内容 |
| --- | --- |
| 版本与发布逻辑 | 版本选择、全部 manifest/锁文件一致性、首次发布凭据、注册表错误，以及已发布版本冲突、制品不同、部分成功重试的回归测试 |
| 原生工具 | Ubuntu 24.04 x86_64 / ARM64，工具链与服务启动测试，release 构建，`wgt` / `weightc` 版本检查 |
| Python SDK | Python 3.10 / 3.12，CPU PyTorch，sdist 构建 wheel，Twine 严格元数据检查，安装 wheel 后运行完整 SDK 测试与训练示例 |
| Python 独立安装 | 仓库外隔离导入、版本与类型标记、编译和推理、梯度、无编译器时恢复 checkpoint 与 IR |
| Rust SDK | Cargo 打包并独立编译四个 crate；检查标准库资源；从解包目录构建真实下游应用，连接同版本服务验证推理 |
| Rust 模型 | 原生指令、控制流、状态、训练、导出消费者和真实服务提交/回滚测试 |
| 发布预演 | 候选版本执行 `cargo publish --dry-run`，核对重打包字节与已测试 crate 完全相同 |

Python CI 安装实际 wheel，使用隔离模式与 pytest importlib 模式，避免源码目录遮蔽已安装包。Rust 的标准库和 prelude 在仓库中保持单一来源；Cargo 打包时将 crate 内的资源链接展开为普通文件，发布包能独立编译。Cargo 的多包发布负责按照依赖顺序上传四个 crate，参见 [Cargo package](https://doc.rust-lang.org/cargo/commands/cargo-package.html) 与 [Cargo publish](https://doc.rust-lang.org/cargo/commands/cargo-publish.html)。CI 使用 stable Rust。

相关工作流、发布脚本或版本清单的 push / pull request 会触发发布构建验证。普通 Python、Rust SDK 和 Rust 模型 CI 也保持独立运行；并发分组区分独立 CI 与 Release 调用，避免互相取消。自动运行均不发布。

## 制品与发布顺序

所有验证成功后，`assemble` 将下面的文件合入 `weightlang-release-X.Y.Z` artifact，并生成覆盖全部文件的 `SHA256SUMS`。

| 制品 | 文件 |
| --- | --- |
| Linux x86_64 原生工具 | `wgt-linux-x86_64`、`weightc-linux-x86_64`、`wgtime-linux-x86_64` |
| Linux ARM64 原生工具 | `wgt-linux-aarch64`、`weightc-linux-aarch64`、`wgtime-linux-aarch64` |
| Python SDK | `weightlang-X.Y.Z-py3-none-any.whl`、`weightlang-X.Y.Z.tar.gz` |
| Rust SDK | `wgtime-X.Y.Z.crate`、`wgt-language-X.Y.Z.crate`、`wgt-model-macros-X.Y.Z.crate`、`wgt-model-X.Y.Z.crate` |
| 完整源码 | `weightlang-X.Y.Z-source.tar.gz` |
| 完整性检查 | `SHA256SUMS` |

手动发布分四步完成：

1. 检查默认分支没有在构建期间移动，将经过测试的版本提交与标签原子推送，上传全部制品到草稿 Release。
2. 发布 Rust crates：先检查每个远端版本，再预演打包并比对测试制品，然后交给 Cargo 上传和等待索引可用。
3. 发布 Python wheel 与 sdist：直接使用测试过的制品，通过 `uv publish` 上传。
4. 两个包仓库均成功后，将完整 GitHub Release 公开并设为 Latest。

多个包仓库没有共同事务。Rust 已成功而 PyPI 失败时，Rust 包会保留，GitHub Release 保持草稿；重跑失败任务即可继续。Rust 仅跳过内容完全一致且未被撤回的已有版本，发现相同版本不同内容会失败；`uv publish` 对 Python 文件也按字节匹配处理重试。已经公开的 GitHub Release 不替换资产。重试机制见 [uv 发布指南](https://docs.astral.sh/uv/guides/publish/)。

## 安装同版本 SDK

正式发布后，可以安装对应版本。下例的 `0.2.1` 需要替换为已经公开的版本：

```sh
python -m pip install 'weightlang==0.2.1'
cargo add wgt-model@=0.2.1
```

Python 源模型编译需要同一 Release 的 `wgt`，可通过 `WGT_COMPILER` 指定路径；已保存的 SDK 模型可以在不调用编译器的情况下加载。Rust 模型执行需要同一 Release 的 `wgtime` 服务和系统 C 编译器。当前原生发布目标是 Linux x86_64 / ARM64，`wgtime` 使用 CPU 后端；完整接口见 [Python SDK](python-sdk.md) 与 [Rust 后端](native-backend.md)。

从同一 Release 下载本机的三个可执行文件、源码包和 `SHA256SUMS`，在下载目录执行：

```sh
release_version=0.2.1
release_arch=$(uname -m)
sha256sum --ignore-missing --check SHA256SUMS
mkdir -p "$HOME/.local/bin"
for tool in wgt weightc wgtime; do
    install -m 755 "$tool-linux-$release_arch" "$HOME/.local/bin/$tool"
done
export PATH="$HOME/.local/bin:$PATH"
export WGT_COMPILER="$HOME/.local/bin/wgt"

tar -xzf "weightlang-$release_version-source.tar.gz"
cd "weightlang-$release_version"
cargo fetch --locked
export WGT_CONTROL_SOURCE="$PWD/crates/wgtime-control"
```

保留解压源码，`weightc compile` 的 Rust/Cargo 构建需要其中的 `crates/wgtime-control`。启动服务使用当前用户拥有、权限 `0700` 的目录：

```sh
install -d -m 700 /tmp/weightlang/wgtime
wgtime --test-directory /tmp/weightlang/wgtime
```

保持服务运行，在客户端终端设置 `WGTIME_ENDPOINT=/tmp/weightlang/wgtime/service.sock`。服务不提供 `--version` 或 `--help`。

## 失败处理

| 失败位置 | 处理方法 |
| --- | --- |
| 版本不一致、格式错误或降级 | 修正输入或版本记录后重新运行；版本脚本尚未写入文件。 |
| 标签已存在 | 新发布选择新版本；原运行的失败恢复使用 **Re-run failed jobs**。 |
| 构建、SDK 测试或发布预演 | 修复失败项后重新运行；远端分支、标签和包仓库尚未修改。 |
| 构建期间默认分支移动 | 从最新 `main` 重新运行。 |
| Git refs 推送失败 | 按仓库规则处理机器人权限；原子推送不会只写入其中一个 ref。 |
| `First crates.io publication requires CARGO_REGISTRY_TOKEN` | 在 `crates-io` environment 或仓库 secrets 配置允许发布所列新 crate 的 token，再重跑失败任务。 |
| `No Trusted Publishing config found` | 首次发布配置 `CARGO_REGISTRY_TOKEN`；已有 crate 核对各自的 owner、repository、`release.yml` 和 `crates-io` environment。 |
| 包仓库认证失败 | 检查对应 environment 的 Trusted Publisher 或 secret，再重跑失败任务。 |
| 某个 crate 已存在且内容不同，或已撤回 | 不覆盖该版本；修正发布版本并重新验证。 |
| Rust 成功但 Python 失败 | 重跑失败任务，Python 使用原始 artifact 继续上传，成功后公开 GitHub Release。 |
| 标签已推送但草稿上传失败 | 重跑原运行；先核对已有标签的源码树，再补全草稿资产。 |

重跑需要原运行 artifacts 仍然可用。标签创建后，不要用同一版本另起一次完整发布；新的准备任务会拒绝已存在的标签。应在原运行选择 **Re-run failed jobs**，复用已验证制品并继续后续任务。GitHub 重跑保留原事件的 `GITHUB_SHA` 和 `GITHUB_REF`；修改 `main` 上的工作流不会改变旧运行，认证失败的恢复仍须补齐对应 secret 或注册表身份。参见 [GitHub 重跑规则](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)。
