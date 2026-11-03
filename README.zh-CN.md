# WeightLang

[English](README.md)

用原生 WGT 程序构建、训练和运行模型。`weightc`、`wgt` 和 `wgt-native` 都是
Rust 可执行程序。当前源码已清退 Python 实现、测试、回调入口、打包与安装链路；
这不代表已发布版本也已更新。

## 安装与更新

源码安装需要 Rust/Cargo：

```sh
cargo install --locked --force --path crates/wgt --bin weightc --bin wgt
cargo install --locked --force --path crates/wgt-runtime --bin wgt-native
```

[Shell 安装脚本](install.sh) 和 [PowerShell 安装脚本](install.ps1) 同时安装、更新
`wgt` 与 `weightc`。`WEIGHTLANG_SOURCE` 指定本地仓库；仓库外默认构建 GitHub main，
也可用 `WEIGHTLANG_VERSION` 指定包含原生实现的 tag。不安装 Python、uv 或 Torch。
已有模型和实验目录不受影响。

```sh
wgt update --version X.Y.Z
# 或选择最新完整原生发行版：
wgt update
```

更新要求 `wgt-vX.Y.Z` 发行版同时包含当前平台的两个二进制和 `SHA256SUMS`。
两者的哈希、版本均通过校验后才切换选择；`weightc` 与 `wgt` 使用同一个项目版本锁
或默认版本。旧 Python 发行版、缺少编译器的发行版不作为更新候选。
完整参数和发布打包约定见 [迁移说明](docs/weightc-migration.md)。

## 运行

```sh
cargo build --locked --workspace --bins
./target/debug/weightc check examples/native_training/build.wgt
./target/debug/wgt run examples/native_training/supervised.wgt
./target/debug/weightc legacy-build examples/native_training/build.wgt -o weights/native-demo
./target/debug/weightc compile examples/native_training/compiled.wgt -o .cache/weightlang/demo
./.cache/weightlang/demo
```

当前原生脚本以 `wgt 0.5` 开头。`check` 只解析、不执行；`compile` 生成本机程序，
训练和保存由脚本显式控制。`legacy-build`（兼容别名 `build`）要求 `main` 返回模型，
输出完整模型目录，拒绝覆盖任何已有目标。

`wgt new demo` 创建原生脚本项目；`wgt sync` 只同步包依赖；`wgt run main.wgt`
执行脚本，不再准备 Python 环境。制品检查、图形渲染和依赖获取与训练执行分开。

## 能力范围

当前子集见 [原生训练](docs/native-training.md)、[原生推理](docs/native-backend.md)、
[量化](docs/quantization.md) 和 [自适应执行](docs/self-modifying-runtime.md)。
清退 Python 不意味着全部历史算子、训练器或 CLI 功能已有原生替代。
仍依赖旧 CLI 的 `wgt check/build/package/install/train/export/test/trace/debug`
明确报错；执行训练请使用上述原生脚本入口。

WGT 标准库源码保留在 `std/`，算子签名保留在 `language/primitive_specs.json`。
历史声明式模型及制品夹具作为迁移资料保留，不表示原生执行器已经支持。
标为历史说明的文档可能包含已删除的 Python API 和示例。

完整模型和保留实验结果放在 `weights/`。大型构建、下载缓存必须在磁盘上，
小临时文件放在 `/tmp/weightlang/`，详见 [存储约定](docs/storage.md)。
模型制品与包含优化器状态的训练断点不同；初始化不等于已训练。

## 验证与发布打包

```sh
cargo test --locked --workspace
cargo build --locked --release -p weightlang-wgt --bins
sh scripts/package_wgt.sh --binary target/release/wgt --output dist/native-release
# Windows：scripts/package_wgt.ps1 -Binary target/release/wgt.exe -Output dist/native-release
```

打包检查两个程序版本一致，在全新目录写入平台文件与校验和；不会自动发布。

[MIT 许可证](LICENSE)。
