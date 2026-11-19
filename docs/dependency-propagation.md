# 依赖传播清单

本清单维护 WeightLang 自身的消费者和验收责任。依赖边表示编译、运行或安装实际使用了上游的代码或制品；共用开发机、维护工具或同一条协作规则不会自动产生依赖。

本轮固定功能源码为 [072582a03a123b7d95c815d563689383033019d4](https://github.com/WeightLang/WeightLang/commit/072582a03a123b7d95c815d563689383033019d4)，源码版本仍为 0.5.4。源码验证、正式发布和已安装程序更新分别记录。

## 仓库与消费者

| 项目 | 角色与依赖证据 | 需要维护的来源 |
| --- | --- | --- |
| WeightLang/WeightLang | 编译器、Rust/Python SDK、原生执行服务；[根 Cargo 工作区](../Cargo.toml)和 [Python 元数据](../python/pyproject.toml)定义包边界 | 根 `Cargo.lock`、SDK 包内容、编译器和服务的实际来源 |
| 同仓 iris | Rust 应用；[packages/iris/Cargo.toml](../packages/iris/Cargo.toml)以别名 `wgt` 引用 path 包 `wgt-model` | [独立 Cargo.lock](../packages/iris/Cargo.lock)；应用自身版本 0.5.0，WGT path 包版本 0.5.4 |
| 同仓 gates | Rust 应用；[packages/gates/Cargo.toml](../packages/gates/Cargo.toml)以别名 `wgt` 引用 path 包 `wgt-model` | [独立 Cargo.lock](../packages/gates/Cargo.lock)；应用自身版本 0.5.0，WGT path 包版本 0.5.4 |
| WeightLang/benchmark | 原生权重初始化基准；[Cargo.toml](https://github.com/WeightLang/benchmark/blob/main/Cargo.toml)引用 `.vendor/weightlang/crates/wgt-model`，[bootstrap.sh](https://github.com/WeightLang/benchmark/blob/main/scripts/bootstrap.sh)构建固定源码的服务 | [WEIGHTLANG_REVISION](https://github.com/WeightLang/benchmark/blob/main/WEIGHTLANG_REVISION)、`Cargo.lock`、源码归档和原生验收回执 |
| LIghtJUNction/lightnest | Python 模型应用；[pyproject.toml](https://github.com/LIghtJUNction/lightnest/blob/main/pyproject.toml)声明 `weightlang[torch]==0.5.4`，uv source 指向 `../WeightLang/WeightLang/python` | `uv.lock`、`requirements.txt`、实际导入的 SDK 路径和本机模型验收 |

iris、gates 各自声明独立 Cargo 工作区，根工作区检查不会自动覆盖它们。lightnest 的可编辑源码路径会跟随开发树变化，验收记录必须同时写出实际使用的 WeightLang 提交。

## 直接和间接传播路径

| 上游 | 下游与关系 | 来源 |
| --- | --- | --- |
| `wgt-model` | 直接进入 iris、gates、benchmark 的 Rust 依赖 | 三个应用的 Cargo manifest |
| `wgt-language` | 直接进入 `wgt-model` 和 `wgt-model-macros`，再间接影响上述 Rust 应用 | [宏包 manifest](../crates/wgt-model-macros/Cargo.toml)、[SDK manifest](../crates/wgt-model/Cargo.toml) |
| `wgtime` 客户端库 | 经 `wgt-model` 间接影响上述 Rust 应用 | [SDK manifest](../crates/wgt-model/Cargo.toml) |
| `wgtime-core` | 直接进入 `wgtime-service`；同时是 `weightlang-wgt` 的开发测试依赖 | [服务 manifest](../crates/wgtime-service/Cargo.toml)、[编译器 manifest](../crates/wgt/Cargo.toml) |
| `wgtime-service` | 运行时影响 benchmark 的原生执行；此关系不等于 benchmark 直接链接 `wgtime-core` | benchmark 的固定源码与 bootstrap |
| Python `weightlang` | 直接影响 lightnest 的模型加载、执行和权重接口 | lightnest 的 Python 元数据与源码绑定 |

原生服务、编译器调用和 Python 导入形成的运行依赖需要单独验收；Cargo 包级反向依赖只覆盖其 manifest 能表达的关系。当前可证实的仓库级消费者是 benchmark、lightnest；没有证据时不扩写额外的间接下游。

## 与 Project Z 的边界

本轮将 WeightLang 生态 manifest 和五份锁文件（根、iris、gates、benchmark 的 Cargo.lock，以及 lightnest 的 uv.lock）与 Project Z 当前维护树的已登记包名交叉核对，未发现对这些 Z 库的直接声明或锁定依赖。此结论限于检查到的声明和锁定源码；它不把未声明的外部进程调用当作已证明不存在。

因此，WeightLang 生态在本仓维护传播清单。只有出现真实 Z 库依赖证据，才与 Project Z 负责人协调纳入对应影响范围；是否进入其聚合快照由该负责人决定。此前的登记尝试仅保留在隔离候选补丁中，未提交、未推送，也未改变活动 Project Z 的登记文件。

benchmark 的 bootstrap 使用固定提交的源码归档，归档目录不是 Git 子模块。Project Z 现有源码等价检查要求包的相对位置、完整 Git 提交和干净检出一致；版本相同或归档标记相同不能替代该证明。若以后纳入，仓库级 `requires` 可以表达运行关系；将 benchmark 纳入 Cargo 聚合解析还需要单独解决可复现的来源初始化，保留其原生验收。

## 闭环记录与责任

| 范围 | 本轮维护责任 | 本轮状态与证据 |
| --- | --- | --- |
| WeightLang 核心与 SDK | WeightLang 主维护人 | 功能源码 072582a 已推送；核心、SDK、默认导出共 113 项通过，配套产物和独立 wheel 验收完成，详见[效率记录](efficiency-2026-11-20.md) |
| iris、gates | WeightLang 主维护人 | 四个 WGT path 包对齐 0.5.4；各自 locked/offline cargo test 退出 0，当前各 0 个测试，作为编译验收记录 |
| benchmark | 本轮 benchmark 维护人 | pin 固定到 072582a；Rust 4、Python 4、原生单步 6 次通过；[319e9ab](https://github.com/WeightLang/benchmark/commit/319e9ab515adfa069e28a7f6dc95b4ede2bc3cb4) 已推送 |
| lightnest | lightnest 维护人或本轮接手人 | editable 0.5.4 来源沿用本地 SDK；循环与 attention 两项真实服务回归通过；[a9c4e57](https://github.com/LIghtJUNction/lightnest/commit/a9c4e5732b4ce9168875669de321104a3f4f278d) 已推送验收回执 |
| Project Z | 现有聚合协调人 | 只决定有真实 Z 库关联的适用登记；独立候选补丁不代表聚合已采用 |

维护人应把消费者的不可变提交、验证结果及未解除的实际阻碍补进本轮验收记录。交接时写明接手人和下一步；消息已投递、源码已推送、消费者已验收和安装件已更新是不同状态。

以下命令可复查本仓三个 Rust 工作区的清单，不编译或改写锁文件：

```sh
cargo metadata --no-deps --locked --offline --format-version 1
cargo metadata --no-deps --locked --offline --format-version 1 --manifest-path packages/iris/Cargo.toml
cargo metadata --no-deps --locked --offline --format-version 1 --manifest-path packages/gates/Cargo.toml
```

metadata 结果用于核对声明；应用执行、原生服务和已安装 wheel 的验证继续使用各自的本机验收。
