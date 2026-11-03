# 包注册表

原生 `wgt` 支持使用 Git 索引解析注册的包，并从发布资源获取制品；默认索引为 [WeightLang registry](https://github.com/WeightLang/registry)。`wgt add`、`wgt update` 和 `wgt fetch` 的实际参数以各命令 `--help` 为准，`--offline` 与 `--locked` 分别限制网络访问和锁变更。

注册表和模型包用于定位与校验文件，不会将旧版声明式源码或传统模型制品转换成当前原生模型。WGT 0.5 脚本可使用 `dependency(module)` 获取已声明依赖的本地目录，详见[包与脚本依赖](packages.md)和[依赖契约](native-training.md#单文件依赖注释与直接执行)。

发布资源和更新注册表是独立的外部操作，具体选项可用 `wgt publish --help` 与 `wgt register --help` 查看；校验哈希不证明模型经过训练或达到特定质量。
