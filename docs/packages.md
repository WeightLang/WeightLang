# 包与脚本依赖

包内每个模型的源码入口统一命名为 `model.wgt`。单模型可放在包根目录，多模型放在各自子目录，
例如 `packages/gates/xor/model.wgt`。打包器递归收集 Git 跟踪的 WGT 源文件，
索引中的 source_path 保留相对包根目录的完整路径；同名入口不会被压成同一个文件名。
已发布版本的索引、哈希与 `source.wgt` 制品快照保持原样，不因本地源码整理而重写。

`wgt` 解析和校验精确版本的模型包依赖。脚本可通过项目 `wgt.toml`（兼容 `wgt.mod`）或文件开头的 `# /// script` 注释块声明依赖。版本、锁文件、缓存和离线行为见[单文件依赖契约](native-training.md#单文件依赖注释与直接执行)。

已发布的原生 0.5 Gates 和 Iris 模型包可通过完整示例加载：

```sh
# 首次运行按注释块获取并校验依赖；模型只用于推理，不执行包中的训练入口。
wgt run examples/package_usage/model.wgt
# 已有锁和缓存时可离线运行。
wgt --offline --locked run examples/package_usage/model.wgt
```

[示例源码](../examples/package_usage/model.wgt)使用 `dependency(module)` 获取包缓存目录，并把 `model.json` 的路径交给 `load model from PATH using "sgd"`。模型文件和优化器断点 `checkpoint.json` 用途不同；加载推理模型不会训练。Gates、Iris 的结构、训练和评估范围分别见[包说明](../packages/gates/README.md)和[包说明](../packages/iris/README.md)。

`dependency` 返回的是文件路径，不导入 WGT 函数，也不把旧版制品自动转换为原生 IR。
纯张量算子库使用独立的 `operators NAME` 与 `import alias from "MODULE/LibraryName"`，
`release-assets` 无需预构建权重即可打包，安装沿用哈希与版本锁。
完整定义、发布、调用和兼容边界见[原生自定义算子](custom-operators.md)。
模型构造器仍不支持通用模型 import、use 或 block。`--check` 不下载依赖，
只解析本地库；需要包依赖上下文的导入明确报错。

依赖信息和 `wgt.sum` 锁定的版本及哈希应随项目保存。锁定哈希能检测内容变化，但不能证明训练过程或模型质量。可用 `wgt add --help`、`wgt fetch --help` 和 `wgt sync --help` 查看当前 CLI 参数。
