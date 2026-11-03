# 工程组织与层级命名

WeightLang 采用以下术语作为工程组织约定；它们对应已有语法，不引入新的关键字。
完整可运行示例见 [complete_project](../examples/complete_project/README.md)。

| 名称 | 英文 | 边界与职责 | 与 Python 的近似关系 |
| --- | --- | --- | --- |
| 工程 | Project | 一个 `wgt.toml` 所在目录；管理入口、依赖、运行环境、数据和测试 | 应用或发行工程 |
| 构件库 | Library | 一个只含版本、`use`、`block` 的 `.wgt` 文件；组织可复用网络结构 | 源码模块 |
| 构件 | Block | 一个带命名参数的 `block`；调用时展开为网络结构 | 函数或类的结构模板 |
| 模型 | Model | 一个含 `model` 声明的 `.wgt` 文件；定义输入输出、层、初始化及可选训练 | 可构建入口 |
| 层 | Layer | `layer name: ...` 声明的一次算子或构件实例 | 对象实例，非源码模块 |
| 参数张量 | Parameter tensor | 层中的 `weight`、`bias` 等实际数值 | 对象状态 |
| 模型制品 | Artifact | `weights/<model>/` 下完整的结构、权重与来源记录 | 构建产物 |
| 权重包 | Weight package | 以工程路径和版本发布的一组模型制品 | 发行包，但复用的是权重 |

不再笼统地把 `.wgt` 文件、工程目录、网络层都叫“模块”。已有的
`project.module`、`wgt.mod` 中的 `module` 和包索引字段保留兼容：它们表示
**包标识路径**，不是 Python 的可导入源码模块。

## 两条层级

源码组织与运行结构是两条不同的层级，不强行套成一棵目录树：

```text
工程
├── 构件库（lib/*.wgt）→ 构件（block）
└── 模型入口（models/*.wgt）
    └── 层实例（layer，可以调用构件）
        └── 子层 → 参数张量

模型入口 + 数据 + 初始化/训练/导入 → 模型制品
一个或多个模型制品 + 包标识 + 版本 → 权重包
```

构件可以调用其他构件，包括其他本地构件库的构件；循环引用不允许。
模型也可以直接声明私有构件，小模型不必为了目录结构而拆文件。
每次构件调用拥有独立参数，源码复用本身不共享已训练权重。

- `use "../lib/sensors.wgt" as sensors`：复用结构。
- `layer calibration: sensors.calibrate(channels=2)`：实例化构件。
- `import ... from "../weights/..." layer=...`：复制制品中的权重；不是导入源码。

构件库别名、构件名和层实例名是不同概念。示例中的参数路径
`calibration.readout.linear.weight` 由**层实例名**构成，不含目录名、库别名或
构件名。重命名层实例会改变参数路径，也会影响权重导入、设备放置和补丁目标。

## 标准目录

```text
my_project/
├── wgt.toml                # 工程配置；[model].source 指定默认入口
├── wgt.sum                 # 权重依赖锁，无依赖时为空；提交 Git
├── wgt.lock                # wgt sync 生成的 Python 运行环境锁；生成后提交
├── models/                 # 模型入口；每个文件一个模型
├── lib/                    # 本地构件库；可按领域建立子目录
├── data/                   # 小型可分发数据及其说明
├── tests/                  # wgt test 读取的 JSON 行为断言
├── weights/<model>/        # 保留完整模型制品；不提交 Git
├── .cache/weightlang/      # 可再生的运行、测试、环境缓存
├── dist/                   # 显式打包的发布资产
├── .gitignore
└── README.md
```

`models/`、`lib/` 是约定目录名，不是自动发现或导入机制。
本地 `use` 和数据路径相对**声明它们的源文件**解析；配置中的入口、输出目录和
测试 JSON 的 `source` 相对**工程根目录**解析。进入子目录运行 `wgt` 时仍向上寻找工程。
`lib/` 无需 `__init__.wgt`，目录本身不是包，也没有隐式导出或通配导入。
目前远程依赖发布的是模型权重，不支持远程源码构件库。

名称统一使用 `snake_case`：库文件按领域命名（`sensors.wgt`），构件按结构或职责
命名（`calibrate`），模型按用途命名（`sensor_calibration`），层按角色命名
（`readout`）。模型名决定输出目录名，与入口文件名可以不同；同一输出根中的模型名应唯一。

一个工程可以有多个模型入口，但当前 `[model].source` 只指定一个默认入口。
构建其他入口需显式执行 `wgt build models/other.wgt`；不要假设工具会递归构建
`models/`。`wgt check` 同样只检查配置的入口及其引用链，未被引用的库应显式检查。
当前没有多工程 workspace 或自动模型依赖构建图。

数据较大时采用磁盘上的专用存储并记录来源与哈希，不把大数据放进临时内存文件系统。
生成物的保留和覆盖规则见 [storage.md](storage.md)，权重发布见 [packages.md](packages.md)。
