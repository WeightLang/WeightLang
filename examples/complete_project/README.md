# 完整工程：传感器校准

这个示例用两个传感器读数预测一个校准值，学习人为定义的
`y = 2 * sensor_a - sensor_b + 0.5`。只有 3 个可训练参数，无需下载数据或权重。
它展示真实可执行的工程拆分，不是实际传感器精度或泛化能力的证明。

采用当前源码版的 `wgt` 和 `weightc`；配置工作流尚未随已发布的 0.2.0 wheel 提供。
安装原生工具见 [wgt](../../docs/wgt.md)，术语规范见
[工程组织与层级命名](../../docs/project-layout.md)。

```text
complete_project/                 # 工程 Project
├── wgt.toml                      # 默认入口、CPU 运行配置、包标识
├── wgt.sum                       # 空依赖锁：无需远程权重包
├── models/
│   └── calibration.wgt           # 模型 Model：数据、训练、输入输出
├── lib/
│   ├── sensors.wgt               # 构件库 Library：calibrate 构件
│   └── projections.wgt           # 构件库 Library：affine 构件
├── data/
│   ├── README.md                 # 数据生成公式和拆分说明
│   ├── train.csv                 # 8 条训练样本
│   └── validation.csv            # 3 条独立验证样本
├── tests/
│   └── calibration.json          # 数值行为断言
├── .gitignore
└── README.md
```

模型通过 `use` 引用 `sensors`，它再引用 `projections`。两个文件各自声明一个
`block`，最终展开为 `calibration.readout.linear` 层，参数是 `.weight` 和 `.bias`。
`use` 只复用结构；训练后的数值保存在模型制品里。

## 从检查到推理

在本目录执行：

```sh
wgt sync
wgt check
wgt build
wgt test
wgt run --input '[[0,0],[0.5,0.5],[0.25,-0.75]]'
```

首次 `sync` 可能需要下载 Python/PyTorch；模型本身没有外部依赖。
`sync` 生成的 `wgt.lock` 应提交到自己的工程版本库。
运行环境准备好后可以使用 `wgt --offline --locked check` 等命令。

预期推理输出接近 `[[0.5],[1.0],[1.75]]`。行为测试检查 4 组输入，绝对误差
不超过 `0.001`；前 3 组是验证数据，第 4 组未出现在训练或验证 CSV 中。
这仍然只是一个合成线性问题。

`wgt build` 执行 600 步训练，生成并保留：

```text
weights/sensor_calibration/
├── manifest.json                 # 展开结构、数据哈希、训练/验证指标、参数来源
├── weights.safetensors           # 学到的参数
└── source.wgt                    # 模型入口源码记录
```

制品中的展开结构用于加载推理；`source.wgt` 不是整个工程的源码归档，重新训练需要
保留 `lib/`、`data/` 和配置。模型名为 `sensor_calibration`，所以目录不叫 `calibration`。

再次 `wgt build` 会拒绝覆盖；保留比较版本可用 `wgt build -o weights/baseline_2`。
仅在明确要替换时使用 `--force`。`wgt run` 和 `wgt test` 会重新训练到
`.cache/weightlang/run/` 和 `.cache/weightlang/test/`，不会替换保留的模型。

若要直接使用已经构建的模型，不再训练：

```sh
weightc verify weights/sensor_calibration
weightc inspect weights/sensor_calibration
weightc run weights/sensor_calibration --input '[[0,0],[0.5,0.5],[0.25,-0.75]]'
```

上面的 `weightc` 需在安装了当前编译器的环境中运行。已有本仓库开发环境时，
也可以跳过 `sync`，用 `WEIGHTC=/绝对路径/weightc wgt --offline check` 等命令
显式选择编译器；详见 [运行环境](../../docs/environment.md)。

## 修改与扩展

- 换网络结构：编辑 `lib/` 中的构件，模型负责选择参数。
- 换任务：新增 `models/other.wgt`，执行 `wgt build models/other.wgt`。
- 换数据：修改 CSV，同时更新模型里的 SHA-256；不一致会在训练前报错。
- 加验收：新增 `tests/*.json`，`source` 指向相对工程根的模型入口。
- 分发权重：构建通过后按 [包发布文档](../../docs/packages.md) 发布；本示例不执行发布。

所有数据路径与 `use` 路径都相对各自 `.wgt` 文件，不依赖执行命令时的目录。
