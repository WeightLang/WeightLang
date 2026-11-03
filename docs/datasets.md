# 独立数据与 Python 入口（wgt 0.3，当前源码实现）

> 当前 `build` 只生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例改用 `weightc legacy-build`；参见[分发包与迁移规范](bundles.md)。


完整可运行示例见 [external_data](../examples/external_data/model.wgt) 和
[Python 回调示例](../examples/external_data/python.wgt)。这些是合成数据示例，未发布。

数据文件 `data_x.wgt` 只含版本头与一个或多个 `data` 声明，不需要模型、层或 init：

```wgt
wgt 0.3
data samples:
  [0] -> [1]
  [1] -> [3]
```

模型通过 `data samples: file="data_x.wgt" sha256=<原始文件摘要>` 引用。
默认读取同名数据集；`dataset="另一个名字"` 显式选择。
数据文件也可引用其他文件；路径始终相对声明所在文件，每层都需 SHA256，禁止循环引用。
独立数据文件不允许模型声明或 Python 加载器；Python 加载器直接声明在模型中。
`weightc check data_x.wgt` 只检查声明语法，不读取外部载荷或验证模型形状；
检查引用它的模型时会验证所选文件数据及形状。

## 文件加载

完整选项：`data NAME: file="PATH" sha256=HEX [format=FORMAT] [dataset=NAME] [skip=0] [license="TEXT"] [source="TEXT"]`。
所有文件必须提供原始字节的 64 位小写 SHA256。format 默认从扩展名推断；未知格式报错。
0.2 的旧文件行为保持 CSV；新格式和选项要求 0.3。

| 格式 | 内容 |
| --- | --- |
| csv / tsv | 数字列，输入列在前、目标列在后；支持带引号的数字字段，忽略空行和以 # 开头的行 |
| json | `{"inputs": [[0], [1]], "targets": [[1], [3]]}` |
| jsonl | 每行 `{"inputs": [0], "targets": [1]}`；忽略空行 |
| npz | NumPy 数组 inputs、targets；形状均为 `[样本数, 展平宽度]`；禁止 pickle/object 数组 |
| wgt | 上述独立数据声明；dataset 默认取当前数据集名 |

skip 默认 0，只允许 CSV/TSV/JSONL，按原始文本行跳过；dataset 仅用于 wgt。
所有格式最终都需要非空、等行数的 inputs/targets，有限数值且每行宽度一致。
多维样本按最后一维最快的顺序展平。输入宽度须等于输入形状乘积；
MSE/BCE 目标匹配输出宽度，交叉熵每个位置一个合法类别索引。
整数输入还需符合 i64 与词表范围。数据全部载入内存，目前不支持流式迭代器。

## Python 数据与训练

```wgt
data samples: python="hooks.py:load"
train data=samples loss=mse epochs=1 python="hooks.py:train"
```

路径相对模型文件；`:函数名` 可省略，data 默认 load，train 默认 train。
函数必须是脚本顶层可调用对象。脚本依赖从当前 Python 环境导入。
执行具有普通 Python 进程权限；解析、check、build --no-train 不执行脚本或入口点。
独立制品的推理不需要原脚本。trace/debug 对 Python 数据要求显式 `--sample`。

`load(context)` 返回且仅返回 `{"inputs": rows, "targets": rows}`。
rows 可为二维列表或可转换为列表的张量/数组，每行是展平数值，不能是标量。
context 为 `weightlang.datasets.DataContext`：

| 属性 | 含义 |
| --- | --- |
| name | 当前数据集名 |
| source_dir | 模型文件所在目录的绝对 Path，可用于读取相对文件 |
| input_shape / output_shape | 不含 batch 的声明形状 tuple |
| input_dtype | f32 或 i64 |
| seed | 模型 init seed，加载器自行使用以保持可复现 |

`train(context)` 使用现有 [TrainingContext](trainers.md#python-api)，原地更新允许训练的参数，
返回有限数值指标字典或 None。必填 data/loss/epochs；可选 lr=0.01、batch、validate。
不接受 optimizer、weight_decay 或 trainer。不能替换结构、改形状/dtype、修改冻结参数或产生非有限权重。
模型文件声明回调后无需命令行再次注册。

## 安装包入口点

用 `entrypoint="NAME"` 代替 python。data 使用 `weightlang.data_loaders` 组，
train 使用 `weightlang.trainers` 组；这是 Python 安装包 metadata 入口点，不是命令行 main。
名称必须恰好匹配一个已安装入口点；缺失、重名或非可调用对象都报错。
包的 pyproject.toml 示例：

```toml
[project.entry-points."weightlang.data_loaders"]
my_data = "my_package.data:load"

[project.entry-points."weightlang.trainers"]
my_train = "my_package.training:train"
```

```wgt
data samples: entrypoint="my_data"
train data=samples loss=mse epochs=20 entrypoint="my_train"
```

file、python、entrypoint 互斥；train 的 trainer、python、entrypoint 互斥。
脚本记录路径与源码 SHA256；安装包入口记录名称、目标与发行包版本（若可得），
它们并不固定传递依赖。最终数据内容摘要与训练指标进入制品 manifest。
