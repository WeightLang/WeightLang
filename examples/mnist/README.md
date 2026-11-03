# MNIST：训练一个手写数字识别模型

网络由 `main.wgt` 定义，使用 WeightLang 编译、初始化、保存和加载。
`train.py` 通过项目已有的 [Python 互操作接口](../../docs/python.md) 训练网络。
数据独立准备，训练和推理不会联网。无需 GPU，无需 torchvision。

这是一个 `784 → 128 → 64 → 10` 的全连接网络，共 109,386 个参数。
训练目标是识别 28×28 灰度图片中的 0–9，而不是记住几个演示样本。

### 本地实测

CPU、PyTorch 2.15.0、Python 3.14.9，固定种子 42：

| 模式 | 训练图片 | 训练轮次 | 保存轮次 | 验证准确率 | 独立测试准确率 | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整 | 55,000 | 10 | 8 | 97.60% | **97.55%（9,755/10,000）** | 约 13.0 秒 |
| 快速 | 2,000 | 2 | 2 | 77.84% | 79.94%（7,994/10,000） | 约 1.1 秒 |

耗时包含初始化、训练与训练/验证指标计算，不包含下载和独立测试。
完整模型的测试交叉熵为 0.0786，错误 245 张；模型目录约 548 KB（含报告和预览）。
权重 SHA256：`d762cb12e3845c532a9ac9a65755b31693c80fe426c972e9c40f10d73b031bbc`。
训练损失用于学习，验证损失用于选择轮次；以上测试结果来自保存后重新加载的模型。

## 安装与运行

下面命令从 **WeightLang 仓库根目录**执行。需要源码版 WeightLang、NumPy、
PyTorch 和 Pillow；前三者由项目依赖提供，Pillow 用于图片读写。
如已有开发环境，可跳过安装。首次安装依赖可能较大，先检查磁盘空间；
不要把数据、虚拟环境或下载缓存放到 tmpfs。

```sh
python -m venv .venv
.venv/bin/python -m pip install -e . Pillow

# 1. 下载、校验、转换，重复执行会校验并复用已有缓存。
.venv/bin/python examples/mnist/prepare_data.py

# 2. 检查网络，然后初始化、训练 10 轮，保存验证集损失最低的那轮。
.venv/bin/weightc check examples/mnist/main.wgt
.venv/bin/python examples/mnist/train.py

# 3. 从磁盘重新加载，评估全部 10000 张官方测试图片。
.venv/bin/python examples/mnist/evaluate.py

# 4. 识别评估脚本导出的第一张测试图片（真实标签为 7）。
.venv/bin/python examples/mnist/predict.py examples/mnist/weights/mnist/sample.png

# 5. 校验模型和离线数据处理逻辑。
.venv/bin/weightc verify examples/mnist/weights/mnist
.venv/bin/python examples/mnist/test_pipeline.py
```

已经训练过时，直接从第 3 步开始。`train.py` 拒绝覆盖已有模型；重新训练时
用 `--output examples/mnist/weights/mnist_run2` 保存到新目录，评估和预测则
传入相同的 `--model`。本例不自动删除旧模型。

`wgt build` 或 `weightc build main.wgt` **仅初始化这个网络，不训练 MNIST**。
训练入口是 `train.py`，因为当前语言的文件数据入口是文本/CSV，不支持直接读取
MNIST 的 IDX/NPZ。本例通过正式 Python API 训练 `.wgt` 网络，避免把图片
转换成体积庞大、解析缓慢的 CSV；未改动编译器，也没有另写一份 Python 网络。

## 从空目录初始化

如需体验 `wgt new`，先安装源码版 Rust 工具：

```sh
cargo build --locked -p wgt
export PATH="$PWD/target/debug:$PATH"
export WEIGHTC="$PWD/.venv/bin/weightc"
TEMPLATE="$PWD/examples/mnist"
wgt new examples/my_mnist
cp "$TEMPLATE"/*.py "$TEMPLATE/main.wgt" "$TEMPLATE/wgt.toml" \
   "$TEMPLATE/README.md" examples/my_mnist/
# 删除 new 自动生成的 identity 测试；本例用 test_pipeline.py 和独立评估。
rm examples/my_mnist/tests/main.json
rmdir examples/my_mnist/tests
wgt -C examples/my_mnist sync --no-env --offline --locked
wgt -C examples/my_mnist check
.venv/bin/python examples/my_mnist/prepare_data.py
.venv/bin/python examples/my_mnist/train.py
.venv/bin/python examples/my_mnist/evaluate.py
```

复制过去的项目会使用自己的缓存和 `weights/`。无需再执行 `wgt build`，
`train.py` 已调用编译器进行初始化。

## 数据怎么管理

- 下载四个原始压缩包，采用 [torchvision 官方源码](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html)
  中的 HTTPS 镜像与固定 MD5 校验值；来源是 [MNIST](http://yann.lecun.com/exdb/mnist/)。
- 下载先写 `.part` 临时文件，通过校验后原子改名；失败清理本次临时文件，
  不用错误下载替换旧文件。已有正确文件直接复用。
- 解码时检查 IDX 格式、尺寸、长度和标签范围；缓存为压缩 NPZ，使用 `uint8`
  保存像素。原始压缩包与准备后的数据合计约 23 MB，不保存一份巨大的浮点 CSV。
- 官方训练集 60,000 张按种子 42 固定打乱，前 5,000 张用作验证，其余
  55,000 张用于训练。官方测试集 10,000 张不参与训练、调参或选择训练轮次。
- `dataset.json` 记录来源、各压缩包 SHA256、划分索引哈希、处理规则和 NPZ
  哈希。训练和评估都重新校验 NPZ，并拒绝损坏缓存。加载不使用 pickle。
- 只在内存中把像素转换为 float32、除以 255，并按行展平成 784 个数字。
  不额外下载数据增强、预训练权重或模型包。

缓存路径为 `.cache/weightlang/data/`，相对于本示例目录，而非调用命令时的目录。
`--data-dir /磁盘上的目录` 可指定共享缓存或手工准备的数据路径；准备、训练、
评估三个脚本要使用同一目录。

```sh
# 已准备的数据可完全离线复用。
.venv/bin/python examples/mnist/prepare_data.py --offline

# 本地已有四个官方 .gz 文件时，放在 DATA_DIR/raw/，离线转换。
.venv/bin/python examples/mnist/prepare_data.py --data-dir DATA_DIR --offline

# 准备后的缓存损坏时，从通过校验的原始压缩包重新生成。
.venv/bin/python examples/mnist/prepare_data.py --offline --rebuild
```

原始包可重新下载，转换缓存可重建，二者都不提交 Git。缓存是磁盘数据，
不要把 `--data-dir` 指向 `/tmp`、`/dev/shm` 等内存文件系统。

## 快速模式与训练细节

```sh
.venv/bin/python examples/mnist/train.py --small --epochs 2
.venv/bin/python examples/mnist/evaluate.py --model examples/mnist/weights/mnist_small
```

快速模式只使用固定划分后前 2,000 张训练图片；仍用独立的 5,000 张验证图片和
10,000 张测试图片。产物保存为 `weights/mnist_small/`，结果明确标记为 `small`，
不与完整训练混用。准备数据仍下载完整 MNIST。

默认 CPU、4 个计算线程、种子 42、batch 128、Adam 学习率 0.001，训练 10 轮。
每轮记录训练损失和验证集指标，选择**验证集交叉熵最低**的权重。
测试集只在保存模型后由 `evaluate.py` 单独评估，不参与选择。
同一环境尽量保持确定性；跨 PyTorch 版本和硬件不保证权重字节相同。

## 产物与验收

```text
mnist/
├── main.wgt, wgt.toml, wgt.sum
├── prepare_data.py, data.py
├── train.py, evaluate.py, predict.py, test_pipeline.py
├── .cache/weightlang/data/      可复用的本地数据缓存
└── weights/mnist/              完整模型与结果，一起保留
    ├── manifest.json          权重来源、结构、哈希和训练记录
    ├── source.wgt
    ├── weights.safetensors
    ├── training.json          每轮曲线、所选轮次、数据与代码哈希
    ├── evaluation.json        测试准确率、损失、混淆矩阵及全部错误索引
    ├── predictions.png        前 40 张测试图片及预测（非挑选成功样本）
    ├── errors.png             前 40 个识别错误
    └── sample.png             第一张测试图片，供单图预测
```

保存后会核对内存模型与重新加载的模型预测逐位一致。评估检查训练数据版本、
权重来源，推理加载器检查制品哈希与形状。测试代码覆盖离线行为、下载校验、
失败不覆盖旧缓存、IDX 截断、固定且不重叠的数据划分与预处理。
评估会更新本模型目录里的报告和预览，不改权重。

图片预测支持居中的 **28×28、黑底白字**图片；白底黑字加 `--invert`。
本例刻意不自动裁切、纠偏或猜测颜色，其他尺寸会提示先处理图片。
输出的 `score` 是 softmax 分数，不是校准过的可靠性保证。MNIST 测试表现
不等于手机照片或任意书写风格的准确率。

`source.wgt` 保存网络定义，外部 Python 训练由 manifest 的 `python_history`
记录。要复现训练，必须同时保留这里的训练脚本与数据准备规则，而不是只执行
模型目录里的 `source.wgt`。
