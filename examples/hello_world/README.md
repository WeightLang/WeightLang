# hello_world：从初始化到训练、运行

这个完整项目训练一个 96 参数的小网络，让它按顺序输出 `hello_world`。
输入是字符位置的 one-hot 向量（11 个数字中只有对应位置是 1），输出是
8 个字符的分数。`run.py` 取每行最高分对应的字符，再拼成文字。
文字来自保存的模型预测；脚本不会训练模型，也不会直接打印预设答案。

这是对 11 个训练样本的记忆演示，不是语言模型，也不证明泛化能力。
无需 GPU、外部数据或预训练权重。

## 直接运行本示例

需要本仓库的源码版 `wgt` 和 `weightc`。以下命令在仓库根目录执行，
复用已有 `.venv` 和 `target/debug/wgt`，不重复下载运行环境：

```sh
export WEIGHTC="$PWD/.venv/bin/weightc"
target/debug/wgt -C examples/hello_world sync --no-env --offline --locked
target/debug/wgt -C examples/hello_world check
target/debug/wgt -C examples/hello_world build --offline --locked
.venv/bin/python examples/hello_world/run.py --check
target/debug/wgt -C examples/hello_world test --offline --locked
target/debug/wgt verify examples/hello_world/weights/hello_world
```

运行脚本输出：

```text
hello_world
PASS: 11/11 characters; trained weights; loss decreased
```

本示例已经本地训练，若 `weights/hello_world/` 已存在，直接执行
`run.py --check` 即可。只有明确要替换模型时才执行
`wgt -C examples/hello_world build --force`。默认构建拒绝覆盖已有模型。

## 从零初始化一个新项目

如果尚未安装源码版工具，先在仓库根目录安装（依赖下载和 Rust 编译可能超过
100 MB，开始前检查磁盘空间，缓存和虚拟环境应放在磁盘上）：

```sh
python -m venv .venv
.venv/bin/python -m pip install -e .
cargo build --locked -p wgt
```

然后仍在仓库根目录执行：

```sh
export WEIGHTC="$PWD/.venv/bin/weightc"
export PATH="$PWD/target/debug:$PATH"
PYTHON="$PWD/.venv/bin/python"
TEMPLATE="$PWD/examples/hello_world"

# new 创建配置、模型入口、空依赖锁和测试目录；目标目录必须尚不存在。
wgt new examples/my_hello_world
# 将默认的 identity 示例替换为本例，模型名仍是 hello_world。
cp "$TEMPLATE/wgt.toml" "$TEMPLATE/main.wgt" "$TEMPLATE/run.py" \
   "$TEMPLATE/input.json" "$TEMPLATE/vocabulary.json" examples/my_hello_world/
cp "$TEMPLATE/tests/main.json" examples/my_hello_world/tests/main.json

cd examples/my_hello_world
wgt sync --no-env --offline --locked
wgt check
wgt build --offline --locked
"$PYTHON" run.py --check
wgt test --offline --locked
wgt verify weights/hello_world
```

`sync --no-env` 复用上面明确指定的源码版编译器，只同步模型依赖（本例为零）。
也可使用 `wgt sync` 自动准备独立 Python/PyTorch 环境，详见
[环境管理](../../docs/environment.md)；这会另外下载依赖。
`wgt build` 会执行 `main.wgt` 中的 `init` 和 `train`：固定种子 42，
用 Adam 优化器训练 400 轮，目标是把输出拟合成对应字符的 one-hot 向量。

## 文件与验收

```text
hello_world/
├── wgt.toml            项目入口、CPU 配置和依赖
├── wgt.sum             空依赖锁（没有模型依赖）
├── main.wgt            网络、11 行训练数据、初始化和训练指令
├── input.json          推理所需的 11 个位置向量
├── vocabulary.json     输出索引对应的字符表
├── run.py              加载保存的模型、推理和解码
├── tests/main.json     检查所有输出分数，允许绝对误差 0.001
└── weights/hello_world/
    ├── manifest.json
    ├── source.wgt
    └── weights.safetensors
```

`run.py --check` 检查实际预测为 `hello_world`、训练损失下降、
所有张量来源均为 `trained`。加载模型时也会校验权重哈希和形状。
`wgt test` 会在 `.cache/weightlang/test/` 重新训练并检查数值结果；
`run.py` 检查的是 `weights/` 中已保存的模型，两者用途不同。

完整模型目录保存在本示例自己的 `weights/hello_world/`，不要只保留权重文件。
权重和缓存被 `.gitignore` 排除，不提交 Git；克隆后执行构建即可重新训练。

本地实测（CPU、PyTorch 2.15.0）：400 轮后 MSE 从 `0.1957220584`
降至 `3.5098e-12`，11/11 个字符正确，数值测试和制品校验均通过。
具体训练记录与哈希保存在 `weights/hello_world/manifest.json`；
重训后结果和哈希可能随 PyTorch 版本或硬件变化。
