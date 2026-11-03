# Python 管程序，WGT 定义模型

从仓库根目录运行，使用安装了项目依赖的 Python：

```sh
PYTHONPATH=src .venv/bin/python examples/python_first/train.py
PYTHONPATH=src .venv/bin/python examples/python_first/predict.py
```

模型定义在 `model.wgt`，`weightlang.load` 在内存中加载为普通 `nn.Module`。
训练由 `train.py` 控制，输入与标签来自 Python；合成数据只用于接口验证。
推理由 `predict.py` 控制，输入准备、softmax 和类别选择都在 Python。

训练输出完整模型目录 `weights/python-first/classifier/`，已存在时拒绝覆盖。
再次实验使用 `train.py --output weights/python-first/run-2`，推理传入该目录。
发布使用已有 `Artifact.save`，加载架构不需要中间制品；发布时才构建临时基线。
示例核对保存重载后的输出逐位一致；它不保存优化器状态，不是精确续训示例。
完整模型包含 manifest、源文件与权重，不要只保留 safetensors。

当前单输入接口接收 `[batch, 4]` 张量。需要多个独立张量的 mask/条件/cache
必须先扩展输入契约和运行时；标准库仅复用已能表达的张量计算。
已有 Python 自定义层可在外层组合，但当前完整制品保存器不能保存任意组合代码。

## 在 Python 中打印中间值

```sh
PYTHONPATH=src .venv/bin/python examples/python_first/debug.py
```

`forward_hooks(model, {"hidden.step0": callback, "head": callback})` 在上下文内监听指定层。
callback 接收 module、inputs、output；返回 None 不替换输出。退出后自动移除，仍可正常反向传播。
使用 named_modules() 查路径。完整制品不能保存活动 hook，发布前先退出上下文。

## 单包分发与默认续训（当前源码）

在此目录运行 `wgt build`，生成 `dist/model.wgtpkg`；它只包含 WGT 源码和标准元数据。
首次显式 `wgt train --fresh`，以后 `wgt train` 延续现有权重版本。
`wgt.toml` 的 `[train]` 指向 `task.py:train`；Python 任务拿到已恢复权重的 `ctx.model`。
模型包不自动包含任务脚本，也不执行任务。参见 [规范](../../docs/bundles.md)。
