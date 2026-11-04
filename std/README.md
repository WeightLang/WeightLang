# 标准库目录

真实可导入的纯 WGT 库（WGT 0.5，本地未发布）：

- `import activation from "std/activation/operators"`：swish、swiglu、squared_relu。
- `import loss_ops from "std/loss/operators"`：squared_error、half_squared_error、categorical_logits。

仅在 forward/loss 调用；输入为非空同形 f32 向量，全部参数必填，无隐含广播。
激活保持形状，损失归约到 f32[1]；权重与训练由调用模型显式管理。
完整真实训练例子为 [standard_operators/model.wgt](../examples/native_training/standard_operators/model.wgt)。
定义、标签限制、CPU 梯度和 AOT 契约见[标准库文档](../docs/standard-library.md)。

下列分类文件中的 main 是运行时内置的演示，不能作为可导入库；
根目录 block/use 文件目前不能由原生源码前端执行。图制品的形状验证与查看能力
不证明这些源码能编译执行，逐模块阻碍见标准库文档的审计表。


- `math/`：原生标量数学的完整可运行示例。
- `activation/`：稳定激活函数的完整可运行示例。
- 根目录旧 block 文件：保留既有路径，属于尚未完整迁移的图标准库。

新增公共算法由 Rust 运行时实现，WGT 直接调用，不需要复制示例函数。
签名、数值边界及硬件支持见 [标准库文档](../docs/standard-library.md)。

- `linalg/`：内积、L2 范数。
- `reduction/`：向量均值。
- `probability/`：softmax、log_softmax、logsumexp。
- `normalization/`：带显式 eps 的 L2 归一化。
- `loss/`：MSE、MAE、logits 交叉熵（标量计算，无自动微分）。
- `statistics/`：极值、方差、标准差、中位数、线性插值分位数。
- `ordering/`：稳定排序、索引排序、极值位置、前缀和。
- `linalg/transforms.wgt`：裁剪、余弦相似度。

- `activation/piecewise.wgt`：ReLU、Leaky ReLU、ELU、Hard Sigmoid/Swish。
- `loss/robust.wgt`：logits BCE、Huber、log-cosh。
- `signal/`：一阶差分、EMA、有效窗口滑动均值。

- `data/`：zeros/ones/full、linspace、独热编码、拼接、切片、取值、bool 筛选、倒序。
  这些数据操作允许空列表；已有统计归约仍要求非空。

- `probability/distributions.wgt`：非负权重归一化、熵、KL/JS 散度、概率交叉熵。
- `probability/temperature.wgt`：正温度 softmax。
