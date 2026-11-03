# 模型数据流

当前原生声明式模型有一个固定长度 `f32[N]` 输入和一个固定长度 `f32[M]` 输出，不包含 batch 轴。`layer name: expression` 按声明顺序把前一层输出传给下一层；输出宽度必须与最后一层相符。当前支持 `linear`、`relu`、`tanh`、`sigmoid` 和 `identity`，完整参数及约束见[语言参考](reference.zh-CN.md#声明式模型)。

`predict(model,input)` 接受一个样本向量并返回一个输出向量。训练时 `fit(model,inputs,targets,lr,loss)` 接受多个样本组成的列表；MSE 目标是输出向量列表，交叉熵目标是类别索引列表。样本维度与单个样本内部的特征维度是不同的轴。`fit` 显式执行一次更新，详见[原生训练](native-training.md#梯度与参数更新)。

普通顺序模型的 `predict` 不改变参数；自更新模型的 `fn forward` 可以显式 `write` 参数，成功时提交，失败时回滚。其张量操作和状态契约见[自更新运行时](self-modifying-runtime.md)。当前声明式入口不支持多输入、多输出、通用张量图或隐式缓存。
