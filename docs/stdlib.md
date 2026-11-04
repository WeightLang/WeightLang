# 标准库

当前 WGT 0.5 标准库由原生预导入声明、纯张量计算库和普通脚本模块组成，均为本地实现，尚未发布。完整签名、形状、状态边界及有效模型示例见[标准库契约](standard-library.md)。

[`std/prelude.wgt`](../std/prelude.wgt) 声明无需导入的原生函数和张量 intrinsic，各名称只能在其登记的上下文使用。显式逻辑名 `std/activation` 和 `std/loss` 提供六个可在 forward/loss 展开的纯张量函数；`std/openresponses/harness` 提供普通脚本的 async 请求及工具回合辅助，不自动执行工具。内部 `/operators` 路径不是公共导入名。

旧 block 图占位和带 main 的分类数学演示已删除。标准库模块不充当模型入口，不隐式初始化或训练模型，也不代表多轴神经网络结构已全部实现。其他语言能力见 [WGT 0.5 语言参考](reference.zh-CN.md)，完整模型和实验遵循[存储规范](storage.md)。
