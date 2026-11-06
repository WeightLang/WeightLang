# 标准库

当前 WGT 0.5 的 `std/` 聚焦模型内部纯张量计算库，均为本地实现，尚未发布。完整签名、形状、状态边界及有效模型示例见[标准库契约](standard-library.md)。

[`language/prelude.json`](../language/prelude.json) 记录无需导入的原生函数和张量 intrinsic，各名称只能在其登记的上下文使用，维护规则见[内部内置目录](prelude.md)。显式逻辑名 `std/activation`、`std/loss`、`std/attention` 和 `std/numerics` 提供模型内部算法。旧 `std/prelude.wgt` 用户声明源与 `std/openresponses/harness` 宿主模块已移除；内部 `/operators` 路径不是公共导入名。

旧 block 图占位和带 main 的分类数学演示已删除。标准库模块不充当模型入口，不隐式初始化或训练模型，也不代表多轴神经网络结构已全部实现。其他语言能力见 [WGT 0.5 语言参考](reference.zh-CN.md)，完整模型和实验遵循[存储规范](storage.md)。
