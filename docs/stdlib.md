# 标准库

当前原生 WGT 0.5 提供可直接调用的标量数学、激活函数及向量算法；完整签名、数值约束及示例见[原生数学标准库](standard-library.md)。`std/math/`、`std/activation/` 等分类目录保存可运行示例，不引入模块导入语法。

`std/` 根目录的旧 block 图源码仍作为迁移材料保留。当前原生模型构造器不支持 `block` / `import`，不能因为新函数可用就宣称这些旧图可执行。其余语言能力见 [WGT 0.5 语言参考](reference.zh-CN.md)，完整模型和实验遵循[存储规范](storage.md)。
