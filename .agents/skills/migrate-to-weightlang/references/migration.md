# 原生 WGT 0.5 迁移路线

模型源码入口必须命名为 `model.wgt`。每个独立模型/训练示例一个目录，使用 `<模型或示例名>/model.wgt`，不要创建 `main.wgt`、`train.wgt` 或功能命名的模型入口。`fn main()` 是执行函数，不是入口文件名。迁移或移动入口时同步命令、文档链接、测试引用及相对数据/保存路径；标准库模块和制品的 `source.wgt` 快照不改名。

只接受 `wgt 0.5`。旧 Python 导入、反编译和旧语法入口已经移除，不能沿用历史命令。
先读取当前仓库 `docs/reference.zh-CN.md`、`docs/native-training.md` 和 `docs/storage.md`，
并核对 `wgt --help` 与 `weightc --help`。

## 项目接入

在待迁移项目的当前分支修改实际入口。声明式模型目前只支持单个 f32 向量输入输出，
顺序 linear/relu/tanh/sigmoid/identity；动态计算使用同一文件的 `fn forward`。
复杂注意力、任意张量算子和 Python 回调没有完整迁移，遇到缺口须明确报告，不能悄悄替换网络。

需要旧权重时，逐层核对轴序、dtype、形状、激活及共享关系，再生成原生模型 JSON。
使用相同输入对比输出；只有检查点而没有前向定义不能声称恢复完整模型。
未经用户允许，不以随机初始化或重新训练替代旧权重。

## 编译和执行

```sh
weightc check model.wgt
weightc compile model.wgt -o .cache/weightlang/model
.cache/weightlang/model
wgt run model.wgt
```

源码应声明显式 `fn main()` 作为执行入口。检查/编译不训练、不读取训练数据、不输出模型权重。
训练在 main 中显式调用 `fit` 或 `reinforce`；带 `fn loss` 的 forward 模型也可使用
`train model on (inputs,targets) rate lr clip max_norm;` 执行带全局梯度裁剪的更新，
目标形状和限制见 `docs/reinforcement-learning.md`。推理模型用 `save_model` 保存，
含优化器的断点用 `save` 保存并用 `restore` 恢复，两种文件不能混用。
路径按当前命令帮助和项目配置核对；保留产物写入 `weights/<model>/`，不得覆盖已有产物。

需要 GPU 训练时，当前本地实现仅支持顺序模型通过 `training_device(model,id,memory_bytes,max_batch)`
显式选择设备；`fn forward` 模型训练仍在 CPU，`predict` 也使用 CPU。按目标设备、构建特性和
`docs/wgsl.md` 验证，不把设备枚举当成训练成功，不把初始化权重当成迁移完成。

## 验收

运行完整源码，验证前向输出、最小优化步骤、保存重载和原项目实际入口。
项目提供 `tests/*.wgt` 或兼容的 JSON 测试夹具时，在项目根目录执行 `wgt test`；
测试发现、过滤和独立运行规则见 `docs/native-testing.md`。测试不会替代原项目的实际入口验收。
检查模型参数更新是否符合冻结/共享约定，给出数值误差与尚未实现的能力。
只迁初始化架构时明确说明未经训练；只有组件转换时不能称整个项目迁移完成。
使用本仓库 `examples/native_training/` 和 `examples/self_modifying/` 的当前示例，
不复制历史示例或未经当前解析器验证的提案。
