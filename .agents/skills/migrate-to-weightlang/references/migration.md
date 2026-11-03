# 原生 WGT 0.5 迁移路线

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
训练在 main 中显式调用 `fit`/`train`；完整模型用 `save_model` 保存，
含优化器的断点用 `save` 保存并用 `restore` 恢复，两种文件不能混用。
路径按当前命令帮助和项目配置核对；保留产物写入 `weights/<model>/`，不得覆盖已有产物。

## 验收

运行完整源码，验证前向输出、最小优化步骤、保存重载和原项目实际入口。
检查模型参数更新是否符合冻结/共享约定，给出数值误差与尚未实现的能力。
只迁初始化架构时明确说明未经训练；只有组件转换时不能称整个项目迁移完成。
使用本仓库 `examples/native_training/` 和 `examples/self_modifying/` 的当前示例，
不复制历史示例或未经当前解析器验证的提案。
