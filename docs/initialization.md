# 模型初始化

当前原生声明式模型通过 `init seed=N` 指定初始化随机种子。`N` 为非负 u64，模型必须且只能声明一次 `init`；调用模型构造函数时才初始化参数。线性层权重采用 Xavier uniform，偏置为零。

```wgt
wgt 0.5
model Example
# 单样本输入和输出都为一个 f32 特征，不含 batch 轴。
input x: f32[1]
output y: f32[1]
layer projection: linear(out=1)
init seed=42

fn main() {
    # 此处只初始化，predict 不训练这个顺序模型。
    let net = Example("sgd");
    return predict(net, [1]);
}
```

构造时优化器参数必填，普通顺序模型支持 `"sgd"` 和 `"adam"`。两个构造调用创建独立模型；复用同一个句柄会共享参数和优化器状态。`weightc check` 不构造模型或训练，保存必须显式调用 `save_model` 或 `save`。初始化不等于训练；保存格式和覆盖保护见[原生训练](native-training.md#保存与续训)。

使用 `fn forward` 的自更新模型使用显式 `parameter` / `frozen` 初值，不能与 `layer` / `init` 混用。其构造和状态规则见[语言参考](reference.zh-CN.md#自更新前向函数)。

概率参数可写成 `parameter p: prob = 0.7`：初值必须严格位于 `(0,1)`，训练的是内部 logit，读取时得到概率。固定概率可写 `frozen p: prob = 1`，允许精确端点。普通 `f32` 权重不会默认限制到 `[0,1]`；目前也没有 `{0,1}` 布尔参数类型或自定义参数范围声明。概率参数不能直接作为 `linear` 权重。完整约束见[概率参数](differentiable-programs.md#概率混合自定义损失与采样语法)。

`[0,1]` 包含零，不能单靠限制权重范围排除除零。普通脚本数值运算遇到除零、对零取余或非有限结果时会报错，不会自动替换除数；调用者仍需选择适合问题的非零分母或显式处理条件。
