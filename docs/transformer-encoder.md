# Transformer Encoder：固定形状验收样例

> 状态：新增验收目标。必须以当前分支实际执行的 parser、CPU runtime、AOT 与服务测试结果更新状态。源码可解析不等于前向、反向或训练已经验收。

模型源码：[model.wgt](../crates/wgt-model/tests/fixtures/transformer_encoder/model.wgt)；Rust 验收入口：[transformer_encoder.rs](../crates/wgt-model/tests/transformer_encoder.rs)。

## 架构

该样例遵守当前单向量服务边界：输入/输出各为一个 f32[8]，内部将两个 token 解释为 [sequence=2, hidden=4]。两个 attention head 各有 head_dim=2，将 heads 维折叠为 BatchMatmul 的 batch 维。

完整 Encoder block 顺序：

1. Q/K/V 独立线性投影。
2. 拆分 heads，执行 scaled dot-product attention。
3. 合并 heads 并执行输出投影。
4. Attention residual + LayerNorm。
5. Position-wise FFN：4 → 8 → 4，ReLU 激活。
6. FFN residual + LayerNorm。
7. 展平回服务要求的 f32[8]。

当前实现使用固定长度、显式 reshape/permute 和 batch_matmul，不依赖动态 batch、隐式广播或通用多轴索引。Attention 缩放系数针对 head_dim=2 固定为 1/sqrt(2)。

## 合同与限制

- Shape/轴名是编译期元数据；实际数据仍为 row-major f32。
- permute、.T 及可被 view-aware kernel 消费的路径可共享存储；AOT/其他算子是否物化必须分别验证，不能由语法推断零拷贝。
- batch_matmul 的 batch 维要求相等，不支持 broadcasting。
- softmax(scores, 2) 沿 key sequence 轴分别归一化。
- 参数是确定性初始化值，不是预训练权重。
- 当前样例无 dropout、mask、位置编码、动态序列长度、任意 dtype 或 GPU 融合 kernel。
- 服务接口仍是单个 f32 向量；这不是多输入/动态 batch forward API。

## 验收

定点检查命令：

```sh
cargo test -p wgt-model --test transformer_encoder
cargo test -p wgt-language --test tensor_source_contracts
cargo test -p wgtime-core --test tensor_contracts
cargo test -p wgtime-core --test compiled_forward
```

真实 daemon 验收需显式启用 ignored service test，并设置项目要求的 WGTIME_TEST_DAEMON。验收需检查源码解析、Rust include macro 一致性、native-cpu forward、训练后 MSE 下降、safetensors 回载一致和实例释放。

CPU interpreter 与 AOT 数值一致、输入梯度及所有可训练参数梯度的有限差分，仍需独立增加 core 层数值回归；不能从 service smoke test 推断这些项目通过。
