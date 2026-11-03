# 小尺寸 Transformers 架构

这 10 对 `.json` / `.wgt` 文件用于检查迁移与标准库组合。
配置均为小尺寸、2 层（OLMo3 为 4 层）、8 token，WGT 不含权重值或训练声明；不能用于声称预训练模型质量。
输入是单样本 token 序列（不含 batch），输出是基础模型最后的隐藏状态，不是词表 logits。
各编号 block 的参数独立；只保留标准库定义一次，不逐层复制整个 block 正文。

```sh
weightc check examples/transformers/qwen3.wgt --shapes
weightc transformers --family qwen3 --config examples/transformers/qwen3.json \
  --sequence 8 -o /tmp/weightlang/qwen3.wgt
```

目录包含 GPT2、Llama、Mistral、Qwen2、Qwen3、Gemma 1、OLMo 1、StarCoder2。
没有 Qwen3.5。模型名称不代表完整 API 已迁移；接口与数值验证范围见
[迁移记录](../../docs/transformers-migration.md)。

`olmo2.wgt` 使用完整 Q/K 投影归一化和输出端 RMSNorm；它与 OLMo 1 的归一化位置不同。

`olmo3.wgt` 前三层使用 window=3，第四层完整注意力，以 8 个 token 检验实际窗口截断；
四层权重独立，文件仅含架构，不含训练权重值。
