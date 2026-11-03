# Transformers 迁移验收

当前优先级调整为：先扫描原子需求、完善语言与契约，只用少量代表模型验收，暂不追求全库迁移。
见 [原子能力扫描与路线](migrations/primitive-roadmap.zh-CN.md)。以下已验证适配器作为回归资产保留。

固定参考版本：**Transformers 5.7.0**。本项目正在逐项迁移，**没有完成整个库**。
[完整源文件清单](migrations/transformers-5.7.0-inventory.json) 包含 442 个
`modeling_*.py` 文件、SHA256、类名和静态转换的实际失败原因；扫描不等于迁移。
通用 Python 转换器对这 442 个原文件均明确拒绝了不支持的结构，未执行输入文件。
下列 10 个基础模型使用单独审计过的结构适配器转换，没有假装通用转换器已经覆盖 HF。

| 已验证的基础类 | 对应标准库 block | 已覆盖的差异 |
| --- | --- | --- |
| GPT2Model | hf_decoder.gpt2_block | Conv1D 权重转置、tanh GELU、绝对位置、可配置 LayerNorm |
| LlamaModel | hf_decoder.rope_block | GQA、RoPE、RMSNorm、可选 attention/MLP bias |
| MistralModel | hf_decoder.rope_block | GQA、完整注意力和滑动窗口 |
| Qwen2Model | hf_decoder.rope_block | Q/K/V bias，输出投影无 bias，按层滑动窗口 |
| Qwen3Model | hf_decoder.rope_block | 每个 Q/K 头 RMSNorm、可选 attention bias、滑动窗口 |
| GemmaModel（Gemma 1） | hf_decoder.gemma_block | embedding 乘 sqrt(dim)、1+weight RMSNorm、tanh-GELU 门控 |
| OlmoModel（OLMo 1） | hf_decoder.olmo_block | 无 affine LayerNorm、可选 Q/K/V 裁剪 |
| Olmo3Model | hf_decoder.olmo2_block | OLMo2 的数学结构、逐层完整/滑动注意力，默认三层滑动后接一层完整注意力 |
| Olmo2Model | hf_decoder.olmo2_block | 整个 Q/K 投影 RMSNorm、注意力/FFN 输出端 RMSNorm 后残差 |
| Starcoder2Model | hf_decoder.starcoder2_block | LayerNorm、非门控 FFN、可选线性 bias、滑动窗口 |

验证范围：CPU、f32、单个 token 张量、固定且无 padding 的序列、位置从 0 开始、
无 KV cache，返回基础模型的最后隐藏状态。输入不携带外部 attention mask 或 position_ids。
不包含 `ForCausalLM` 等任务头、生成方法、参数绑定、encoder-decoder、多模态、MoE、
量化、训练优化器或预训练权重。这里的 Gemma/Qwen/OLMo 范围不能外推到其他版本；
尤其 **Qwen3.5 尚未迁移**。缩放/部分维度 RoPE 等不支持选项明确报错。

所有基础模型都经过相同参数下的原实现对比：WGT 构建→保存→重载→前向→全部参数梯度。
测试覆盖 batch=1/3、长度 7、2 层小配置（OLMo3 混合注意力为 4 层）、GQA、head_dim 与 hidden/heads 不同的情况，
以及部分 bias/window/padding-index 组合。前向容差 rtol=3e-5/atol=3e-6，梯度
rtol=3e-4/atol=3e-5。参考运行使用 eager attention，dropout=0；不声称已证明随机
dropout 掩码逐值相同或 GPU/低精度等价。另验证生成模型能再次导出为独立 Python。

## 结构转换，不下载权重

```sh
weightc transformers --family llama --config examples/transformers/llama.json \
  --sequence 8 -o /tmp/weightlang/llama.wgt
weightc check /tmp/weightlang/llama.wgt --shapes
# 只有用户需要保留初始化模型时才构建；这不是训练过的 Llama。
weightc build /tmp/weightlang/llama.wgt --no-train --no-store -o weights/llama-scaffold
```

`--family` 必填，取上表对应的 gpt2/llama/mistral/qwen2/qwen3/gemma/olmo/olmo2/olmo3/starcoder2。
`--config` 为本地配置 JSON，`model_type` 若存在须匹配；尺寸和层数从配置读取，
不加载 Python 模型或 checkpoint。`--sequence` 是正整数单样本长度，输出
`i64[time] -> f32[time,feature]`，最大层数 1024。`-o` 必填，已有路径拒绝覆盖。
配置中的 `use_cache` 不改变上述无缓存接口；生成文件和命令报告都明确标注这个接口。
示例配置使用小尺寸，**不是官方完整模型配置或训练好的权重**。

全部示例位于 [examples/transformers](../examples/transformers/README.md)。
标准库实现位于 [hf_decoder.wgt](../src/weightlang/std/hf_decoder.wgt)。
完整算子签名与语法见 [中文参考](reference.zh-CN.md#transformers-迁移所需的基础算子03本地实现)。

## 逐文件审计与复跑

```sh
# 只读安装包源码，给每个原文件做一次静态转换尝试；不会实例化几百个模型。
weightc transformers --inventory -o /tmp/weightlang/transformers-inventory.json
# 可选测试依赖，建议放在项目环境；不修改全局环境。
python -m pip install -e '.[test,transformers]'
python -m pytest -q tests/test_transformers_migration.py tests/test_python_import.py
```

本机验证发现全局 tokenizers/huggingface-hub 不满足 Transformers 5.7.0 声明，
兼容依赖单独安装在 `.cache/weightlang/transformers-validation/`；本机可复跑：

```sh
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python -m pytest -q \
  tests/test_transformers_migration.py tests/test_python_import.py
```

源文件清单保留每个文件的 `not_migrated` 状态，因为整个文件通常包含缓存、任务头等多个类。
`adapter_class` 仅指向已实现的特定基础类，不能把整份文件改标为完成。后续迁移需继续
补多输入/输出、显式 mask、位置输入、缓存状态、共享参数、跨注意力和其他模型专用数学。
必须保存不支持的配置证据，不能靠删除选项、把未知层当 identity 或加载 Python 黑盒过关。

参考来源为本地 Transformers 5.7.0 的十个模型定义及配置文件，路径和哈希在清单中。
这些上游文件采用 Apache-2.0；本实现以公开的数学计算组合成 WGT，验证时使用安装包，
没有复制整套上游 Python 模型到标准库。
