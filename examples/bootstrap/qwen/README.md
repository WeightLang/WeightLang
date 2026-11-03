# Qwen3.5-4B bootstrap

从官方 `Qwen/Qwen3.5-4B` 预训练权重继续，固定 revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。不从零训练大模型。
`model.wgt` 用现有 `torch-dynamic` 前端实现文本网络：24 层 Gated DeltaNet、
8 层门控 GQA、部分 RoPE、SwiGLU、offset RMSNorm 和共享 embedding/LM head。
这不是此前 `wgt architecture 1` 的不可执行清单，也不调用 Transformers 模型 forward。

输入为无 padding 的 `[batch,time]` 整数 token。只支持默认 RoPE、SiLU、无偏置
及共享词嵌入的文本配置。暂不实现视觉编码器、padding、自定义位置或 KV cache。
`check --shapes` 只验证动态源码，数值证据来自 `tests/test_bootstrap_qwen.py`：
与 Transformers 5.7.0 的小配置比较长度 1/7/65 的前向、梯度和因果前缀。
小配置只用于测试；真正的实验加载完整 4B 预训练文本权重。

第一轮结构改动在 final norm 后加入 `x + up(silu(down(x)))`，默认 rank=32。
新增分支零输出初始化，冻结基座，先验证继承权重、真实更新和独立评测链路。
这只是第一步结构改造，不是全参数训练；合成任务结果不能代表专用语言模型完成。

```sh
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python -m pytest tests/test_bootstrap_qwen.py -q
PYTHONPATH=src .venv/bin/weightc check examples/bootstrap/qwen/model.wgt --shapes
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python examples/bootstrap/qwen/train.py \
  --upstream .cache/weightlang/qwen-bootstrap/upstream \
  --output weights/bootstrap_qwen_new --steps 8 --rank 32 --lr 0.0002
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python examples/bootstrap/qwen/evaluate.py \
  --experiment weights/bootstrap_qwen_new --output weights/bootstrap_qwen_new/generation.json
```

本机兼容依赖已有 Transformers 5.7.0、tokenizers 0.22.2，使用上面的局部 PYTHONPATH，
不改变全局环境。训练需要 CUDA；基座每次只将一层放入 GPU，缓存冻结网络的输出特征，
Python 负责分词、数据、优化器与设备调度，WGT 定义网络。损失只计算 assistant answer。
训练/验证/测试按激活算子族隔离；验证损失选步数，保留测试仅报告。
启动训练前核对固定 revision 的 checkpoint/config/tokenizer SHA256，拒绝静默换基座。
完整文本参数 key 集合也必须匹配 WGT 定义。

每次需要一个全新输出目录。baseline/candidate 包含原始完整 checkpoint 分片、
配置、tokenizer、可执行 WGT 源码、残差权重与 SHA256 manifest。分片中保留原视觉权重，
但本实现不执行视觉路径。它们是实验快照，不能直接交给 `weightlang.load` 或旧 ByteModel。
`candidate/surgery.safetensors` 是最后一步，`selected.safetensors` 是验证选中的分支，
可能仍为零步基线；`evaluate.py` 明确加载 selected。optimizer.pt 独立于模型，
只供恢复训练使用，不通过模型加载执行。报告记录真实更新、选择步数、数据和源码摘要。

从上一轮残差继续（基座仍为固定原权重；AdamW 状态重新初始化）：

```sh
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python examples/bootstrap/qwen/train.py \
  --upstream .cache/weightlang/qwen-bootstrap/upstream \
  --resume weights/bootstrap_qwen35_4b_v2/candidate/surgery.safetensors \
  --output weights/bootstrap_qwen_next --steps 24
```

首轮本地证据 `weights/bootstrap_qwen35_4b_v2/`：8 步、163840 个可训练参数，
验证 NLL 1.0942 → 0.7251，测试 NLL 1.0749 → 0.7113，权重保存/重载一致。
但真实贪婪生成失败：`linear(8) -> tanh` 应返回 `f32[8]`，基座输出 `f32`，
候选输出 `f32[4]`。因此 `quality_accepted=false`，不能把损失改善当语言能力通过。
后续复用这些已检查任务时，报告标记为开发回归，不能再称新盲测。

第二轮 `weights/bootstrap_qwen35_4b_v3/` 从第一轮真实分支继续 24 步（累计 32 步），
没有重新初始化基座，也没有重置分支权重。验证 NLL 0.7251 → 0.06285，
已查看任务上的开发回归 NLL 0.7113 → 0.06361。父分支 SHA256 写入 manifest/report，
`parent.safetensors` 保留续训起点；这轮明确重建 AdamW，不声称恢复优化器轨迹。
数据仍只有 6 个训练样本、2 个验证样本和 2 个回归样本，容易过拟合；
必须扩充真实 WGT 任务、建立新的盲测和通用能力回归后，才能判断改造价值。

第二轮实际生成回归已跑完：相同 `linear(8) -> tanh` 问题，续训前分支输出
`f32[4]`，续训后输出正确的 `f32[8]` 并正常结束；证据在 v3/generation.json。
这只是一条已知开发样本的修正，不是新的独立能力验收。

## 内部改造与独立任务集（v4）

新增 `Refinement` / `EditedTail` 是现有 torch-dynamic 前端中的模型声明，不增加语言算子。
`Refinement(width,rank=64,steps=2)` 对 RMS 标准化后的隐藏向量做
`sigmoid(gate(x))*silu(value(x))`，经 `up` 投影写回残差，迭代 steps 次。
一次调用内共享这组参数；两个解码层各有独立分支，up 零初始化保持基线行为。
`EditedTail(c,start=30,rank=64,steps=2,parent_rank=32)` 在第 30/31 层（零起点）
之前分别插入分支，实际穿过冻结的 DeltaNet/GQA/MLP，最后继承 v3 的输出分支。
这轮优化内部新分支及继承分支，共 1146880 个参数，原 Qwen 参数不更新。

Python 缓存冻结前 30 层的完整序列输出；只在经过尾部注意力之后才切出答案位置，
因此梯度不会错误地丢失 prompt 上下文。CPU 磁盘 cache 为实验目录的
`prefix.safetensors`。加载模型不生成数据或启动训练。

`curriculum.py` 提供 54 条训练（含 6 条单独的通用指令回放）、6 条验证、
8 条按程序族隔离的测试和 6 条通用探针。所有 WGT 参考代码实际执行。
形状/参数题、声明补全和修复均有明确任务契约；代码补全不是整项目生成。
生成代码在有限静态子集中执行，与固定种子参考程序比较数值、shape 和轴名，
不会把能够解析等同任务正确。不是自然采集或人工标注的真实项目语料。

```sh
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python examples/bootstrap/qwen/train_tail.py \
  --parent weights/bootstrap_qwen35_4b_v3/candidate --output weights/bootstrap_qwen_tail_new \
  --start 30 --rank 64 --refinements 2 --steps 80 --accumulate 2 --lr 0.0002
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python examples/bootstrap/qwen/evaluate_tail.py \
  --experiment weights/bootstrap_qwen_tail_new --output weights/bootstrap_qwen_tail_new/generation.json
```

训练只用训练集，验证 NLL 选择候选；在选择完成后才评测新测试集及通用探针。
`acceptance.json` 在训练前固定阈值：测试任务正确率至少 75%，代码运行率 100%，
通用探针正确数不下降。测试输出最多 64 token，通用输出最多 24 token。
这只是一组小规模本地验收，不代表通用编程能力或生产可用性。

v4 的 baseline/candidate 用 `weightlang.qwen-tail-experiment` manifest，分别包含
完整原 checkpoint/config/tokenizer、当前 WGT 源码及 `tail.safetensors`。
后者记录两个内部改造及继承分支；候选文件就是验证选中的状态。
`last.safetensors`/`optimizer.pt` 单独保留最后一步，不能与选中步骤的优化器混淆。
旧版 `evaluate.py` 不适用此格式；必须使用 `evaluate_tail.py`。
`generation.json` 是生成验收证据，训练报告的 loss 本身不会把质量状态改为通过。

正式生成评测使用无量化的冻结层 GPU 缓存（预算 3 GiB）及最多 8 条序列的前缀批处理。
仅在末尾填充未来 token，并在返回前丢弃填充位置；这依赖模型已验证的因果性，
不等于支持任意 padding mask。运行前对真实预训练权重执行逐样本/批量探针对照，
记录 BF16 数值差异并要求两种候选的探针贪婪 token 相同。不是逐位相同的计算调度。
中断的逐样本评测保留 `generation-interrupted.json`；完整重跑写
`generation-batched.json`，具体执行脚本另保存在同名 `.sources/` 目录。

v4 已完成真实训练和完整生成评测（`weights/bootstrap_qwen35_4b_v4/`）：

| 指标 | v3 基线 | v4 验证选中候选 |
| --- | ---: | ---: |
| 验证答案 NLL | 1.0618 | 0.5399 |
| 新测试答案 NLL | 1.1504 | 0.7538 |
| 新任务正确数 | 0/8 | 2/8 |
| 代码解析/运行成功数 | 0/4 | 1/4 |
| 通用指令正确数 | 6/6 | 6/6 |

本轮执行 80 步，验证集选择第 60 步。两处内部 gate/value/up 六个张量及继承
分支确实更新；冻结尾部参数逐位未变，保存重载一致。27 项相关测试通过。
实际生成证据是 `generation-batched.json`，其 `quality_accepted=false`：
没有达到训练前约定的 75% 正确率和全部代码可运行门槛。
主要剩余错误是虚构声明语法、遗漏投影后的维度变化，以及参数量计算错误。

`export_feedback.py` 将已检查的六条失败转换成带诊断和已执行参考答案的修复数据：

```sh
PYTHONPATH=.cache/weightlang/transformers-validation:src .venv/bin/python examples/bootstrap/qwen/export_feedback.py \
  --experiment weights/bootstrap_qwen35_4b_v4 \
  --report weights/bootstrap_qwen35_4b_v4/generation-batched.json \
  --output weights/bootstrap_qwen35_4b_v4/repair-feedback.jsonl
```

该文件已经生成，标记为 development，`eligible_for_blind_evaluation=false`；
尚未用它再训练。未来使用这些反馈时必须另外建立盲测，不能把旧测试的修复算新泛化。
