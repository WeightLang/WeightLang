# WeightLang bootstrap：第一个本地实验

这是用 `.wgt` 定义并从零训练的 **142,344 参数小模型**，用于验证训练、保留评测、
模型保存和 Open Responses 文本接口的完整流程。它目前会套用训练模板，不能当作
懂 WeightLang 的编程助手。整个 bootstrap 目标还没有完成。

## 训练与评测

```sh
# 在仓库根目录，使用现有 CPU PyTorch 环境。
PYTHONPATH=src .venv/bin/weightc check examples/bootstrap/model.wgt --shapes
PYTHONPATH=src .venv/bin/python examples/bootstrap/train.py \
  --output weights/bootstrap_new --steps 300
PYTHONPATH=src .venv/bin/python examples/bootstrap/check_generated.py \
  --report weights/bootstrap_new/report.json \
  --output weights/bootstrap_new/runtime-report.json
```

`--output` 必填且必须是新目录，不支持覆盖；`--steps` 默认 300，必须为正整数。
CPU 四线程，seed 42，batch 8，AdamW，学习率 0.002，weight decay 0.01，梯度范数
裁剪至 1。按验证损失选模；保存后检查 logits 完全一致、改变未来 token 不影响前缀。
模型为两层四头因果 Transformer，64 维隐藏状态，SwiGLU 中间维度 128。
没有导入预训练权重，没有下载语料。

输入使用 UTF-8 字节及独立角色标记，共 264 个 token；上下文 384 token。任意中文
可以编码，但这不代表模型学会了中文。训练数据是英文合成提示与 WGT 答案，监督
只覆盖答案及 EOS。推理使用贪心解码，没有 KV cache，不进行滑动窗口截断。

原创合成任务包括生成完整 WGT 程序和推导输出类型。训练 72 条（relu/sigmoid/gelu），
验证 24 条（silu），独立测试 24 条（tanh）；按整个激活算子族隔离，测试集不参与
选模。各组仍共享同一合成模板，不能推广为真实项目能力。预先设定测试精确正确率
至少 80%、代码解析率至少 90%，未通过时 `report.json` 的 `accepted` 为 false。
评测正确性使用严格文本匹配，可能拒绝其他等价程序；解析率不等于任务正确率。

首次完整实验保存在 `weights/bootstrap_v3/`：

| 指标 | 随机初始化 | 300 步后 |
| --- | ---: | ---: |
| 24 条测试任务精确正确率 | 0/24 | 12/24 |
| 12 条代码任务解析通过率 | 0/12 | 10/12 |

12 条类型推导全部正确，代码任务全部未精确满足要求：常把 tanh 替换成 relu，
部分输入维度也错误。**验收未通过**。训练循环约 11.3 秒，不含启动、基线生成、
保存和最终逐 token 评测；这不是跨机器性能基准。

实验目录保留初始化制品、已训练制品、字节 tokenizer、数据划分、验收门槛、
训练/评测报告、协议报告和规范快照。最初两个失败尝试只保留初始化模型及失败说明，
不是训练成功的模型。实验期间仓库存在其他并行修改，`source-snapshot-status.json`
记录了语法源码漂移；不能声称整个实验已拥有完整冻结的源码快照。

## 本地 Responses 服务

```sh
# 仅用于本地服务的访问令牌，不是 OpenAI 云端密钥。
export WEIGHTLANG_API_KEY='choose-a-local-token'
PYTHONPATH=src .venv/bin/python -m weightlang.responses \
  --model weights/bootstrap_v3/trained --port 8000
```

默认仅监听 `127.0.0.1`，串行处理请求；这是实验服务器，未部署或发布。
`--model` 是完整制品目录，`--model-name` 默认 `weightlang-bootstrap`，端口默认 8000。
发送 `Authorization: Bearer <本地令牌>`，请求体为 JSON，最大 64 KiB：

```sh
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer $WEIGHTLANG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"weightlang-bootstrap","input":"WGT input f32[2], linear(out=4) -> tanh. Output type only.","max_output_tokens":32,"store":false}'
```

已实现的范围：

- `input` 字符串，或 system/developer/user/assistant 文本消息列表；`instructions`
  加在输入前。支持回传本服务的输出消息，服务端不保存会话。
- `stream` 默认 false；true 时逐步输出 SSE 事件。EOS 为 completed，达到输出
  预算为 incomplete，生成异常为 failed；断连停止本次流式生成。
- `max_output_tokens` 默认 128，可填 1..4096，但提示与预算之和必须不超过模型
  的 384 token 上下文；超出返回 400。这里一个字节是一个 token，还计入角色标记。
- `temperature` 只支持 0，默认 0；`store` 只支持 false，默认 false。
- usage 按实际输入与生成 token 计数，包含生成 EOS；不把 UTF-8 字节数说成汉字数。
  输出截断在不完整 UTF-8 序列时使用替换字符，不虚报完整 Unicode 输出。

工具调用、JSON Schema 输出、`previous_response_id`、多模态及其他参数仍未实现，
会明确报错。能接收角色和历史仅证明传输能力，训练集没有多轮任务，不能据此声称
模型具有多轮指令遵循能力。当前没有 Chat Completions 接口。

## 协议验证

依据 [Open Responses 2026-04-24](https://www.openresponses.org/specification) 的
JSON schema，已用真实制品和 `openai==2.6.1` 验证非流式响应、SDK 流式助手、文本
增量、终态、usage、历史回传及拒绝不支持参数。**这是文本子集验证，不是完整协议
合规认证**。SDK 的依赖组合出现过 Pydantic 序列化警告，但 schema 和断言均通过。

可用安装了 `openai==2.6.1`、`jsonschema`、PyTorch 和 WeightLang 的 Python 环境重跑：

```sh
PYTHONPATH=src .venv/bin/python examples/bootstrap/verify_api.py \
  --model weights/bootstrap_v3/trained \
  --schema weights/bootstrap_v3/openresponses-2026-04-24.json \
  --report weights/bootstrap_v3/protocol-recheck.json
```

报告路径必须不存在。验证脚本自动启动随机本地端口并在结束时关闭，不访问 OpenAI
云端。下一阶段需要扩大有执行证据的语言任务语料，解决复制算子与维度的失败，再
训练多轮、工具调用与结构化输出；不能只扩展接口后就宣称模型具备对应能力。
