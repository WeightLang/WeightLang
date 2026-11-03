# Bootstrap：理解 WeightLang 的专用语言模型

状态：项目目标与验收约定，尚未完成。本文不代表已经训练出专用模型或发布兼容服务。

本地第一阶段已开始：[bootstrap 实验](../examples/bootstrap/README.md) 实现了小模型
训练、独立评测与 Open Responses 文本/SSE 子集，已用真实权重验证 SDK。
首次完整实验尚未通过语言能力验收；工具调用和结构化输出仍是待实现目标。

WeightLang 的 bootstrap 目标是：用本语言描述、训练和迭代一个理解本语言的
专用语言模型，使它能够解释、编写、修改和调试 `.wgt` 程序，再用实际编译与
运行结果检验它的输出。这里的 bootstrap 与 `wgt sync` 安装运行环境无关。

## 模型能力

- 理解当前实现的语法、类型、轴与形状、标准库、权重复用和训练语义。
- 根据自然语言生成可解析、可运行的 `.wgt`，解释形状变化和权重共享。
- 根据编译错误、运行结果和测试反馈修复程序；不把未实现的提案当作语法。
- 保留通用指令遵循、多轮对话、工具调用和结构化输出能力，避免专用训练后
  只能续写代码。具体模型规模、基础模型和训练路线由实验决定。

训练材料以当前实现、[中文语言参考](reference.zh-CN.md)、标准库和经过验证的
示例为依据，记录语言版本与源码提交。按任务或程序族隔离训练集和保留评测集，
避免把同一程序改个名字就当作未见样本。损失下降不能代替能力评测。

## 唯一对外协议目标

对外提供 **Open Responses 协议，兼容 OpenAI Responses API 客户端**，入口为
`POST /v1/responses`。规范依据是 [Open Responses](https://www.openresponses.org/specification)，
它基于 OpenAI Responses API。完整兼容范围必须由实现和协议测试明确，不能仅凭路径
相同就宣称完全兼容。不额外要求 Chat Completions、Anthropic Messages 或自定义协议。

以下是 bootstrap 的目标范围，不是当前已有功能：

| 能力 | 验收要求 |
| --- | --- |
| 文本请求 | `model`、`instructions`、字符串或消息列表形式的 `input`、`max_output_tokens`；保留角色与消息顺序 |
| 普通响应 | 返回 Response 对象、输出消息及 `output_text` 内容项、状态与真实 token usage |
| 流式响应 | `stream: true` 使用 SSE；输出文本增量、内容项生命周期和成功、失败或未完成的终态 |
| 多轮对话 | 首先支持客户端回传历史 input/output items；服务端保存历史及 `previous_response_id` 单独声明支持状态 |
| 工具调用 | 支持函数定义、`function_call`、`function_call_output`，以 `call_id` 对应调用与结果；执行由宿主负责 |
| 结构化输出 | 支持 `text.format` 的 JSON Schema 子集，明确支持范围；不能把提示模型输出 JSON 当成严格约束 |
| 错误处理 | 对不支持的参数、工具类型、模态和非法输入给出明确错误，不能静默忽略 |

HTTP/JSON、SSE、会话状态和参数校验属于推理服务层；模型负责文本生成与工具
决策。协议支持不要求把 HTTP 或 API 参数加入 `.wgt` 语言语法。多模态、托管工具、
WebSocket 和后台任务不作为首轮 bootstrap 的前置条件，也不宣称已经支持。

协议实现依据：
[Responses API](https://developers.openai.com/api/reference/responses/overview)、
[流式事件](https://developers.openai.com/api/docs/guides/streaming-responses)、
[函数调用](https://developers.openai.com/api/docs/guides/function-calling)、
[结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)。
实现时应记录所依据的规范快照及 SDK 版本，并保留请求、响应与事件序列的契约测试。

## 完成标准

1. **可复现训练**：保留模型结构、tokenizer、数据来源与划分、训练配置、随机种子、
   权重和报告。初始化制品与经过训练的权重必须分别标明。
2. **理解语言**：在保留任务上报告生成程序的解析率、运行成功率、任务正确率和
   修复成功率，与训练前基线对比。训练前固定样本和通过阈值，不在看过结果后修改。
3. **现代交互**：验证多轮指令、工具调用往返和结构化输出；报告专用训练前后的
   能力变化，而不是仅测试服务能否返回 JSON。
4. **客户端可用**：用明确版本的 OpenAI SDK 和自定义 `base_url`，验证普通请求、
   SSE、工具往返、输出截断、错误及断连处理。模拟后端只证明协议逻辑；最终验收
   必须连接实际训练出的模型。
5. **证据可追溯**：分别标注本地实现、训练结果、协议验收和已发布状态。

模型、基线、候选与评测报告按 [存储约定](storage.md) 保存在
`weights/<experiment>/`，保留完整模型目录，不覆盖已有模型。现有
[`tiny_lm`](../examples/tiny_lm/README.md) 可供训练与生成流程参考，不能作为
已经实现本目标的证据。
