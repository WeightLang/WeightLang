# 原生 OpenResponses 客户端与 harness

本文记录协议研究、已有候选代码与待定契约，尚未发布。用户最新架构要求是：可复用执行核心由机器级唯一 `wgtime` service 进程宿主；WGT 编译程序仅为轻量 IPC client，不内嵌推理核心或创建 GPU context，连接失败不得回退到内嵌运行时。HTTP、OpenResponses 会话和工具循环明确保留在客户端 WGT 控制层；`wgtime` 仅负责权重 forward/backward、参数更新及其资源预算。客户端控制层不因此获得内嵌权重执行的回退路径。

协议核心已通过首批本地测试；WGT bridge、真实客户端 AOT 与多 agent 没有完成整体验证，不能宣称当前可用。下面接口是客户端控制层候选契约，必须完成控制层/模型执行核心的依赖分离与 IPC 接线验证。

协议基线为 OpenResponses `2026-04-24`，OpenAPI 3.1.0；官方源码固定到
[`dce9ddb91921427c2ba06901c7dfaf0cc7056799`](https://github.com/openresponses/openresponses/tree/dce9ddb91921427c2ba06901c7dfaf0cc7056799)。
[规范](https://www.openresponses.org/specification)、[参考](https://www.openresponses.org/reference)、
[版本化 OpenAPI](https://github.com/openresponses/openresponses/blob/dce9ddb91921427c2ba06901c7dfaf0cc7056799/public/openapi/2026-04-24/openapi.json)。
该 OpenAPI SHA-256 为 `d598753c3a86fd8a2434828fe39dcc8786a7a694bbe46080a0d24c9fa40e72df`。

这是 Rust 原生 HTTP 客户端与 WGT 工具循环，不是 provider 服务器。现有 `agent/environment` 是模型策略采样与学习器，不等同于这里的 LLM 会话。没有 Python、SDK 回调或任意 shell 工具。

## 单服务架构与客户端控制层边界

`wgtime` service 的跨用户单例、启动互斥、连接已运行实例与本地 IPC 认证由统一运行时架构定义，harness 不另启动第二个服务。客户端不能提交任意 profile、原始句柄或猜测的 session ID 获得其他用户认证与会话。连接失败必须报错，不回退当前候选的进程内 Runtime。

provider HTTP、认证、会话历史、pending calls 和显式 WGT 工具循环归客户端控制层所有。工具调用权重模型时必须通过 `wgtime` IPC，不能转向客户端内嵌模型执行。网络/工具账本在客户端所有 worker 间共享；推理预算由服务另行计账，二者不是已经实现的统一总沙箱。失败 POST 与副作用工具不会因为 IPC 重连而自动重放。

目标依赖方向是客户端控制层依赖轻量 `wgtime-client`，后者依赖纯 IPC wire；权重/设备执行核心只由 service 链接。当前候选 bridge 尚位于 weightlang-runtime crate，不能据此声称最终编译程序已排除模型执行核心；抽离与依赖图/产物验证仍待统一架构接线，禁止循环依赖。

无论最终归属如何，单个 harness 的子 worker 必须共享总额度，用户会话与 mutable state 必须隔离；共享权重注册/复用键不允许合并 harness 历史、工具权限、随机数或可变训练状态。不能将权重实例共享等同于会话共享。IPC 消息需要携带不可伪造的归属/租约信息并由 service 校验，不能仅信任客户端数据中的 user/parent/session 字段。

固定标准库 `std/openresponses/harness` 的导入与下面 responses_* API 在此架构下均待重新接线、未实现完成。现有候选代码的纯数据参数限制和本地 session 索引只是进程内保护，不证明跨用户 IPC 隔离。

## 实现范围

当前代码提供 POST JSON 与 SSE、函数工具注册和结果回传、会话历史、共享请求/字节/工具/回合/并发预算。HTTP 客户端禁用代理和重定向；不自动重试 POST。WebSocket、主动 `/responses/compact`、background、previous_response_id 状态续接不在首期范围。

输入可为文本或 items。输出保留 message、function_call、function_call_output、reasoning、compaction 和带 provider 前缀的扩展 JSON。未知字段保留；未知工具类型不执行。客户端不下载输入 URL、不隐式读取本地文件。图像/文件数据的结构化传送不代表客户端实现了相应模态推理。

首期仅验证客户端所依赖的 envelope 字段，不提供完整 OpenAPI schema 验证。输入参数与输出资源 required 字段不同，不把输出必填 id/status 强加给输入。

## WGT 接口契约

以下是本地新增接口，整体行为仍待解释/AOT 测试闭合。普通函数与对象/列表表达式复用现有 WGT 0.5 语法，不新增 agent/tool 声明。

| 签名 | 目的与约束 |
| --- | --- |
| `responses_scope(limits, profiles, tools)` | 同步父函数一次性建立执行级 harness。profiles/tools 是字符串列表；读取执行时 profile，创建共享账本，返回 null。不得在 async 内重建。 |
| `responses_restrict(profiles, tools)` | 在当前 worker 创建任何 session 前缩减继承的端点与工具集合，返回 null；不能增加父权限。 |
| `responses_session(request)` | 创建当前 Runtime 独占的会话句柄。request 是对象，model 为非空字符串，input 为字符串或 items；返回句柄。最多保留 64 个 session。 |
| `responses_next(session, profile)` | 仅在 async fn 中创建请求 future，用 `await` 驱动；profile 是当前 grant 内的别名。pending 工具未提交或 session 已结束时拒绝。 |
| `responses_submit(session, results)` | 提交本回合所有工具结果，返回 null；results 是 `{call_id, output}` 对象列表，未知/重复/缺失 ID 整体拒绝。 |
| `responses_cancel(session)` | 关闭当前会话，清空 pending 调用，返回 null；不会回滚外部副作用。 |
| `responses_usage()` | 返回所有 worker 共用的累计 requests/bytes/tools/rounds/inflight。 |

所有参数必填，无隐式默认额度。`limits` 字段为 `requests/bytes/tools/rounds/concurrency/deadline_ms`，全为非负整数；concurrency 为 1..64，deadline_ms 为 1..30000。请求与回合目前一一对应；bytes 计发送 JSON 与实际接收正文。普通脚本操作预算依旧生效。scope 原有 30 秒上限也生效。

请求默认由 session 强制设置 `store:false`、`background:false`；拒绝非空 previous_response_id。工具参数 schema 缺省为不允许字段的空对象。客户端单请求正文上限 4 MiB，SSE 单事件上限 1 MiB、事件总数上限 10000；工具结果文本上限 1 MiB。实际剩余额度可能更小。

`await responses_next` 成功返回对象含 `status/response/events/calls/usage`。
status 为 `tools` 时 calls 每项含 call_id/name/arguments；为 `completed` 时正常结束；provider failed/incomplete 返回对应状态且不调工具。传输、协议、schema、权限和预算错误目前遵循脚本失败契约，不是可捕获异常。错误输出不包含认证材料。

工具仅按注册 function schema 和显式白名单调用。脚本用具名函数与分支映射名称，不将模型字符串传给任意动态 dispatch。调用关系使用 wire 的 `call_id`；item `id` 不是调用对应 ID，协议没有另加 `tool_call_id` 字段。完整成功 terminal 之前不暴露可执行 calls。

## WGT harness 循环示例

下面的完整程序是当前接口候选例子，尚待整体测试；不会执行网络直到显式运行 main。`sum` 是唯一明确映射的工具，模型无法选择其他 WGT 函数。循环回合由执行级 ledger 限制，两个 worker 的配额不分别重置。

```wgt
wgt 0.5
fn add_values(arguments) { return arguments.x + arguments.y; }
async fn harness(prompt) {
    responses_restrict(["local"], ["sum"]);
    let session = responses_session({
        model: "fake-model", input: prompt, stream: true,
        tools: [{type: "function", name: "sum", strict: true,
            parameters: {type: "object",
                properties: {x: {type: "integer"}, y: {type: "integer"}},
                required: ["x", "y"], additionalProperties: false}}]
    });
    while true {
        let step = await responses_next(session, "local");
        if step.status == "completed" {
            responses_cancel(session);
            return step.response;
        }
        assert step.status == "tools", "provider did not complete";
        # 本例逐次只接受一个显式注册工具，拒绝静默丢失其他调用。
        assert len(step.calls) == 1;
        let call = step.calls[0];
        assert call.name == "sum", "tool is not mapped";
        responses_submit(session, [{call_id: call.call_id,
            output: add_values(call.arguments)}]);
    }
}
fn main() {
    responses_scope({requests: 8, bytes: 200000, tools: 4,
        rounds: 8, concurrency: 2, deadline_ms: 30000},
        ["local"], ["sum"]);
    tasks {
        let a = spawn harness("Add 1 and 2");
        let b = spawn harness("Add 3 and 4");
        let first = await a;
        let second = await b;
        return [first, second];
    }
}
```

执行环境 profile 示例为 `{"local":{"url":"http://127.0.0.1:18080/responses"}}`；有认证时增加 `authorization_env`，值是执行时环境变量名。该 profile 应由宿主建立，不从模型文本生成。标准库可复用普通函数控制流；目前没有完成通用可导入 harness 模块，不能将上例称为已经安装的标准库 API。

标准库接线使用 `import harness from "std/openresponses/harness"` 的固定导入，仅提供普通脚本函数，明确拒绝在 forward/loss 内调用。导出 `harness.next(session, profile)`、`harness.submit(session, results)` 与 `harness.finished(step)`，负责状态/结果辅助，绝不默认执行工具。完整循环与显式 add 映射放在 `examples/openresponses/addition/model.wgt`。普通脚本库声明 `module NAME`，可通过逻辑标准库名或显式 `./module.wgt` 本地开发路径导入，支持同一普通/async 函数能力；不能在 forward/loss 调用。没有泛 dynamic dispatch。现有 `operators` 导入是纯张量库，不能把 async/副作用循环混入其中。第一版保留用户具名工具分支，库负责协议回合辅助，不做模型控制的任意函数派发；这一导入边界在实施前需同步到主参考与技能，且需真实解释/AOT验证。

## 工具 JSON Schema 子集

首期不是完整 JSON Schema dialect 实现，不声称符合整个 Draft 2020-12。只接受下列离线子集；任何其他 keyword（含 `$schema/$ref/$defs/oneOf/anyOf/pattern`）注册时拒绝，不访问网络或文件。

| 内容 | 首期支持 |
| --- | --- |
| object schema | 支持，最大嵌套 32；boolean schema 拒绝 |
| type | 单一 object/array/string/number/integer/boolean/null；类型数组拒绝 |
| properties | 属性名到子 schema 的对象 |
| required | 字符串列表，校验必需字段 |
| additionalProperties | 仅 boolean；省略允许额外字段，false 拒绝未声明字段 |
| items | 单个 object schema，逐元素校验；tuple/prefixItems 拒绝 |
| enum | 非空列表，按 JSON 值相等判断 |
| minimum/maximum | 含端点数值上下限；exclusive/multipleOf 拒绝 |
| minLength/maxLength | 非负整数，按 Unicode 字符数计 |
| minItems/maxItems | 非负整数，按元素数计 |
| title/description | 注释信息，无执行权限 |

常见 strict 对象工具可使用 type/object/properties/required/additionalProperties:false。provider `strict:true` 不替代本地参数校验；不支持的 schema 不静默弱化。函数 arguments 是完整 JSON 字符串，全部接收后再解析、验证。结果 output 为字符串或被序列化为 JSON 字符串的数据。

## SSE 生命周期

增量解码支持跨块 UTF-8、CR/LF/CRLF、多行 data 和注释。event 名与 JSON type 必须一致；sequence_number 递增，不要求连续或起于零。item added 在前，done 在后，完成 item 不能再修改。

必须收到 response.completed/response.failed/response.incomplete 之一。terminal 后无 `[DONE]` 合法；单独 `[DONE]`、缺 terminal EOF 或半个事件 EOF 都报错。completed 状态与 terminal 类型一致。完整 response 验证后才允许工具闭环。

核心事件覆盖目标包括 response created/queued/in_progress、output_item added/done、content_part added/done、output_text delta/done/annotation.added、refusal delta/done、function_call_arguments delta/done、reasoning delta/done、reasoning_summary_part added/done、reasoning_summary_text delta/done、error 和三种 terminal。当前代码已增加 content added/done、文本 delta/final、function arguments delta/done、item done 与 terminal output 一致性检查；已覆盖跨块中文 UTF-8 与不一致 arguments/缺失 item 事件的负例。尚不声称对全部事件 schema 做完整 OpenAPI 验证。

## 授权、隔离与取消

执行时设置 `WGT_RESPONSES_PROFILES` 指向不超过 1 MiB 的 JSON 对象。键是 profile 别名，值含 `url` 和可选 `authorization_env`；后者引用执行进程的认证环境变量。仅 HTTPS 或 loopback HTTP；URL 不含 userinfo/query/fragment。profile 和认证不进入 WGT 源码或编译制品。没有授权头时支持本地假服务。

`WGT_HOST_POLICY` 或 compute 上下文活跃时首期拒绝 harness，避免绕过尚未接通的共享 Host POST 授权；不通过重新读取策略或复制资源获得新额度。这不等于已经支持 Host/harness 组合。

父同步函数先建 responses_scope，再在 tasks 内 spawn async worker。worker 继承安全 profile 引用与收缩 grant，创建自己的 session；session 句柄不是纯数据，不能跨 async 参数或返回值传递，也不从数字恢复。所有 worker 共用同一账本和绝对 deadline。子 worker 不能重新 responses_scope 刷新预算。父子结果仅传普通数据，不自动复制全部历史。

scope 错误取消兄弟任务，退出等待在途阻塞 HTTP 完成和计账；合作取消不能立即打断 OS read，最多等待剩余请求 timeout。已发生的外部工具副作用不回滚。当前不支持 async 内 spawn、嵌套 tasks、递归 agent 树；平级并发受 tasks 和 harness 双重上限约束。

## 验证状态

首批独立 Rust 核心测试通过 5 项，包含 HTTP 工具结果往返、call_id 拒绝、terminal、schema 与共享 grant/预算。接入公共模块后重新执行 `cargo test -p weightlang-runtime --no-default-features openresponses --lib` 为 6 passed，`cargo test -p weightlang-runtime --no-default-features --test openresponses_core` 为 4 passed；`cargo check -p weightlang-runtime --no-default-features` 通过。

`cargo test -p weightlang-wgt --test openresponses -- --nocapture` 已在构建脚本路径修复后通过 1 项：本地分块 SSE、双 worker、具名 WGT 工具执行、call_id 回传、共享请求/工具额度，以及移除 model.wgt、PATH=/nonexistent 后的真实 AOT。此为现有控制层的中间验证，尚未证明最终轻量客户端已排除权重执行核心。固定官方 ResponseResource fixture 已加入 roundtrip 测试。

async 对象构造另修正为同步/异步共用借用字段校验、逐节点直接扣 ledger，验证后再 clone；失败前已访问节点照常收费。`record_callbacks` 定向测试通过，覆盖失败计费与两个线程竞争同一原子额度。

官方 compliance runner 面向 provider 服务端，首期只复用固定官方 schema/fixtures/场景作本地客户端契约测试，不宣称通过服务端官方认证。不需要付费 API 或真实密钥。
