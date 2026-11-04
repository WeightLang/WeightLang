# 原生 agent、environment 与整批 learn

WGT 0.5 本地实现支持专用声明和同步实例；尚未发布。完整可运行监督例子为
[agent_environment/model.wgt](../examples/native_training/agent_environment/model.wgt)。
它不要求奖励、PPO 或 GAE，也不使普通宿主函数自动可微。

```wgt
agent Predictor { policy; learner: update; }
environment Counter {
    observation: f32[1]; action: f32[1]; feedback: f32[1];
    reset: reset_counter; step: step_counter;
}
```

声明和函数共用同一 token/parser、版本规则及行号。名称不能与函数、模型、内置或其他声明冲突。
声明字段不可重复，不接受未知字段；各向量宽度为整数 1..16384。声明本身不初始化模型、不训练、无环境副作用。
方法只对专用句柄解析，不提供任意对象方法、继承、闭包或函数值。

## agent

`agent NAME { policy; [critic;] [learner: FUNCTION;] }` 中 policy 必填。
无 critic 时构造签名为 `NAME(policy)`，有 critic 时为 `NAME(policy,critic)`。
两者必须是已有实际模型句柄，保留别名，不复制权重或优化器状态；构造不训练。
`actor.policy`、已声明时的 `actor.critic` 取出原句柄。
`actor.act(observation)` 只抽取零基类别动作，复用模型采样、单次 RNG 和不提交 forward 写入的契约。
需要同次 log probability 时，先 `let policy=actor.policy;`，再
`let {action,log_prob}=sample from policy(observation);`；sample callee 仍要求局部变量。
单标量回归模型不应通过 act 解释为互斥动作，直接调用取出的 policy。

## 整批 learn

`learn actor { BODY }` 使用 agent 声明的默认 learner；`using other` 显式覆盖。
没有默认 learner 时明确报错；普通模型必须 `learn model using learner { BODY }`。
赋值形式为 `let report = learn actor { BODY };`，非赋值形式可省末尾分号。
learner 静态解析为同步用户函数，恰好两个位置参数；不能用内置、异步函数或闭包替代。
首参数是原始 target：agent target 传 agent，普通模型 target 传模型；第二参数为完整记录批次。
learner 返回普通 Value 作为 report，可执行零次、一次或多次显式训练/自定义提交；learn 不自动优化。
自定义算法状态继续使用 gradient_snapshot/commit_gradients 的模型状态，不另建状态副本。

BODY 复用 collect/yield，字段 shape、预算、嵌套作用域、控制流相同。
成功完成采集后先释放目标 policy 的活实例保护，再恰好调用一次 learner；空批次同样调用一次。
采集失败或 return/break/continue 穿透时不调用 learner，保护可靠释放。
保护按模型 identity 对所有别名生效；agent 只保护 policy，独立 critic 和其他模型可更新。
保护期间拒绝训练、GPU 训练入口、自定义 commit、优化器重置、reload、weightfs 参数写入及含 Write IR 的 forward。
无写入 forward、只读采样/evaluate、梯度计算和只读保存允许；IR 含 Write 时即使分支未执行也拒绝普通 forward。
同目标嵌套 learn 拒绝。保护不跨快照/新实例复制，不在整个采集期间持模型 mutex。
环境 IO、文件副作用和普通脚本 RNG 不属于模型事务，失败不承诺回滚它们。
Rust `AdaptiveSession::set_optimizer` 现在返回 Result，保护失败不重置状态，调用者必须处理错误。

## environment

构造签名 `NAME(initial_state,seed)`，seed 是 0..1000000000 整数。
构造调用 reset 一次，全部输出验证成功才返回实例；不训练模型。
`env.observation` 读取最近成功观测；`env.reset()` 再调用一次 reset，返回初始观测。
reset handler 为同步二参 `fn reset(initial_state,seed)`，返回恰好 `{state,observation}`。
`env.step(action)` 调用同步二参 `fn step(state,action)`，返回恰好
`{state,observation,feedback,terminated,truncated}`。step 返回完整记录供算法读取。
action/observation/feedback 分别为声明宽度的有限 f32 可转换列表，两个结束标志严格 bool。
feedback 可表达监督目标、成本或奖励，不强制 reward 字段；不自动计算 GAE/bootstrap。
terminated 表示真正结束，truncated 表示截断，两者保留独立值；任一为 true 后 step 拒绝直到 reset。

state 为有限普通数据，首个成功 reset 确立字段/种类/固定列表形状，之后包括 reset 的 state 必须匹配。
复用记录数据上限：64 层、256 字段、1000000 节点与1000000 UTF-8 文本字节，不接受资源句柄。
普通记录字面量 `{field: expression, ...}` 按左到右求值，字段唯一，返回普通对象；不是块或可微张量。
完整 state、观测、反馈、标志验证后才提交实例；失败不改变 state/observation/done 或实例 RNG。
各实例拥有独立随机流：构造 seed 固定，reset 从原 seed 重放，step 从上次成功流位置推进。
handler 内 random/seed 作用于候选实例流，不改变调用者脚本 RNG；失败候选流丢弃。
外部 IO 的真实效果无法回滚，仍遵守现有显式宿主权限；声明不授予额外权限。

## 保存边界

首版不提供 agent/environment 整实例 checkpoint，也不保存可执行函数、环境句柄或外部资源。
显式 `save checkpoint actor.policy to PATH;` 保存模型和既有优化器/自定义状态；
加载模型后需按当前源码重新构造 agent/environment，构造环境仍执行其显式 reset。
`save json` 递归拒绝 agent/environment，包括列表/对象容器中的句柄；不会静默序列化为编号。
异步参数和结果也递归拒绝这些资源，handler/learner 首期均同步，不宣称多进程或多 GPU 环境。
