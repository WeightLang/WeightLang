# SNN、局部 FF 与原生遗传算法

当前源码在 WGT 0.5 上增加通用数值能力、张量元组和状态模型组合；没有算法专用关键字。本页描述实现接口，尚未由本次开发发布到 PyPI/crates.io。模型负责内部计算和显式更新，Rust/Python 负责时间步、数据、正负样本和种群调度。

## 1. 语言与执行合同

- `ge(a,b)`：同形有限 f32 的逐元素大于等于，输出精确 0/1；相等触发，AD 明确拒绝。
- `div(a,b)` / `a/b`、`sqrt(x)`：同形运算，非有限结果拒绝，具有常规 VJP。
- `scale(x,s)`：s 可以是数值常量或运行时 `f32[1]`，显式缩放 x；没有通用广播。
- `(a,b)`、`let (a,b)=helper(...)`：2..16 项扁平张量元组，各项可以有不同 shape；仅编译期组合，不扩张服务 ABI。
- 操作互调、有状态子模型、静态实例数组：共享实例共用参数与 state，子调用的读写属于父操作的同一次事务。调用和实例递归拒绝。

typed `forward` 只读；更新使用普通具名函数，例如 `step`、`adapt`、`reset`。这些名称不触发隐式调度。旧普通 forward 的写入与训练合同保留。内部运行的 48 项 IR 同步覆盖 Rust 解释、原生 C 编译及 PyTorch；无 AD 的具名操作不会创建反向图。

纯库的完整签名、shape 和前置条件见[标准库](standard-library.md#脉冲与局部学习模块)。数学上的“局部梯度”与运行时“不调用 AD”不同；本页 FF 属于前者的显式解析实现。

## 2. 脉冲网络与局部可塑性

完整定义：[SNN model.wgt](../examples/native_training/non_backprop/snn/model.wgt)、[STDP model.wgt](../examples/native_training/non_backprop/stdp/model.wgt)。SNN 的两个 Cell 实例分别保存电位与增益；整个网络的一次 step 原子提交所有单元的更新。STDP 的 reset 清除迹、保留学习后的权重。

```python
import torch
import weightlang as wl

model = wl.load("examples/native_training/non_backprop/snn/model.wgt")
model.call("reset")
for current in [torch.tensor([0.6, 0.2]), torch.tensor([0.6, 1.0])]:
    spike = model.call("step", current)
    print(spike)
```

一次 step 是宿主约定的离散时间步；LIF 使用衰减、硬阈值、减阈值复位，每步每个单元最多一次脉冲。它没有隐含不应期、延迟或事件队列，不据此承诺事件驱动性能或节能。

`std/plasticity` 提供 Hebbian、单输出 Oja 和选定的 pair-STDP 公式。STDP 先衰减旧迹，计算权重变化，再加入当前脉冲；同一步零历史的前后脉冲不互相配对。需要权重边界时在模型中显式裁剪，再 `<-` 提交。

## 3. Forward-Forward

完整定义：[FF model.wgt](../examples/native_training/non_backprop/ff/model.wgt)。首期规则为 Dense/ReLU，goodness 是未归一化活动平方和；每一对正负样本从同一组旧参数计算更新，优化 pair mean 局部二元对数似然。向后层传递前使用 L2 长度归一化，零向量保持为零。[原始 FF 目标](https://arxiv.org/html/2212.13345v1)

```python
import torch
import weightlang as wl

model = wl.load("examples/native_training/non_backprop/ff/model.wgt")
report = wl.train_forward_forward(
    model,
    positive=[torch.tensor([1.0, 0.2]), torch.tensor([0.2, 1.0])],
    negative=[torch.tensor([-0.5, 0.1]), torch.tensor([0.1, -0.5])],
    rate=0.01,
    epochs=10,
)
print(report.updates)
```

Python `train_forward_forward(model,positive,negative,*,rate,epochs=1,adapt_operation="adapt") -> FFReport` 在首次写入前校验数据、dtype/device 和正负数量；每次调用 `adapt(positive,negative,rate)`，rate 为 `[1]`。`FFReport` 包含 pairs、epochs、updates。宿主不生成负样本，不创建 optimizer，也不调用 backward。

Rust `wgt::evolution::fit_forward_forward(&model,pairs,rate,"adapt") -> Result<usize>` 接受 `(Vec<f32>,Vec<f32>)` 的迭代器，返回更新次数。每对样本独立提交；之后出错不撤销前面成功的学习步。宿主可用不同的 adapt 入口分别训练不同层。

该解析实现不构建自动微分图，但数学上仍使用局部梯度。它不提供任意网络的自动 FF 转换，也不能直接把 ReLU 换成硬脉冲后沿用同一导数。

## 4. 候选参数接口

`ParameterSchema.candidate_updatable` 标识是否允许外部候选更新。普通未冻结参数和合法 learn 参数可更新；frozen/probability 排除。bool 值必须精确为 0/1，int 必须为域内整数；提交层不隐式投影。

- Rust：`Model::update_parameters(&[ParameterCandidate])`，每个候选项为 `(原始名称,shape,Vec<f32>)`。
- Python：`OperationsModule.update_parameters({原始名称: Tensor})`，先验证整个集合的名称、shape、dtype/device、有限性、合法域和资格，再在同一实例锁内提交。
- Python `parameter_schema` 提供 schema 副本，`parameter_values()` 提供独立参数值；PyTorch 的 `parameters()` 仍然是标准参数迭代器。

这些接口替换一个候选的多个参数，不是多候选 batch。普通 Rust forward 的外部更新清除旧 optimizer 状态；typed operations 没有 optimizer。候选更新不会隐式 reset 神经状态。

## 5. 官方 GA：Python

```python
import torch
import weightlang as wl

model = wl.load("examples/native_training/non_backprop/snn/model.wgt")
search = wl.GA(model, config=wl.GAConfig(population_size=16, elite_count=2), seed=7)

# Toy objective for demonstrating the lifecycle, not a performance benchmark.
def fitness(candidate, episode_seed):
    output = candidate.call("step", torch.tensor([0.7, 1.1]))
    return float(((output - torch.tensor([1.0, 1.0])) ** 2).sum())

for _ in range(10):
    scores = search.evaluate(
        model, fitness, operation="step", reset_operation="reset",
        episode_seeds=(11, 23),
    )
    search.tell(scores)

search.apply_best(model)
search.save("weights/ga-example/search.json")
```

`GA` 是 `GeneticAlgorithm` 的短别名。构造器接受 typed 模型或 `GenomeSchema`，配置为 `GAConfig`；使用 `ask()` / `tell(scores)` 可把候选生成与本地、分批或远程评估分开。scores 以候选 ID 绑定有限适应度，目标可选 maximize/minimize，平局以 ID 稳定排序。库提供 tournament 选择、精英保留、均匀交叉和域感知变异。

`evaluate_isolated(model,candidates,fitness,*,schema=None,operation=None,reset_operation=None,episode_seeds=(0,))` 为每个候选的每个 episode 创建独立副本。含 state 的模型必须声明无输入、void、无参数写入且覆盖全部 state 的 reset。fitness 得到只允许调用指定只读参数操作的 Evaluation 门面；可推进 state，默认不能执行局部参数学习。整数 seed 传给 fitness，由宿主据此重置环境和独立 RNG，不修改全局 torch/Python RNG。

`state_dict()` / `from_state_dict()` / `load_state_dict()` 支持宿主 checkpoint；`save(path,overwrite=False)` / `load(path)` 使用 JSON，默认拒绝覆盖。保存的是搜索状态，不是环境或活跃神经状态的通用快照。

## 6. 官方 GA：Rust 原生路径

```rust,ignore
use wgt::{Client, InstanceOptions, ModelDefinition};
use wgt::evolution::{
    evaluate_isolated, EvaluationOptions, GAConfig, GeneticAlgorithm, GenomeSchema,
};

let definition = ModelDefinition::from_file(
    "examples/native_training/non_backprop/snn/model.wgt"
)?;
let client = Client::connect()?;
let model = client.initialize(&definition, InstanceOptions::default())?;
let schema = GenomeSchema::from_model(&model)?;
let mut search = GeneticAlgorithm::new(schema.clone(), GAConfig::default(), 7)?;
let options = EvaluationOptions {
    operation: Some("step".into()),
    reset_operation: Some("reset".into()),
    episode_seeds: vec![11, 23],
    ..Default::default()
};
for _ in 0..10 {
    let scores = evaluate_isolated(
        &client, &definition, &schema, &search.ask(), &options,
        |candidate, _seed| {
            let output = candidate.call(&[vec![0.7, 1.1]])?.unwrap();
            Ok(output.iter().map(|x| f64::from(*x - 1.).powi(2)).sum::<f64>())
        },
    )?;
    search.tell(&scores)?;
}
search.commit_best(&model)?;
```

Rust 引擎不依赖 Python。`Candidate { id,values }`、`Score { id,fitness }` 和排序后的 `GenomeSchema` 使用固定参数顺序；共享参数只编码一次。`GenomeSchema::from_definition(definition,initial_values)` 也可用于不连接服务的搜索初始化。两种语言默认目标均为最小化，上例返回正的平方误差。

`GAConfig` 字段为 population_size、elite_count、tournament_size、mutation_rate、mutation_scale、crossover_rate、integer_step、objective。`checkpoint_json()` / `from_checkpoint_json()` 与 `save_checkpoint()` / `load_checkpoint()` 保存配置、完整模型定义/schema、种群、代数、最优结果和 SplitMix64 RNG 状态。Rust/Python 共用 `weightlang.ga.v1` 格式；跨平台浮点 fitness 的一致性仍取决于模型后端与环境。

每个候选/episode 初始化新服务实例，评估失败即丢弃该实例；仅在成功获得完整分数后 `tell` 推进搜索。Rust 的 reset_operation=None 使用每次新实例的声明初态；Python 对含 state 的模型要求显式覆盖全部 state 的 reset，两者均不会沿用上个 episode 的状态。`commit_best` 只提交参数，不复制评估中的膜电位或迹。若固定的 frozen/probability 参数来自已训练权重，传 `EvaluationOptions.initial_weights`，避免新实例错误地回到声明初值；未提供时拒绝与声明不一致的固定基线。

## 7. 范围与验收

本版 GA 固定模型定义、参数/state schema 和函数签名。bool/int 可以控制现有上界内的分支、开关或深度；新增节点存储或改变层宽的拓扑进化需要新的编译定义，未包含 NEAT 实现。

行为检查包括：硬阈值及拒绝 AD；逐时间步 SNN 标量参考；FF 独立局部目标有限差分；STDP 时序；子调用返回/事务；Rust/PyTorch 输出、参数和 state 差分；GA 域、隔离、错误原子性及 checkpoint 续跑。具体执行结果记录在 PR 中，解析通过不替代这些行为检查。没有训练效果、速度或节能的普遍保证。

搜索采用逐候选实例隔离保证正确性。之后是否需要快照、并行 worker 或批量执行，应根据实际剖析数据决定；当前 Rust 模型句柄不能直接跨线程共享。
