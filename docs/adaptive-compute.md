# 条件计算、自适应深度与多尺度组合

以下是当前源码的 `wgt 0.3` 能力，尚未发布。Python 负责数据、损失、优化器与应用调度；
WGT 定义参数、张量计算和有界控制流。示例源文件只初始化参数，不包含训练权重。
多输入、多结果 block 与显式多状态 scan 见 [composition.md](composition.md)。

## 按样本选择实际执行的分支

`route(branch0, branch1, ...) from selector, x`，至少两个分支，无隐式默认分支。
branch 可以是一个算子、管线或普通单输入单输出 block；不接受带 return 记录的 block。
selector 为单样本 `i64[]`，值为从零开始的分支编号；越界在执行任何分支前报错。
两个分支时也接受 `bool[]`：false 选 branch0，true 选 branch1。
x 为 f32，所有分支须保持 x 的完整形状、dtype 和已知轴语义。

每个分支只收到选中它的样本；没有样本时不调用该分支。结果回填到原 batch 顺序。
空 batch 不调用分支；未选中分支不消耗随机数，也不产生该次计算的参数梯度。
参数在不同分支间默认独立，在同一分支的不同样本间共享；明确的 parameter/share 仍有效。
硬整数/布尔选择不可微，模型输出对被选中分支和输入可微。
这是一选一的路由，不是通用 top-k 加权 MoE，也不自动学习 selector。
`where` 仍保留两支先计算再选择的原语义。

完整可运行例子：[conditional_route.wgt](../examples/conditional_route.wgt)。
层参数路径为 `selected.branch0...`、`selected.branch1...`，保存、导出和反编译均保留。

## 压缩活动样本的概率退出循环

完整签名 `recur(N, body, until=NAME, compact=0)`。
N 是 1…1024 的静态整数；until、compact 都可省略。compact 只允许 0/1，显式使用时
必须同时指定 until。until 的 f32[1] 概率层、形状保持、每轮更新后采样及梯度规则，
与 [中文参考](reference.zh-CN.md) 中原有概率退出相同。

compact=0：保留原行为，部分样本停止时仍计算整批，已停止样本的结果冻结。
compact=1：每轮把仍活动的样本送进 body，将完成结果回填；批内全部停止后直接返回。
不改变样本顺序，不持久保存状态，不共享不同样本的状态。压缩会改变实际随机数消费顺序，
因此相同随机 seed 不保证与 compact=0 抽到完全相同的路径，也不保证固定加速倍数。
空 batch 时压缩循环不执行 body。单个非空样本至少执行一轮。

```wgt
wgt 0.3
model compact_exit
input x: f32[4]
output y: f32[4]
block think():
  # [4] -> [4]，各轮复用同一组状态更新参数。
  layer next: linear(4) -> tanh from input
  # [4] -> [1]，每样本一个停止概率；这里只初始化，没有训练。
  layer stop: linear(1) -> sigmoid from next
  layer result: identity from next
# 只让仍活动的样本继续计算；随机退出判断不可微。
layer reasoning: recur(8, think(), until=stop, compact=1)
init seed=42
```

## 可训练的有界停止分布

完整签名 `ponder(N, body, until=NAME)`；三个参数均必填，无默认停止门。
N 是 1…1024 的静态整数。body 和 until 的契约与概率 recur 相同：单个 f32 状态，
每步保持完整 shape/dtype/轴，直接子层 NAME 输出 f32[1] 的有限 [0,1] 概率。
循环体最后一层输出新状态；不接受记录返回，不能用 compact 选项。

ponder 在 train/eval 下都执行完整 N 步，不随机退出，保留可微的每步概率。
设第 t 步停止概率为 p_t：t<N 时，首次停止概率 q_t=p_t∏_{j<t}(1-p_j)；
最后 q_N=∏_{j<N}(1-p_j)，吸收全部剩余质量，因此 q 的和为 1。
最后一轮 p_N 仍作为诊断输出，但不影响 q_N；N=1 时 q_1=1，停止门不通过 q 学习。
这是有限步停止分布构件，不自动加入 PonderNet 的 KL 先验或其他训练算法。

若输入单样本形状为 S，结果是：

| 字段 | 单样本类型/形状 | 含义 |
| --- | --- | --- |
| output | f32[S] | 最后一轮状态；不是按概率加权的平均状态 |
| states | f32[N, *S] | 每轮新状态，第一条模型轴为思考轮次 |
| probabilities | f32[N,1] | 每轮原始停止概率 p_t |
| halt_weights | f32[N,1] | 首次停止分布 q_t |
| expected_steps | f32[] | Σ t·q_t，轮次从 1 开始 |

表中的 S/*S 是签名记法，不是可直接写入 WGT 的语法。
可用 `output weights: f32[8,1] from reasoning.halt_weights` 暴露给 Python。
每次调用重置状态，各轮共享参数；保存完整计算图的训练内存随 N 增长。

训练通常使用“每步损失的期望”，而非“平均状态的损失”。Python 示例：

```python
result = model(x)
per_step_loss = (result["states"] - target[:, None, :]).square().mean(-1, keepdim=True)
loss = (result["halt_weights"] * per_step_loss).sum(1).mean()
loss = loss + 0.01 * result["expected_steps"].mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

[model.wgt](../examples/adaptive_compute/model.wgt) 和 [train.py](../examples/adaptive_compute/train.py)
构成完整例子：在内存中用合成目标训练 30 步，断言停止门参数改变，不写模型制品。
这只证明训练链路，不证明有效的任务策略或泛化质量。
推理可另写相同 cell/层名的 `recur(...,until=stop,compact=1)`，显式迁移同名参数；
不隐式修改 ponder 的 eval 行为。

## 多尺度空间算子

输入均为非空 f32[C,H,W]，batch 是额外前导轴；不接受 i64/bool 或空模型轴。
配置必须是静态整数，样本输出不含 batch。

| 完整签名 | 输出和约束 |
| --- | --- |
| resize2d(height, width, mode=0, align_corners=0) | f32[C,height,width]；height/width>0。mode=0 最近邻，mode=1 双线性；align_corners 为 0/1，最近邻只允许 0。无 antialias、scale_factor 或自动尺寸对齐 |
| pad2d(left, right, top, bottom, mode=0) | f32[C,H+top+bottom,W+left+right]；四个边距必填且非负。mode=0 常量零、1 反射、2 复制；反射每侧边距严格小于对应输入尺寸。不支持负边距裁剪或自定义填充值 |
| avgpool2d(kernel, stride, padding=0, count_include_pad=1) | 每个空间轴变为 floor((size+2·padding−kernel)/stride)+1；kernel/stride>0，0≤padding≤floor(kernel/2)，输出非空；count_include_pad 为 0/1。方形窗口、floor 输出，无 ceil_mode/divisor_override |

以上签名中的必填参数实际调用须使用 `key=value`，例如
`resize2d(height=8,width=8,mode=1)`。这些算子无可训练参数，值的梯度保留；
插值和池化结果对照 Torch 默认数值语义。
[多尺度例子](../examples/multiscale.wgt) 组合卷积、降采样、双线性上采样和通道跳连。

## 验收范围

Python 测试覆盖被选分支的真实调用、batch 顺序、梯度、逐步活动数量、停止分布、
合成优化器更新、非法契约、空间算子的非连续输入、保存重载、独立 Python 导出和反编译。
Rust 使用 Python 生成的结构记录验证输入输出、权重布局、结果字段和篡改拒绝。
Rust 校验与图查看不等于 Rust 张量执行。没有发布或真实模型训练质量声明。
