# 原生数学标准库（WGT 0.5，本地未发布）

原生标量函数直接调用，无需 import。解释器与 AOT 都经过同一个 Rust 实现。
全部参数必填，无默认参数；输入输出均为有限 f64 标量，不接受张量或列表，
不创建模型、权重或训练状态，也不提供张量自动微分。非有限结果报错。

| 分类 | 完整签名 | 用途与约束 |
| --- | --- | --- |
| math | `rsqrt(x)` | `1/sqrt(x)`，x > 0；精确平方根再取倒数，未使用低精度快速近似 |
| math | `fma(x,y,z)` | 融合乘加 x*y+z，只舍入一次 |
| math | `hypot(x,y)` | 欧氏长度，避免直接平方造成中间溢出/下溢 |
| math | `expm1(x)` | exp(x)-1，保留接近零的小量精度 |
| math | `log1p(x)` | log(1+x)，x > -1，保留接近零的小量精度 |
| activation | `sigmoid(x)` | logistic，按正负分支避免指数溢出 |
| activation | `softplus(x)` | max(x,0)+log1p(exp(-abs(x)))，避免大正数指数溢出 |
| activation | `silu(x)` | x*sigmoid(x)，使用稳定 sigmoid |

实现调用 Rust 原生数值接口。FMA 可由目标平台硬件或软件实现；是否更快取决于
目标 CPU、编译选项和调用规模。rsqrt 当前没有 SIMD 近似指令、GPU 内核或误差容限模式；
不能宣称硬件加速比。稳定算法的价值也包括避免溢出和消减误差。
参考 [Rust f64 文档](https://doc.rust-lang.org/std/primitive.f64.html)。

完整可执行示例：

- [基础数学](../std/math/scalar.wgt)：`wgt run std/math/scalar.wgt`
- [稳定激活](../std/activation/stable.wgt)：`wgt run std/activation/stable.wgt`

追加 `--compiled` 可走原生编译路径。根目录原有 std/*.wgt 保留兼容路径，
其中 block 图声明仍是迁移材料，不能用这些新函数的可执行性推断旧图已支持。
新内容按 math/、activation/ 分类；这里的目录是源码与示例组织，不引入模块导入语法。

## 向量算法

以下函数也属于 WGT 0.5 本地原生实现，参数全部必填，无默认值。
向量为非空一维有限 f64 列表 `[N]`，没有隐含 batch 轴，不接受嵌套列表。
返回列表时保持输入长度。全部函数无权重、无自动微分，不替代 `fit` 的训练损失。
每次调用除普通解释器费用外额外计费 N 个操作（AOT 相同），时间 O(N)、工作空间 O(N)。
浮点归约不保证精确求和；无法表示的结果报错。此批未接入 GPU/SIMD 或性能基准。

| 分类 | 完整签名 | 返回与算法/约束 |
| --- | --- | --- |
| linalg | `dot(x,y)` | 标量内积；长度相同，顺序 FMA 累加 |
| linalg | `norm(x)` | 标量 L2 长度；逐步 hypot 避免直接平方溢出 |
| reduction | `mean(x)` | 标量均值；先除 N 再求和以减少求和溢出 |
| probability | `softmax(x)` | `[N]` 概率；减最大值后指数归一化 |
| probability | `log_softmax(x)` | `[N]` 对数概率；直接在对数域计算，避免概率下溢后取 log |
| probability | `logsumexp(x)` | 标量 log(sum(exp(x)))；减最大值稳定计算 |
| normalization | `l2_normalize(x,eps)` | `[N]`，x/max(norm(x),eps)；eps 有限且 >0；缩放后求范数，零向量返回零 |
| loss | `mse(x,y)` | 标量均方误差；等长，均值取全部 N 个元素 |
| loss | `mae(x,y)` | 标量平均绝对误差；等长 |
| loss | `cross_entropy(logits,target)` | 单样本标量负对数似然；target 为 0..N-1 的整数，不接受概率标签 |

损失函数的逐元素差和内积的累加仍可能溢出并报错；不会静默截断。
`log_softmax` 的极端类别差若超出 f64 表示范围也报错。
`l2_normalize` 的 eps 位于 max 内，与旧 `tensor_math.wgt` 中平方根内加 eps 的定义不同。

完整例子（均可用 `wgt run 路径` 或追加 `--compiled` 执行）：

- [线性代数](../std/linalg/vector.wgt)
- [归约](../std/reduction/mean.wgt)
- [概率](../std/probability/softmax.wgt)
- [归一化](../std/normalization/l2.wgt)
- [损失](../std/loss/basic.wgt)

## 统计、排序与变换

以下接口继续使用 WGT 0.5 的非空一维有限 f64 列表契约，参数全部必填、无默认值。
不改变输入列表，无权重、训练或自动微分状态。索引从 0 开始。

| 分类 | 完整签名 | 返回、约束及算法 |
| --- | --- | --- |
| statistics | `min(x)` / `max(x)` | 标量最小/最大值 |
| statistics | `variance(x,ddof)` | 标量方差，分母 N-ddof；ddof 为整数且 0<=ddof<N；0 是总体方差、1 是样本方差 |
| statistics | `stddev(x,ddof)` | 同一方差定义的标准差；先缩放和中心化，避免直接平方溢出 |
| statistics | `median(x)` | 排序后中位数；偶数长度取中间两数平均 |
| statistics | `quantile(x,q)` | q 为 [0,1] 数值；对升序序列位置 q*(N-1) 做线性插值，单元素返回该元素 |
| ordering | `sort(x)` | 升序 `[N]`，稳定排序；不修改原列表 |
| ordering | `argsort(x)` | 升序对应的原始索引 `[N]`，相等元素保持原顺序 |
| ordering | `argmin(x)` / `argmax(x)` | 极值首次出现的索引，数值相等的正负零也按首次处理 |
| ordering | `cumsum(x)` | 包含当前元素的顺序前缀和 `[N]`；任何中间和溢出均报错 |
| linalg | `clip(x,lower,upper)` | 逐元素裁剪到闭区间，同形返回；两边界有限且 lower<=upper |
| linalg | `cosine_similarity(x,y)` | 标量余弦相似度，等长且两个向量均非零；分别缩放后求内积与范数，最终裁剪到 [-1,1] |

统计量使用浮点近似运算；大量几乎相等的大数仍可能产生舍入误差，不保证精确统计。
标准差可能可表示但方差不可表示；后者报错，不返回无穷。分位数插值分别处理
同号和异号端点，避免不必要的和/差溢出。未增加 SIMD/GPU 或性能保证。

排序、索引排序、中位数和分位数耗时 O(N log N)，额外预算为
`N*(1+ceil(log2(N)))`，N=1 时为 1；其余函数 O(N)、额外预算 N。
所有函数工作空间 O(N)，在分配算法工作空间之前检查预算；与原生编译执行保持一致。
预算是确定性的计费单位，不是 CPU 指令数或实际运行时间。

完整可运行示例：

- [统计](../std/statistics/descriptive.wgt)
- [排序与检索](../std/ordering/ranking.wgt)
- [向量变换](../std/linalg/transforms.wgt)

## 分段激活、稳健损失与信号处理

WGT 0.5 本地实现，所有参数必填、无默认值。标量输入为有限 f64，向量为非空
一维有限 f64 列表，无隐含 batch。函数不修改输入、训练模型或产生自动微分图。

| 分类 | 完整签名 | 定义、形状与约束 |
| --- | --- | --- |
| activation | `relu(x)` | 标量 max(x,0) |
| activation | `leaky_relu(x,alpha)` | x>=0 时 x，否则 alpha*x；alpha 有限且 >=0 |
| activation | `elu(x,alpha)` | x>=0 时 x，否则 alpha*expm1(x)；alpha 有限且 >=0 |
| activation | `hard_sigmoid(x)` | 标量 clamp(x/6+0.5,0,1)，端点 -3 与 3 |
| activation | `hard_swish(x)` | 标量 x*hard_sigmoid(x)，分段计算避免中间溢出 |
| loss | `bce_logits(logits,targets)` | 等长 `[N]` → 标量均值二元交叉熵；targets 每项在 [0,1]，支持软标签；输入为 logits 而非概率 |
| loss | `huber(x,y,delta)` | 等长 `[N]` → 标量均值；delta 有限且 >0；绝对残差 d<=delta 时 0.5*d²，否则 delta*(d-0.5*delta) |
| loss | `log_cosh(x,y)` | 等长 `[N]` → 标量均值 log(cosh(x-y))，小残差用 log1p，大残差在对数域计算 |
| signal | `diff(x)` | `[N]` → `[N-1]`，输出 x[i+1]-x[i]；N>=2 |
| signal | `ema(x,alpha)` | `[N]` → `[N]`；y[0]=x[0]，y[t]=(1-alpha)*y[t-1]+alpha*x[t]；0<alpha<=1 |
| signal | `moving_mean(x,window)` | `[N]` → `[N-window+1]`，仅有效窗口、不填充；window 是 1..N 的整数 |

BCE 根据 logits 符号分支，计算稳定的非负损失，避免直接 log(sigmoid(x)) 下溢。
Huber/log_cosh 的残差或逐元素损失无法表示时显式报错；最终归约溢出也报错。
这些损失用于脚本数值计算与监控，不是可传给 `fit` 的新增训练目标名称。
EMA 状态仅在本次调用内部流动，不跨调用持久化；alpha=1 原样返回输入值。
滑窗均值直接逐窗口求平均，避免滚动加减累计漂移，但仍有普通浮点舍入误差。

标量激活 O(1)，无额外向量计费；损失、diff、EMA 时间 O(N)，额外预算 N。
滑窗均值时间及额外预算为 `(N-window+1)*window`，并非 O(N) 优化内核。
向量函数工作空间 O(N)，在工作空间分配前扣除预算。没有新增 GPU/SIMD 承诺。

可直接运行或追加 `--compiled` 的完整例子：

- [分段激活](../std/activation/piecewise.wgt)
- [稳健损失](../std/loss/robust.wgt)
- [信号平滑](../std/signal/smoothing.wgt)

## 数据构造与索引

WGT 0.5 本地实现，所有参数必填、无默认值。以下数据操作接受一维有限 f64 列表，
**允许空列表**，区别于前面的非空向量归约。所有输出都是新列表，不修改原输入，
无隐含 batch、模型参数、训练或自动微分状态。单次输出上限 1,000,000 项。

| 完整签名 | 定义、形状及约束 |
| --- | --- |
| `zeros(count)` | 长度 count 的全零列表，count 为 0..1,000,000 的整数 |
| `ones(count)` | 同上，填充 1 |
| `full(count,value)` | 同上，填充有限 f64 value；即使 count=0 也验证 value |
| `linspace(start,end,count)` | 等间距列表，含两端点；start/end 有限，count 为 1..1,000,000 整数；count=1 仅返回 start，支持降序 |
| `one_hot(index,classes)` | `[classes]`，index 位置为 1，其余为 0；classes 是 1..1,000,000 整数，index 是 0..classes-1 整数 |
| `concat(x,y)` | 拼接 `[N]` 与 `[M]` 为 `[N+M]`，总长度不超过上限 |
| `vector_slice(x,start,end)` | 左闭右开 `[start,end)`，要求整数 0<=start<=end<=N；输出 `[end-start]`，不支持步长或负索引 |
| `gather(x,indices)` | 按整数索引列表顺序取值，输出长度等于索引数；允许重复，全部索引须在 0..N-1，空 indices 合法 |
| `compress(x,mask)` | mask 是等长 bool 列表，仅保留 true 位置且保持顺序；不将 0/1 隐式转为 bool；全 false 返回空列表 |
| `reverse(x)` | 返回倒序的新 `[N]` 列表 |

`linspace` 使用同号/异号分支插值，避免直接相减造成不必要的溢出；内部值仍有
浮点舍入，两个端点直接使用传入值。数据操作验证整个输入，即使切片/筛选未选中某元素。

额外预算为所有列表参数长度之和，加输出长度；`compress` 使用 N 作为输出长度上界。
先检查输出上限与预算，再扫描数值及分配输出。时间 O(输入长度+输出长度)，
空间 O(输出长度)；compress 输出上界计费不意味着一定分配 N 项。失败不退还已扣预算。
这些是 CPU 原生列表操作，不提供张量视图、零拷贝或 GPU/SIMD 加速保证。

完整示例：

- [构造与编码](../std/data/construction.wgt)
- [索引与筛选](../std/data/indexing.wgt)

使用 `wgt run std/data/construction.wgt`，追加 `--compiled` 可执行编译版本。

## 概率分布分析

WGT 0.5 本地原生接口，参数全部必填、无默认值，输入为非空有限 f64 一维列表。
沿单条 `[类别]` 轴计算，没有隐含 batch。所有对数使用自然对数，熵/散度单位为 nat。
不产生训练梯度，也不是新的 `fit` 损失名称。

| 完整签名 | 用途、定义与输出 |
| --- | --- |
| `normalize_probabilities(weights)` | 非负权重、至少一项>0；先除最大值再归一化，返回同形概率列表，避免直接求和溢出 |
| `entropy(p)` | 标量 Shannon 熵 `-sum(p*log(p))`，零概率项贡献为 0 |
| `kl_divergence(p,q)` | 等长分布，标量 `sum(p*(log(p)-log(q)))`；方向是目标 p 相对参考 q，不对称 |
| `js_divergence(p,q)` | 等长分布，标量 `(KL(p,m)+KL(q,m))/2`，m=(p+q)/2；对称，精确值在 [0,log(2)] |
| `cross_entropy_probs(p,q)` | 等长分布，标量 `-sum(p*log(q))`；p 是目标分布，q 是预测概率；不同于接受 logits 和类别编号的 `cross_entropy` |
| `temperature_softmax(logits,temperature)` | 正且有限的温度；同形输出 softmax(logits/temperature)，温度 1 对应普通 softmax |

entropy、KL、JS 和概率交叉熵要求每项在 [0,1]，总和与 1 的绝对误差<=1e-9。
通过检查后再除以总和修正舍入误差；不会把任意权重自动当分布。需要归一化权重时
显式使用 `normalize_probabilities`。非常小的概率仍可能按 f64 规则下溢为零。

KL/交叉熵中 p=0 的项忽略；p>0 而 q=0 表示无限结果，按有限数值契约报错。
JS 支持不相交分布，在对数域计算混合概率以避免小概率平均时下溢。
KL/JS 的微小负舍入误差截为零；不保证精确实数恒等式。
温度 softmax 先减最大值；当该差溢出时改为先除温度再相减，保留可表示的概率。

这些接口时间、工作空间均 O(N)（双分布必须等长）；额外预算为所有列表参数长度之和
再加第一个列表长度，即单列表 2N、双列表 3N。在工作空间分配前检查并扣除预算，
失败不退还已扣预算。没有新增 GPU/SIMD 保证。

完整可运行例子：

- [分布分析](../std/probability/distributions.wgt)
- [温度 softmax](../std/probability/temperature.wgt)

可用 `wgt run 文件路径` 或追加 `--compiled` 执行。
