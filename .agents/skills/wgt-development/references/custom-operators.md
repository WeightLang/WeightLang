# 自定义算子

以 `docs/custom-operators.md`、`crates/wgt-language/src/script/operators.rs` 和 `crates/wgt-language/src/adaptive/source.rs` 为准。
WGT 0.5 的 `fn` 在 forward/loss 中可作为纯张量算子展开，只支持不可重复 let 与最终 return。
输入/返回为定长 f32 张量；参数显式传入，无捕获、递归、I/O 或状态写入。纯 helper 条件允许比较、`positive` 和布尔字面量。
按 language/prelude.json 的 operator_arity 校验纯张量调用；permute 与 batch_matmul 各为两参，layer_norm 为五参，softmax 为一或两参。多轴形状、轴与字面量约束遵循 docs/tensor-geometry.md，参数/工作内存访问保持模型作用域限制。max_pool2d 与 avg_pool2d 各为三参，通道优先 [C,H,W]，kernel/stride 为整数常量列表；完整定义见 docs/convolution.md。
独立库 `operators NAME` 没有 main 或模型；`import alias from "./ops.wgt"` 与
`import alias from "MODULE/LibraryName"` 仅导入张量算子，调用为 `alias.function(args)`。
实参先按顺序求值一次，调用内局部名称隔离；gate/route 的外部短路与训练梯度保持原语语义。
纯 helper（含显式导入）允许 `x.probabilities`：rank-1 f32 分数/logits 复用稳定 softmax 及梯度，probabilities 验证后原值返回且梯度为恒等。不按数值猜类型，非向量/非有限值拒绝。返回值的概率类型保留，调用结果再接后缀不会重复 softmax；`linear` 仍只在模型 forward 读取原始参数，可把其结果传给 helper。内部 `__probabilities` 不允许源码直接调用。

模型 forward 还提供卷积原子 `conv(x,kernel,bias,stride,padding,dilation,groups)`，恰好七个必填参数，无默认值。`x` 是通道优先 `[Cin,S1,...,Sd]`，kernel 为 `[Cout,Cin/groups,K1,...,Kd]`，bias 为 `[Cout]`；空间 rank 仅支持 1..3，输出 shape 为 `[Cout,O1,...,Od]`，其中 `Oa=floor((Sa+2*padding[a]-(dilation[a]*(Ka-1)+1))/stride[a])+1`。stride、padding、dilation 都是长度与空间 rank 相同的整数常量列表：stride/dilation 每项 1..16384，padding 每项 0..16384，表示对称零填充；groups 是 1..16384 的整数常量，须整除 Cin、Cout，kernel 的输入通道维须等于 Cin/groups。无 batch 轴，kernel 不翻转（互相关）；参数及模型端口仍是行优先扁平 f32，调用前须显式 `reshape`。不支持动态选项、非零 padding 模式、内置激活或 CNN block；池化使用独立原语。

core 卷积 3/3 已通过，覆盖 1D/2D/3D 数值与梯度等检查；真实 wgtime service 的卷积 3 项也已通过，覆盖训练、权重恢复续训与释放。这些是局部验证，不代表完整 service 验收。完整定义与有效 WGT 示例见 `docs/convolution.md`。

库制品 `weightlang.operators.v1` 只含 manifest.json/source.wgt；release-assets 自动构建，
fetch/verify 使用现有版本、源提交、大小、哈希与 wgt.sum。不训练、不产生权重或优化器状态。
最终模型 IR/AOT 已展开，重载或执行不需要原算子库；不表示支持 Rust 动态插件或 WGSL 梯度。

模型纯张量库精确逻辑导入名为 `std/activation` 和 `std/loss`；内部 `/operators` 路径明确拒绝，不依赖宿主目录查找。旧 `std/prelude.wgt` 原生用户声明源已移除；内置目录的过渡状态见 `docs/reference.zh-CN.md`。前者导出 swish(x)、swiglu(gate,value)、squared_relu(x)，非空同形 f32[N] 到 f32[N]；后者导出 squared_error(prediction,target)、half_squared_error(prediction,target)、categorical_logits(logits,target)，同形 f32[N] 到 f32[1]。全部参数必填，无默认值/广播/隐含batch；前两项取特征均值，categorical_logits 沿类别求和且不动态验证 target 分布。库不创建参数或状态，CPU 梯度/AOT 继承展开原语，未知 std 路径报错。标准库不带 main 演示或旧图占位；完整可检查模型示例、数学定义和限制见 docs/standard-library.md。

后缀转换的真实导入库入口为 `crates/wgtime-core/tests/fixtures/probability_helper/model.wgt`，库文件同目录 `ops.wgt`。
`crates/wgtime-core/tests/probability_suffix.rs` 验证本地及导入 helper 的概率类型保持、CNN native 前向、有限差分和训练。
包版本锁/篡改及无源码无工具链 AOT 行为属于独立验证范围，本次后缀测试不表示这些路径已重新验收。
