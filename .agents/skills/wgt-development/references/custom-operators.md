# 自定义算子

以 `docs/custom-operators.md`、`script/operators.rs`、`adaptive/source.rs` 为准。
WGT 0.5 的 `fn` 在 forward/loss 中可作为纯张量算子展开，只支持不可重复 let 与最终 return。
输入/返回为定长 f32 向量；参数显式传入，无捕获、递归、I/O 或状态写入。
独立库 `operators NAME` 没有 main 或模型；`import alias from "./ops.wgt"` 与
`import alias from "MODULE/LibraryName"` 仅导入张量算子，调用为 `alias.function(args)`。
实参先按顺序求值一次，调用内局部名称隔离；gate/route 的外部短路与训练梯度保持原语语义。

库制品 `weightlang.operators.v1` 只含 manifest.json/source.wgt；release-assets 自动构建，
fetch/verify 使用现有版本、源提交、大小、哈希与 wgt.sum。不训练、不产生权重或优化器状态。
最终模型 IR/AOT 已展开，重载或执行不需要原算子库；不表示支持 Rust 动态插件或 WGSL 梯度。

标准 WGT 库精确逻辑导入名为 `std/activation` 和 `std/loss`；内部 `/operators` 路径明确拒绝，不依赖宿主目录查找。前者导出 swish(x)、swiglu(gate,value)、squared_relu(x)，非空同形 f32[N] 到 f32[N]；后者导出 squared_error(prediction,target)、half_squared_error(prediction,target)、categorical_logits(logits,target)，同形 f32[N] 到 f32[1]。全部参数必填，无默认值/广播/隐含batch；前两项取特征均值，categorical_logits 沿类别求和且不动态验证 target 分布。库不创建参数或状态，CPU 梯度/AOT 继承展开原语，未知 std 路径报错。标准库不带 main 演示或旧图占位；完整可检查模型示例、数学定义和限制见 docs/standard-library.md。

真实自定义库入口 `examples/native_training/custom_operators/model.wgt`，库文件同目录 `ops.wgt`。
验证 runtime/tests/custom_operators.rs 与 wgt/tests/operator_package.rs，包含梯度、隔离、
非法捕获/递归/形状、版本锁/篡改、保存/断点和无源码无工具链 AOT 行为。
