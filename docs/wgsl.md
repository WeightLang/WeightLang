# 原生 WGSL 与 GPU 训练

这是当前 WGT 0.5 的本地实验实现，尚未发布。纯 Rust 使用 wgpu 执行 WGSL，
支持构建与驱动提供的 Vulkan/Metal/DX12；无需浏览器或 Python。
默认仍为 CPU，不自动回退，不代表所有 forward 程序或 NPU 均可执行。

```sh
cargo build -p weightlang-wgt --features gpu
./target/debug/wgt run examples/native_training/gpu/model.wgt
./target/debug/wgt run examples/native_training/wgsl/model.wgt
./target/debug/wgt run examples/native_training/wgsl/model.wgt --compiled
```

解释和 AOT 使用相同内置函数，AOT 继承编译器的 gpu feature。
检查/编译只生成程序，不枚举 GPU、编译着色器或执行训练；这些动作发生在运行期。

## 设备与训练

所有参数必填；整数字面量/参数遵循普通函数 [0,1000000000] 范围。

| 签名 | 契约 |
| --- | --- |
| `devices()` | 返回设备对象列表，字段 id/name/backend/kind/memory_model；CPU 在首项，之后为硬件 GPU。CPU 构建仅返回 CPU。枚举不证明模型可执行；设备编号不代表永久身份。 |
| `training_device(model,device,memory_bytes,max_batch)` | 顺序模型（声明式或 mlp）显式选择训练设备，返回与 training_status 相同的对象。device 为 cpu 或 devices 返回的精确 GPU ID，不接受 auto/gpu/npu。后两项为正整数；GPU 用于数值内存预算和最大批次，CPU 接受但不使用这两个限制。失败保持原设备、权重和优化器状态。 |
| `training_status(model)` | 返回 device、steps、inference_device、state。steps 是已提交优化步数；当前 inference_device 恒为 cpu、state 恒为 host_committed。只接受顺序模型。 |

GPU 支持 f32 的 linear/relu/tanh/sigmoid/identity，fit 的 mse/cross_entropy，以及 reinforce，
优化器为 SGD/Adam。批次为 [B,I]、回归目标 [B,O]；交叉熵类别索引和优势规则见
[原生训练](native-training.md#实验性-rust-gpu-训练-api)。不支持 fn forward、自定义 loss、train 裁剪语句、混合精度和 NPU 训练。

选择设备只创建设备和内核，不执行训练；实际模型工作区与输入形状在调用 fit/reinforce 时检查。
前向、损失、反向和优化器合并为一次 GPU 提交；梯度在这一事务内驻留 GPU。
同尺寸 storage/status/readback 缓冲跨调用复用，不同尺寸先丢弃旧缓冲。
每步仍上传当前权重/矩，检查回读成功后才提交 CPU 副本；不是跨步常驻张量实现。
失败不会提交权重、矩或步数；重试会清零 tape 与算术状态。
predict/sample 和默认 benchmark 仍用 CPU，显式 benchmark 设备模式独立选择。
训练设备设置不写入模型或断点；restore 后默认 CPU，可显式重选。

目前宿主设备授权只覆盖推理队列。设置 WGT_HOST_POLICY 或 compute_budget 后，
拒绝直接 GPU 训练和自定义 WGSL（包括在设预算之前创建的句柄），避免绕过设备额度。
仍可切回 CPU。设备内存预算不包含驱动、编译器、命令对象和全部宿主堆，不能用作进程硬隔离。

完整训练例子：[gpu/model.wgt](../examples/native_training/gpu/model.wgt)。

## 自定义 WGSL

| 签名 | 契约 |
| --- | --- |
| `wgsl_kernel(source,entry,device,memory_bytes)` | source 是 1..1048576 字节 WGSL 字符串，entry 为非空计算入口名，device 为精确硬件 GPU ID，memory_bytes 为正整数。编译并返回不透明内核句柄，不调度。每个 WGT runtime 最多 64 个内核，退出时释放。 |
| `wgsl_run(kernel,input,output_count,workgroups)` | input 为非空有限 f32 列表，output_count 正整数；workgroups 为三个正整数 [x,y,z]，每项不超过设备限制。同步调度并返回长度 output_count 的有限 f32 列表；输出初始为零。 |

固定缓冲 ABI（张量 shape 为一维，布局为连续 f32）：

```wgsl
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&x) && id.x < arrayLength(&y) {
        y[id.x] = x[id.x] * 2.0;
    }
}
```

没有其他绑定、uniform、纹理或隐式 shape 参数；内核作者必须遵循 f32 ABI，
wgpu 验证着色器和绑定合法性，不证明算法正确或每个输出都已写入。
工作组数量由调用者指定，与 @workgroup_size 相乘得到 invocation 数量；不自动推断。
同一内核可复用 pipeline 多次执行；运行期源字符串写在 WGT 中时也会进入 AOT 程序，不依赖外部 shader 文件。

预算检查使用 `12 * (input.len + output_count)` 字节覆盖托管数值缓冲与复制，
不包括 WGT Value 对象、着色器源、编译器与驱动开销；还检查设备 buffer/binding 大小。
非法 shader、入口、绑定、设备、尺寸、预算或非有限输入/回读输出报错。
自定义内核没有内置训练内核的溢出预检查；WGSL 不保证可移植的 NaN/Inf 传播，
回读有限值检查不能证明中间算术没有溢出。
同步等待上限为 30 秒（两次等待分别设限），不是硬件任务取消或着色器终止保证。
仅用于可信代码；不承诺任意内核终止。

这是普通函数中的显式计算接口，没有模型参数注册、forward 节点或反向传播规则；
不得把 wgsl_run 当作可微算子。NPU 使用独立的宿主图后端注册入口，不能执行本接口。
完整例子：[wgsl/model.wgt](../examples/native_training/wgsl/model.wgt)。

## 验证

```sh
WGT_WEIGHTFS=off cargo test -p weightlang-runtime --no-default-features --test script_gpu
WGT_WEIGHTFS=off WGT_REQUIRE_GPU=1 cargo test -p weightlang-runtime --features gpu --test gpu_training -- --include-ignored --test-threads=1
WGT_WEIGHTFS=off cargo test -p weightlang-runtime --features gpu --test script_gpu -- --include-ignored --test-threads=1 --nocapture
WGT_WEIGHTFS=off cargo test -p weightlang-wgt --features gpu --test script_compiler compiled_gpu_and_wgsl -- --ignored --nocapture
```

硬件测试不能以解析成功或自动跳过代替；不同 GPU 和后端需分别验证。

2026-11-04 本机验证：Vulkan / NVIDIA GeForce RTX 4060 Laptop GPU，GPU 训练套件
5 项、脚本/WGSL 套件 4 项通过；CPU 构建的相关 5 个套件共 31 项通过。
解释/AOT 的两个真实例子及 Adam 断点续训通过；AOT 在删除源码且 PATH 无工具链后运行。
这些结果不代表 Metal、DX12、其他 GPU 或 NPU 已验证，也不包含性能提升结论。

WGSL 工作组数量受普通脚本整数范围限制，每项为 1..1000000000 且不超过所选设备限制；超范围值报错，不截断。
