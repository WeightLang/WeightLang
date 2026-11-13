# Python 的 WGT 模型对象

以下是当前源码的接口变更，不表示已有 0.5.4 发布包包含它。`weightlang.load`、`loads`、`from_ir`、`load_dynamic`、`loads_dynamic` 默认构造 `wgt.Model`，通过 Rust/wgtime 执行。不会因为服务缺失或遇到不支持的模型而隐式转用 PyTorch。公共 `compile/compile_source` 默认输出 `weightlang.wgt.v1` 定义，CLI 为 `wgt compile --target wgt`（默认目标）；不生成或执行 PyTorch。源码编译仍只有 Rust 前端，构造实例才初始化参数并连接服务。

```python
import weightlang as wgt

with wgt.load("model.wgt", optimizer="adam", memory_bytes=64 * 1024 * 1024) as model:
    prediction = model([1.0, 2.0])
    loss = model.train_step([[1.0, 2.0]], [[0.0]], rate=0.001)
```

原生安装不再依赖 PyTorch。需要编译器和同版本的 wgtime 服务；`WGT_COMPILER` 指定编译器，`WGTIME_ENDPOINT` 或 `endpoint=` 指定 Unix socket，默认 `/run/wgtime/service.sock`。当前 IPC 客户端要求 Linux，校验 peer UID，单帧最多 16 MiB。`timeout=30` 设置 socket 超时，不是服务取消/回滚保证；不完整交换会关闭连接，绝不自动重放修改。`with` / `close()` 释放连接持有的实例；关闭可重复调用。跨 fork 的连接不能复用。

`Model.from_source(source, *, compiler=None, dimensions=None, input_shapes=None, timeout=30, **options)` 与 `Model.from_file(path, ...)` 也构造原生实例。`options` 为 `endpoint=None`、`memory_bytes=67108864`、`optimizer=None`、`device="cpu"`、`dtype=None`；原生仅支持 CPU/f32。`optimizer` 可以是 `None`、`"sgd"`、`"adam"`。原生参数与状态属于服务，不注册成 `torch.nn.Parameter`。

- 普通单向量 forward 接受一个有限 f32 列表，返回列表；tuple forward 接受声明顺序的多个逻辑嵌套列表，也可按名字传入，返回 tuple，保持每个输出的逻辑形状。无隐含 batch、dtype 转换或广播。
- `call(name, *inputs)` 调用既有 typed operation，保持无 AD 合同。
- `parameters()` 返回不可变 `Parameter(name, shape, data)` 快照；`set_parameters(parameters)` 显式更新，沿用服务的完整校验和原子提交规则。替换参数会重置训练状态；之后显式 `set_optimizer("sgd" | "adam")` 再训练，配置失败不修改已有优化器。
- `train_step(inputs, targets, *, rate, loss="mse", clip=None, straight_through=False)` 接受打包向量批次，返回更新前 loss；由服务更新参数。含 bool/int learn 参数时须明确选择 straight_through=True，否则拒绝。它不接入宿主 PyTorch autograd。初始化未配置优化器时须先 `set_optimizer`。
- `train_recurrent(inputs, targets, *, input_offset, output_offset, state_width, supervised_offset, supervised_width, rate, clip=None, straight_through=False)` 使用原生显式 hidden BPTT，监督 MSE；`None` target 不监督该步。限制与预算见 [动态模型](dynamic-models.md)。
- `load_dynamic` / `loads_dynamic` 必填 `input_shapes`，一个服务句柄保留权重与优化器。输入形状变化时由共享编译器重新绑定；检查首个定义的源码指纹，并由服务验证参数 schema 和候选瞬时内存。`specialize(input_shapes)` 可显式切换形状。没有 Python WGT 解析器。

## 显式 PyTorch 后端和原有接口

安装 `weightlang[torch]`，使用 `wgt.load(..., backend="torch", device="cpu")` 等入口得到 `TorchModel`，它也是 `wgt.Model`。`forward`、`parameters()`、`state_dict()` 等委托给真实 PyTorch 模块；`to_torch()` 显式取得原始 `nn.Module`，可以使用标准优化器与 autograd。`train/eval/to` 返回同一个 WGT 对象。没有跨后端隐式迁移参数/优化器的操作。

需要 `torch.fx.GraphModule`、`nn.Module` 类型身份或原有研究工具时，使用 `from weightlang import torch_api as wgt`，这个明确的互操作命名空间保留之前的原始 PyTorch 返回值。原有 PyTorch 检查点仍由此接口或 `backend="torch"` 加载。当前统一入口 `save` 只接受显式 TorchModel，原生训练检查点的统一 Python 保存/恢复尚未接入；参数快照不能冒充完整检查点。

`weightlang.modules.bind_models(..., backend="torch")` 是可信 Torch 专用方言的兼容入口，不是原生模型加载器；其生成类同时继承 `wgt.Model` 与 `nn.Module`，保留参数键和训练行为。旧调用默认 torch 仅为该既有专用接口的兼容合同。新通用加载入口默认始终是 native。

## 自定义宿主行为

可以继承 `Model`，覆盖宿主的 `forward`，再调用 `super().forward(...)`。这适用于输入预处理、验证和输出后处理；模型张量计算仍写在 WGT 中，不能借此把任意 Python 计算插入原生 IR。显式 TorchModel 的 `.module` 提供原始模块，自定义 Torch 宿主仍使用普通 PyTorch 规则。
