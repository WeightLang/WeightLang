# 原生测试、评估与计时（本地 WGT 0.5）

`wgt test [--filter NAME] [--max-operations N]` 从项目 `tests/*.wgt` 发现普通无参
`fn test_*()`。文件名、函数名按字典序执行，不递归子目录，不执行 `main()`。
约定复用现有函数语法、语句行号与求值器，不添加注解或特殊 test 关键字。
测试文件不是模型入口；模型和训练源码仍命名为 `model.wgt`。
纯测试编译单元可没有 main；`wgt run` / AOT 编译仍要求显式 main。

每个测试创建新运行时，独立的模型、数据流、随机数、指标和操作预算（默认 10000000）。
辅助函数不以 test_ 开头；带参数的 test_ 函数计为失败。构造器不是测试函数。
失败不中止其他函数/文件；`assert(bool,message)` 失败输出文件:行号与消息，嵌套调用保留错误链。
显示 PASS/FAIL、metric 和通过/失败/总数，任意失败非零退出；无匹配测试也非零退出。
filter 是大小写敏感的 `文件路径::函数名` 子串；JSON fixture 按文件路径匹配。
语法错误无法可靠发现函数名，整文件计一次失败，即使 filter 不匹配也不会隐藏它。
运行时隔离不回滚磁盘、网络等显式外部副作用，也不捕获宿主进程崩溃。
测试 CLI 暂不支持 WGT_HOST_POLICY，检测到时明确拒绝；有宿主策略时使用受监督的普通脚本。

`wgt test --compiled` 当前明确拒绝，尚无 AOT 测试发现入口。
普通 `wgt run --compiled` 的 assert、数据、评估和 benchmark 与解释器共用内置实现；
计时值本来会变化，不要求逐位一致。CLI 的旧设备/分配选项只影响旧 JSON revision 路径，
WGT 用 benchmark 的 mode 明确选择设备，predict 当前仍为 CPU。

## 评估

测试内使用 `model(path,optimizer)` 加载保存的原生模型，不执行训练，不创建模型制品；
可用 `dependency(module)` 解析已锁定依赖路径。相对路径基于测试文件所在目录。
`jsonl(path)` / `batch(reader,count)` / `close(reader)` 流式读取数据；每批使用普通循环
调用 `predict(model,input)`，用 `mse`、`cross_entropy`、`argmax` 和计数器归约指标，
最后 assert 阈值。必须显式检查空数据集。列表的外层样本轴与内层特征轴不能混淆。
模型状态、类别编码、阈值和数据切分由测试明确指定，没有隐藏评估回调或 Python。
完整可运行项目见 [native_evaluation](../examples/native_evaluation/README.md)。

## JSON 共存

`tests/*.json` 与 WGT 函数进入同一汇总。新增原生 fixture：

```json
{"model":"weights/demo/model.json","input":[0.25],"expected":[1.5],"atol":0.01}
```

model 为项目相对路径，input/expected 是数值向量；atol 默认 1e-6，须非负有限。
它只加载原生模型并在 CPU 预测，不运行源码/训练。旧 `source/input/expected/atol`
fixture 的 revision 路径保留；该路径依赖已移除的 Python CLI，目前明确失败，
不会偷偷用随机初始化或重新训练替代当前 revision。迁移时显式指定保存的原生 model。

## benchmark

`benchmark(model,workload,warmup,repeats,name[,mode])` 返回指标对象并登记到 metric。
mode 默认 `"cpu"`；warmup 为 0..10000，repeats 为 1..100000 整数，其他参数必填。
仅顺序模型，不支持 forward 程序模型。计时单位 us，mean 算术平均；p50/p95 为排序后
`floor((N-1)*q)` 项（q=0.5/0.95），无插值。排除脚本 dispatch、数据解析、模型/设备初始化。

| mode | workload | 测量范围 |
| --- | --- | --- |
| `"cpu"` | 单样本数值向量 | CPU 前向；warmup 前额外验证一次 |
| `"gpu"` 或精确适配器 ID（如 `"vulkan:0"`） | 单样本数值向量 | 第一台硬件 GPU / 指定 GPU 的同步 host-to-host 前向 |
| `"train:cpu"` | `[inputs,targets,rate,"mse"]` | CPU 前向、批均值 MSE、反向、优化器更新 |

GPU 需要 gpu 构建特性；无设备、内存/算子不支持即失败，不回退 CPU。
GPU 私有队列在计时前完成一次不计时的同步前向，即使 warmup=0；每次 run 等待本次提交
并映射回读，因此计时包括输入/必要权重上传、编码提交、设备执行、等待和输出回读，
不是 GPU 内核时间，也不是跨队列全局设备屏障。设备张量预算固定 256 MiB。
宿主策略/compute_budget 下拒绝直接 GPU benchmark，沿用直接 GPU 调用权限边界。

训练计时只支持 CPU MSE，rate 为正有限 f32，inputs=[样本,特征]，targets=[样本,输出]。
SGD/Adam 沿用模型优化器；warmup 与 repeats 均真实更新同一模型，不重置或保存。
GPU trainer 不允许作为 train:cpu 计时；GPU 训练步计时尚未接入。
没有新增内置名称或图 primitive，因此 primitive_specs.json 和编辑器词表保持不变。
