# 迁移路线与验证

以下命令以当前本地源码为依据。每次执行前核对 `weightc --help` 和实际实现；
优先读仓库 `docs/decompile.md`、`docs/python.md`、`docs/debugging.md`，避免沿用过期能力。
示例路径是独立任务的占位路径，先检查存在性；不覆盖现有文件。

## 1. 原生 WeightLang 模型

```sh
weightc verify weights/original
weightc decompile weights/original --references -o models/migrated.wgt
weightc check models/migrated.wgt --shapes
weightc build models/migrated.wgt --no-store --no-train -o weights/migration
```

`build -o ROOT` 写入 `ROOT/<源码中的模型名>/`，不是直接写入 ROOT。
默认 `decompile` 只恢复结构、重新初始化；保留权重必须显式 `--references`，生成固定哈希的导入。
保留原制品作为导入来源。改架构复用读 `docs/architecture-reuse.md`，显式选择基线与映射；
重叠复制、裁剪和扩宽属于架构变更，不等于无损迁移。

## 2. Python 架构

```sh
weightc decompile model.py -o models/migrated.wgt
weightc check models/migrated.wgt --shapes
weightc build models/migrated.wgt --no-store --no-train -o weights/scaffold
```

当前静态 Python 转换只支持一个字面量 nn.Sequential：带 bias 的 Linear 和
默认参数的 ReLU/Tanh/Sigmoid/GELU/SiLU/Identity。它只读 AST，不执行输入代码，
不恢复权重或初始化随机状态。自定义类、forward、嵌套序列与非默认选项不能靠此命令自动迁移。

对其他模型，读真实 forward 后手工编写 WGT。重点核对：

- linear 最后一维语义，卷积通道/核布局，reshape/转置的轴序。
- attention 的缩放、mask、QKV 排列和投影 bias；归一化 epsilon 与维度。
- dropout 在 eval/train 的差别，残差连接位置和激活的精确形式。
- repeat 独立参数与 recur/scan 共享参数，状态初值和跨调用重置。
- 权重绑定、缓存、分词器、token ID、位置编码、图像归一化等外部语义。

不要凭“都有 Transformer/GRU”就替换为标准库 block。先证明完整公式一致。

## 3. 检查点与外部格式

```sh
weightc decompile model.safetensors -o models/architecture.wgt
weightc decompile model.safetensors --references -o models/inventory.wgt
```

外部格式默认输出 `wgt architecture 1`，是结构描述，不可 build/run。
引用模式输出 `wgt tensors 1/2`，是张量清单，也不可执行。即使 check 通过，
也未证明完整载荷哈希、forward、权重绑定或运行结果。头部哈希不是载荷哈希。

本地 reader 可检查 safetensors、受限 PyTorch state_dict、GGUF、ONNX、NumPy 等；
支持读取不等于支持执行、反量化或导出。hf:// 路线只支持 safetensors，固定实际 commit。
没有结构源码时先找可用配置与实现，找不到就交付清单和具体缺口，不猜测模型。

读取 PyTorch 权重用受限 `weights_only=True`，不以 unrestricted pickle 作为失败回退；
不要为迁移任意执行陌生仓库或开启远程自定义代码。量化/适配器转换先核对实现与误差要求。

## 4. 显式参数映射

当前通用路径是：构建可执行结构 → 加载原参数 → 显式映射 → 使用 artifact.save 保存新模型。
这不是外部 checkpoint 的一键原生导入。转换脚本应与迁移结果一起保留。

```python
from weightlang import load_artifact

artifact = load_artifact("weights/scaffold/model", warn_untrained=False)
expected = artifact.module.state_dict()
# source_state 由已验证的加载器读取；mapping 为目标名 -> 原名称。
# 两者由具体模型决定，不能用排序、zip 或形状相同来猜对应关系。
assert set(mapping) == set(expected)
mapped = {}
for target, origin in mapping.items():
    value = source_state[origin]
    assert value.shape == expected[target].shape
    assert value.dtype == expected[target].dtype
    mapped[target] = value
artifact.module.load_state_dict(mapped, strict=True)
saved = artifact.save("weights/migrated")
```

这是映射模板，不是能脱离 source_state/mapping 独立运行的脚本。
额外的源张量也要逐个解释：如优化器状态或已确认不用的分支，不能静默丢弃。
需要转置、拼接或 dtype 转换时，在赋值前显式实现并记录；量化整数不能直接 cast 当作反量化。
共享权重要验证绑定语义，而非仅比较初始数值。保存接口不接受架构、forward 或冻结参数的偷偷修改。

`artifact.save` 的路径是新的完整模型目录，不能位于基线目录内，也不能已存在。
纯拷贝外部参数时不要伪造 training 记录；现有保存接口会把变更标为 edited，
不是自动认证的 trained/imported。另存来源、哈希、映射及真实训练元数据作为迁移证据。

## 5. 输出对齐与调试

用实际原模型在 eval 模式生成固定输入/输出，保持相同预处理、batch、dtype 和设备。
比较对象必须包含保存后重新加载的目标模型，不能只测尚未保存的内存对象。
常规 f32 小模型可从 atol=1e-6、rtol=1e-5 开始；这只是起点，
低精度、不同内核或长序列需用任务要求解释容差，不能为通过测试任意放宽。
报告最大绝对误差、容差、样本范围及是否通过；同时检查形状、NaN/Inf 和任务相关输出。

```sh
weightc verify weights/migrated
weightc trace models/migrated.wgt --weights weights/migrated --sample sample.json --json
# 仅当任务要验证训练且源码有有效 train/data 时运行：
weightc debug models/migrated.wgt --weights weights/migrated --steps 3 --json
```

`trace/debug` 不自动加载之前 build 的权重，必须明确 --weights 或源码中的固定导入。
权重模式严格匹配结构和输入输出契约，报告权重哈希；不执行架构迁移。
目前仅 CPU；debug 仅内置 backprop，使用新优化器与固定小批次，只在内存更新。
用 --watch 层路径及 --time 循环索引定位最早出现偏差的输出；逐参数报告说明梯度和更新幅度，
但不能代替原模型对比。诊断运行不会保存修复后的模型，保存需走明确的新制品路径。
