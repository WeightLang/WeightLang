# weightfs v1：运行中的权重文件系统

weightfs 是 Linux FUSE 内存文件系统，直接连接 Rust 进程里的模型，不生成磁盘权重副本。
这是当前源码的本地实现，未发布；不改变 `wgt 0.5` 语法或模型制品格式。

## 自动挂载与生命周期

`wgt run`、`wgt run --compiled`、编译后的独立程序、`wgt-native script` 在模型构造或恢复后
默认尝试挂载；`wgt-native MODEL.json ...` 和 `wgt-native adaptive ...` 在开始接收推理请求前尝试挂载。
同一脚本中的多个模型各有独立目录，句柄副本共享同一个目录。
`weightc legacy-build` 的显式执行阶段同样挂载，返回模型进行导出前卸载。
纯检查、只编译和设备枚举不挂载、不初始化模型、不执行训练。

规范路径：

```text
${XDG_RUNTIME_DIR}/weightlang/weightfs/<pid>-<session>/
├── manifest.json
├── 000000.json
└── 000001.json
```

未设置 `XDG_RUNTIME_DIR` 时使用 `/run/user/<euid>`。运行时目录必须已存在，路径必须绝对，
目录所有者必须为当前有效用户，且不能给予组/其他用户任何权限；拒绝符号链接。
weightlang/weightfs 子目录按 0700 创建。会话目录使用独占创建，不覆盖已有目录或挂载。
实际路径输出到 **stderr**，格式为 `weightfs: /absolute/path`，不会混入 JSON stdout。

需要 `/dev/fuse`、可用的 `fusermount3`（或系统支持的 fusermount）和用户挂载权限；不需要 sudo。
非 Linux 平台目前没有挂载实现。

| 环境变量 | 行为 |
| --- | --- |
| `WGT_WEIGHTFS=auto`（默认） | 尝试挂载；失败打印原因并继续模型运行 |
| `WGT_WEIGHTFS=on` | 必须成功挂载，否则执行报错 |
| `WGT_WEIGHTFS=off` | 不挂载 |

其他值报错。设置 `WGT_HOST_POLICY` 时 auto/off 不暴露文件系统，on 报错，避免绕过宿主 I/O 策略。
训练与推理核心的 Rust `Session` / `Trainer` / `ExecutionSession` 构造器本身不执行 OS 挂载；
嵌入者用 `weightfs::Mounted::auto(model) -> Result<Mounted<T>>` 显式采用同一生命周期，
`path() -> Option<&Path>` 查询路径，`lock() -> Result<MutexGuard<T>>` 执行模型操作，
`into_inner() -> Result<T>` 卸载后取回模型。`Weights` trait 定义参数枚举、revision、读取及替换接口。

会话正常结束、执行错误或 Rust 栈展开时卸载，删除本次创建的空会话目录；不删除共享父目录。
进程被 SIGKILL、断电或强制退出不保证执行析构，可能留下失效挂载或空目录。
确认对应进程已结束后可用 `fusermount3 -u <准确会话路径>`，随后 `rmdir <准确会话路径>`；
不要批量删除其他会话目录。退出不自动保存权重。

## 文件格式与编辑协议

`manifest.json` 只读，包含 `format="weightlang.weightfs.v1"`、PID、当前模型 revision 和参数表。
每个参数记录 `file`、原始 `name`、`shape`、`dtype="f32"`、`writable`。
参数文件用从零开始的六位十进制索引命名；更多位数不截断。顺序为模型原始参数顺序，
同一会话中固定。原始参数名仅出现在 manifest，不作为路径解析，避免 `/`、`..` 等名称逃逸。

参数文件内容是 UTF-8 JSON 的**行优先扁平数组**，例如 `[0.5,-1.25,3.0,0.0]`。
形状通过 manifest 查询。所有元素必须可表示为有限 f32，元素数必须完全匹配；
不支持修改形状、dtype、参数名称、模型结构或优化器状态。
冻结参数、file/add/量化存储保持原运行时的只读规则；可写 view 更新其实际 resident owner。

```sh
# 启动一个长期读取 stdin 的推理会话；用已经保存的原生推理模型：
WGT_WEIGHTFS=on wgt-native weights/native_training/model.json 1048576 --device cpu

# 另一个终端，复制 stderr 打印的准确路径：
MOUNT=/run/user/1000/weightlang/weightfs/12345-0
cat "$MOUNT/manifest.json"
cat "$MOUNT/000000.json"
# 仅当 manifest 表明这个参数有一个元素且可写时：
printf '[2.0]\n' > "$MOUNT/000000.json"
```

读取在 `open` 时获取单个参数的稳定快照；重新打开可看到最新值。不同文件的读取不构成
跨参数快照。编辑是单文件事务：`write` 仅更新该描述符的内存缓冲，`fsync` 或 close 的 flush
校验并提交全部元素。使用 `fsync` 可明确检查提交结果；成功表示已更新模型内存，不表示落盘。
重复 flush 不重复提交。非法 JSON、非有限数值和形状不符返回 `EINVAL`，不部分更新。

推荐以 `O_WRONLY|O_TRUNC` 打开并写入完整数组。只写打开同样从空暂存缓冲开始；
`O_RDWR` 从完整快照开始，可在描述符上截短后重写。不支持 append、路径单独 truncate、
新建/删除文件、rename、原子替换式保存、mmap 或通用文件系统元数据修改。
编辑器必须使用直接写入模式；使用临时文件再 rename 的默认保存方式会失败。
不要忽略编辑器、close 或 fsync 的错误。

## 一致性、训练与预算

FUSE 和模型执行持有同一把锁。单次预测、一个训练 step、保存或单文件提交互斥；
编辑不会在前向或梯度计算中途生效，提交成功后下一次执行使用新值。
训练参数更新或任意其他参数提交导致模型 revision 改变时，旧编辑返回 `EAGAIN`。
重新打开最新文件再编辑；不得静默覆盖训练的新值。不同参数同时编辑也可能发生冲突。
这不是多文件原子事务，也没有自动暂停训练；快速持续训练时需要调用方协调编辑时机。

手工编辑保留 Adam 矩和优化器步数，可能改变后续训练轨迹；不自动重置优化器。
GPU 推理沿用已有 revision 驱动的参数上传，下一次执行同步新值；此次实际挂载回归覆盖 CPU，
GPU 实机编辑验收仍需对应硬件测试。自更新 forward 同样使用模型锁；已有自动微分 tape
仍是创建时的快照，外部编辑不改写旧 tape。

不复制整个模型作为后台镜像。打开文件时分配该参数 JSON 快照，提交时分配 f32 校验数组。
每个会话最多 64 个打开文件，暂存缓冲累计上限 64 MiB；单参数 JSON 上限
`min(元素数 × 32 + 2, 64 MiB)`，参数元素数超过 `64 MiB / 32` 时暂不支持文件访问。
容量不足返回 `ENOSPC`/`EFBIG`，不会把大权重暂存到 `/tmp`。
这些快照、解析副本、FUSE 元数据不计入模型原有 tensor budget，64 MiB 不是进程内存上限。
JSON 浏览与文件编辑会分配内存；底层未封装的 CPU 前向仍保留原有零堆分配契约。

挂载目录是明确的易失权重视图，不属于 `weights/` 制品，也不是下载/训练数据缓存。
持久化必须显式使用 `save model` 或 `save checkpoint` 语句，保存到新的 `weights/<experiment>/` 文件；
仍拒绝覆盖已有结果。详见[存储规范](storage.md)和[原生训练](native-training.md#保存与续训)。

## 验证

```sh
WGT_WEIGHTFS=on cargo test -p weightlang-runtime --no-default-features --test weightfs -- --ignored
WGT_WEIGHTFS=off cargo test -p weightlang-runtime --no-default-features --test training --test script --test unified
```

真实挂载测试显式标为 ignored，避免在没有 FUSE 的构建容器中伪报通过；上述第一条明确运行它们。
覆盖挂载目录浏览、编辑后的预测、训练可见性、冲突/非法写回滚、冻结权重、自更新模型、
CLI 等待 stdin 时的外部编辑和正常退出卸载。
