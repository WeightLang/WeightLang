# 原生自定义算子与分发

本地 WGT 0.5 支持纯 WGT 张量算子，执行和自动求导均在 Rust 内完成。这是当前源码能力，
尚未发布；不包含动态 Rust 插件、FFI、用户自定义 VJP、WGSL 自动求导或 GPU forward。

## 定义与调用

同文件中的 `fn name(arg, ...)` 可以在 `forward` 和 `loss` 中作为算子调用。
参数全部必填、按位置传递，无默认值，输入和返回值均为非空、定长、显式 shape 的 f32 张量；可用秩由所调用原语限定。
一个返回值；长度由实参及原生运算推导，每次调用单独检查。数值不隐式广播。
函数体只允许不可重复绑定的 `let`，最后必须有且只有一个 `return expression;`。
可以调用其他纯算子；不允许递归、赋值、循环、条件语句、I/O、随机采样或参数/工作内存读写。
模型参数必须在调用处显式读出并作为张量传入，算子不能捕获调用者局部变量或模型参数。
算子不创建或拥有可训练权重；是否共享参数由调用者传入的张量决定。

支持 `+ - *`、一元负号、有限数字向量字面量、`mix`、`.mean`，以及
`add/mul/relu/tanh/sigmoid/softmax/gate/route/slice/square/mean/sum/detach/exp/log/log_softmax/minimum/clip`。
这些原语的完整签名、形状、字面量参数及梯度与[语言参考](reference.zh-CN.md)和
[强化学习原语](reinforcement-learning.md)相同。特别是 slice 的起点/长度和 clip 的边界
仍须在算子函数体中写成字面量，不能使用张量参数代替；linear 的参数身份不作为普通张量传递。

纯 helper 也可组合多轴几何原语：`reshape(x,shape)`、`transpose(x)` / `x.T`、
`matmul(a,b)` / `a @ b`、`permute(x,axes)`、`batch_matmul(a,b)`、
`layer_norm(x,gamma,beta,axis,epsilon)` 及 `softmax(x,axis)`。
`permute` 与 `batch_matmul` 各有两个必填参数，`layer_norm` 有五个；
`softmax(x)` 保留单参数形式。轴、epsilon 和形状选项应直接写成合法常量，
张量值与 gamma/beta 则显式传入 helper。完整形状、轴及梯度约束见[张量几何](tensor-geometry.md)。
`conv` 的七个必填参数及 `max_pool2d(x,kernel,stride)` / `avg_pool2d(x,kernel,stride)`
也可用于纯 helper，窗口与步幅为常量列表；限制与直接调用一致，见[卷积与池化](convolution.md)。
这些函数的纯调用资格和 arity 来自统一[内部目录](prelude.md)；参数读写、工作内存和
`linear` 的参数身份访问继续在模型作用域处理。

调用按从左到右先求值各实参一次，再进入算子；每个调用有独立局部变量。
因此 `gate(p, custom(x))` 可在推理时跳过整个 custom 调用，
而 `custom(p, expensive(x))` 中的 expensive 会在进入 custom 前执行。
训练仍保留 gate/route 两支求完整梯度。`detach` 仍明确切断梯度。
调用链最多 32 层，每个 forward/loss 最多展开 1024 次算子调用；张量表达式展开受 100000 条指令/
寄存器检查上限约束，原有执行工作量与内存预算继续有效。

## 独立库和本地导入

库文件声明 `operators LibraryName`，只包含函数；没有模型、main、forward、loss、训练入口或依赖导入。
库名、函数和参数名使用 ASCII 标识符，不能以 `__` 开头；不能覆盖内置或张量原语名。
所有函数都是导出算子，库内调用保持库自身命名空间。每个库最多 1 MiB，每个消费者最多 64 个导入。

```wgt
wgt 0.5
operators Activations
# x 是 f32[N] 特征向量；逐元素 sigmoid 门保持形状，无独立参数。
fn swish(x) {
    let probability = sigmoid(x);
    return x * probability;
}
```

消费者顶层写 `import activation from "./ops.wgt"`，在 forward/loss 中调用
`activation.swish(x)`。导入独占一行，路径是 JSON 字符串，不带分号或尾注释。
本地路径必须以 `./` 开头，禁止 `..`、绝对路径和逃出入口目录的符号链接。
同一库可用不同别名导入。限定算子仅在张量 forward/loss 中可调用，不导入为普通脚本函数。

完整例子：[model.wgt](../examples/native_training/custom_operators/model.wgt)、
[ops.wgt](../examples/native_training/custom_operators/ops.wgt)。执行：

```sh
weightc check examples/native_training/custom_operators/ops.wgt
weightc check examples/native_training/custom_operators/model.wgt
wgt run examples/native_training/custom_operators/model.wgt
wgt run examples/native_training/custom_operators/model.wgt --compiled
```

检查库只证明纯函数结构、名称与调用图有效；实际形状和原语参数在消费者降低时检查。
检查/导入/编译不运行函数、不初始化模型、不训练。`wgt test` 同样支持带导入的模型测试。
`--check` 可读取本地导入，不下载包；包导入在不提供已解析依赖上下文时明确报 unresolved dependency。

## 发布与安装

使用已有 `wgt.toml` / `wgt.mod` 模块与精确版本机制。库源文件须由 Git 跟踪并提交；
`wgt release-assets` 自动发现 `operators NAME`，不要求预构建模型或权重目录：

```sh
# 在已配置模块路径的干净包仓库内执行。
wgt release-assets --version v0.5.0 --output dist
# 发布是显式动作，需要现有 GitHub/registry 凭据及匹配的远端 tag。
wgt publish --index dist/wgt.index.json --license MIT
```

包索引仍为 `weightlang.package-index` v1；算子制品格式为 `weightlang.operators.v1`，
每个算子库只包含 `manifest.json` 和 `source.wgt`，没有模型参数或训练断点。
安装验证源提交、版本、原始 Git 源、大小和 SHA256，`wgt.sum` 锁定两份文件。
同一个包可含多个算子库及原有模型制品，名称不能重复；发布含依赖的包仍不支持。
使用已有 `wgt add` / `fetch` / `sync` 或单文件依赖注释获取包，然后：

```wgt
import activation from "github.com/example/operators/Activations"
```

最后一段是库声明名，前面是依赖模块完整名称；不是任意文件路径或可执行 URL。
例如消费者的 `[dependencies]` 声明 `"github.com/example/operators" = "v0.5.0"`。
首次运行获取并验证依赖，后续可 `wgt --offline --locked run model.wgt`；缓存篡改或版本内容变化报错。
本文示例域名是占位符，本次没有发布远端包。

## 编译、梯度与持久化

算子降低成已有原生张量 IR，继承各原语的一阶 VJP、SGD/Adam 和事务语义。
保存推理模型或训练断点包含展开后的计算，无需安装算子包即可重载；断点保留优化器及 loss。
AOT 在编译时解析库，生成的程序运行时无需 WGT 源、包缓存、Rust 工具链。
只有显式执行 main 或测试才训练；加载模型不会执行库源文件。
这不是算子内核融合，亦不保证比手写等价表达式更快。
