# WeightLang 0.5 语言参考

当前源码只接受 **`wgt 0.5`**。这是破坏性语言更新，未发布；
`wgt 0.1/0.2/0.3`、`wgt script 0.1`、`wgt adaptive 0.1` 全部拒绝。
模型声明、训练函数和自更新前向函数属于同一个编译单元，不能通过选择命令改变语法版本。
工具包版本、模型 JSON 的 `version` 和训练断点版本是独立概念，不随语言版本替换。

当前实现为纯 Rust。原 Python 前端的完整图算子、block/import 等尚未全部迁入原生构造器；
`std/` 和其他模型示例保留为迁移材料，
不能因为文件头改为 0.5 就宣称可执行。不支持的声明显式报错，不回退 Python。

## 编译单元与入口

首个非空、非注释行必须恰好是 `wgt 0.5`，版本头只能出现一次。
`#` 到行尾为注释，函数字符串使用双引号和 JSON 转义。顶层模型声明一行一条；
函数使用 `fn name(args) { ... }`，语句以分号结束，控制流块无需分号。
名称使用 ASCII 字母或下划线开头、后接字母/数字/下划线；`__` 前缀供编译器使用。

一个文件可声明一个模型和多个函数。`fn main()` 无参数，执行程序时才调用。
模型名成为构造函数，调用才初始化会话；只解析/编译不访问训练数据、不训练、不保存。
只有模型声明的文件可检查，但执行或编译可执行文件时必须显式定义 main。

```sh
weightc check examples/native_training/supervised.wgt
wgt run examples/native_training/supervised.wgt
wgt run examples/native_training/supervised.wgt --compiled
weightc compile examples/native_training/supervised.wgt -o .cache/weightlang/regression
```

`wgt script` 是 `wgt run` 的命令别名，不是另一套语言。
解释和编译模式使用同一前端；编译模式生成 Rust 控制流和静态模型计划，运行时不解析 WGT。
`weightc check` 检查语法及受支持的模型声明，不证明所有函数调用、路径初始化或循环终止。

## 声明式模型

以下为当前原生构造支持的完整声明集；端口不含 batch 轴。

| 声明 | 参数、默认值与约束 |
| --- | --- |
| `model Name` | 一个模型名称，不得与函数或内置函数重名 |
| `input x: f32[N]` | 一个输入，固定 N 为 1..16384，先于 layer |
| `output y: f32[M]` | 一个输出，M 必须等于最后一层宽度 |
| `layer id: linear(in=N,out=M,bias=1)` | out 必填；in 默认前层宽度；bias 仅 0/1，默认 1；可用 `linear(M)` |
| `layer id: relu` / `tanh` / `sigmoid` / `identity` | 不改变向量宽度，无参数 |
| `init seed=N` | 必填、不可重复；N 为非负 u64；调用构造函数时进行均匀 Xavier 初始化，bias 为零 |

层按源码顺序执行，层名唯一。每个模型最多 400 万参数；参数名为 `层名.weight/bias`。
模型构造函数 `Name(optimizer)` 的 optimizer 必填，仅 `"sgd"` 或 `"adam"`。
调用两次产生独立参数；复用同一个句柄则共享参数及优化器状态。初始化不等于训练。

```wgt
wgt 0.5
model Regression
# 每个样本一个输入特征和一个回归目标；不含 batch。
input x: f32[1]
output y: f32[1]
layer hidden: linear(out=8)
layer activation: tanh
layer result: linear(out=1)
init seed=42

fn main() {
    # 构造只初始化；以下所有步共享同一组参数。
    let net = Regression("adam");
    for step in range(0, 100) {
        # fit 输入为 [样本, 特征]，MSE 对样本与输出元素取平均。
        fit(net, [[0], [1]], [[1], [3]], 0.03, "mse");
    }
    return predict(net, [1]);
}
```

## 普通函数、训练与数据

值包括有限数值、bool、字符串、null、列表、JSON 对象和不透明资源句柄。
`let name = expr;` 绑定变量，`name = expr;` 修改已有绑定。
支持函数调用、列表/对象索引、一元 `- !`、算术 `+ - * / %`、比较及短路 `&& ||`。
条件必须 bool，无隐式数值或张量真假转换；函数参数按位置传递，参数数目严格匹配。

控制流为 `if condition { ... } else { ... }`、`while condition { ... }`、
`for name in list { ... }`、`break;`、`continue;`、`return expr;`。
当前函数内共享变量环境，if/for 不创建词法局部作用域；重复 let 覆盖绑定。
无任意 Python、shell 或宿主回调。源码函数嵌套与求值均有预算限制。

训练、流式 JSONL、强化学习、保存与 benchmark 的完整内置签名、默认值及限制见
[原生训练](native-training.md#内置接口)。常用 `predict(model,input)`、
`fit(model,inputs,targets,rate,loss)` 没有隐式训练或保存。
`save_model` 保存模型，`save` 保存优化器断点，两者不能混用。

## 自更新前向函数

同一个模型可用 `fn forward(x)` 定义有状态前向，替代 layer/init 声明。
函数参数必须与 input 名称一致，input/output 仍使用相同的 `f32[N]` 端口语法。
参数显式声明为 `parameter w: f32[O,I] = [...]` 或 `frozen w: f32[O,I] = [...]`；
初值是行优先扁平有限 f32 数组，元素数必须与正整数形状匹配。
`frozen` 仅禁止本次调用内 write；不意味着该参数的数学梯度为零。

`fn forward` 复用普通函数的 let、赋值、return、if、while、break、continue 语法，
操作数为定长 f32 张量。暂不支持 forward 中的 for、辅助函数调用、数据 IO 或优化器调用。
当前前向表达式签名如下，除 linear 的 b 外没有默认参数：

| 表达式/语句 | 语义与约束 |
| --- | --- |
| `read(w)` | 读取参数当前值，结果为扁平向量；直接引用参数也会读取 |
| `write(w, value);` | 全量替换非冻结参数，元素数须相同 |
| `linear(x,w[,b])` | w=[O,I]、x=[I]、结果=[O]，可选 b=[O] |
| `add(a,b)` / `mul(a,b)` / `a+b` / `a*b` | 等长逐元素运算，无广播 |
| `relu(x)` / `tanh(x)` / `sigmoid(x)` | 等长逐元素运算 |
| `slice(x,start,length)` | start 非负整数、length 正整数，范围不得越界 |
| `positive(x)` | 仅作为条件；x 必须 f32[1]，严格大于零选真分支 |
| `let y = x;` | 复制张量值并保留梯度，不创建参数别名 |

局部张量的长度由表达式推导，重复赋值不能改变长度；输入不可覆盖。
同一模型内 forward 与 layer 不混用。模型通过 `Name("none")` 构造，
普通 `predict(handle,input)` 执行并提交成功写入；失败回滚，步数计入外层操作预算。
无需 `register`、标签、`jump` 或 `branch` 源码指令，这些仅存在于内部 IR。

完整可执行示例：[self_modifying/model.wgt](../examples/self_modifying/model.wgt)。
事务、原生一阶自动微分、显式状态及预算细节见[自更新运行时](self-modifying-runtime.md)。

## 实现与验证依据

- 编译单元和静态图计划：`crates/wgt-runtime/src/source.rs`。
- 共用函数语法：`crates/wgt-runtime/src/script/parse.rs`。
- 模型构造与普通函数运行：`crates/wgt-runtime/src/script.rs`。
- forward 降低、事务及梯度：`crates/wgt-runtime/src/adaptive/`。
- 原生代码生成：`crates/wgt-runtime/src/script/codegen.rs`。
- 回归：`tests/unified.rs`、`tests/adaptive*.rs`、`tests/script.rs`（runtime crate），
  `crates/wgt/tests/script_compiler.rs` 验证解释/AOT 的结果和操作数一致。

保留模型和实验在 `weights/`，不因升级版本头自动迁移或覆盖。详见[存储规范](storage.md)。
