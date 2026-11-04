# 原生文件与断言语法

以下为本地尚未发布的 WGT 0.5 普通函数语法，解释执行与 AOT 共用实现；forward/loss 不允许文件副作用。检查和编译只解析，不读取文件、不初始化或训练模型。

```wgt
wgt 0.5
fn main() {
    # 文本为 UTF-8，路径相对入口所在目录。父目录必须已经存在。
    save text "你好" to "message.txt";
    let message = load text from "message.txt";
    assert message == "你好", "文本往返失败";
    save json [1,true,null,message] to "data.json";
    let data = load json from "data.json";
    # overwrite 是显式覆盖许可；省略则目标存在时报错。
    save json data to "data.json" overwrite;
    # 这里只初始化一入一出的模型，尚未训练。
    let net = mlp([1,1], "identity", 4, "adam");
    save model net to "model.json";
    save checkpoint net to "checkpoint.json";
    let inference = load model from "model.json" using "adam";
    let resumed = load checkpoint from "checkpoint.json";
    assert net([1]) == inference([1]);
    assert net([1]) == resumed([1]);
    return data;
}
```

保存语句完整形式为 `save KIND VALUE to PATH [overwrite];`。KIND 只能是 `text`、`json`、`model`、`weights` 或 `checkpoint`，属于上下文语法标记。VALUE 先求值一次，随后 PATH 求值一次。PATH 必须是字符串；绝对路径直接使用，相对路径按脚本根目录解析，AOT 沿用其 `--root` 契约。不会自动创建父目录。

读取表达式为 `load text from PATH`、`load json from PATH`、`load model from PATH using OPTIMIZER` 或 `load checkpoint from PATH`。PATH 和 OPTIMIZER 各求值一次。模型必填新优化器字符串：顺序模型允许 `sgd`/`adam`，forward 模型另允许 `none`；不会恢复原训练状态。checkpoint 恢复记录中的优化器、步数及状态，不接受 using。文件格式由显式类别验证，不靠扩展名猜测；加载不会执行原脚本 main 或训练。模型与训练断点分别使用 `save model` / `save checkpoint` 及对应 `load` 形式，不可互换。

`reload model TARGET from PATH using OPTIMIZER;` 和 `reload checkpoint TARGET from PATH;` 原子更新已有模型句柄的状态；TARGET、PATH、OPTIMIZER 依次各求值一次。所有别名和 weightfs 挂载保留同一身份，成功后看到相同的新状态。候选先完整解析和验证，仅接受相同模型类别、结构、参数名/形状/冻结及概率属性、输出类型和计算程序；不支持任意架构热替换。model 显式重新创建优化器状态，checkpoint 恢复记录的状态和历史训练步数；运行时 revision 始终递增，因此旧梯度句柄过期。候选失败、结构不匹配或忙碌的分步执行会保留旧权重、优化器和 revision，不隐式训练或监听文件。此处 reload 不作用于普通文本/JSON，重新赋值 load 结果即可。

`save weights NET to PATH [overwrite];` 写标准 safetensors，`reload weights NET from PATH using OPTIMIZER;` 显式绑定已有源码模型并重置优化器/步数/自定义状态。仅接受精确键集合与 shape 的有限 F32、little-endian C row-major；不改名、不转置、不猜图。顺序模型 view/add 别名保留，仅存独立参数；文件不含训练状态。没有 `load weights` 表达式。示例：

```wgt
wgt 0.5
fn main() {
    # 显式初始化；源码检查与编译不会执行这些语句。
    let net = mlp([1,1], "identity", 4, "adam");
    save weights net to "weights.safetensors";
    let alias = net;
    reload weights net from "weights.safetensors" using "sgd";
    assert alias([1]) == net([1]);
    return net([1]);
}
```

text 只接受字符串，加载验证 UTF-8。JSON 支持有限数、字符串、bool、null、列表和对象；分类向量保存为数组，加载不恢复分类类型。模型、reader、kernel、gradient 等资源句柄不能偷偷序列化为 JSON。JSON 解析保留 serde_json 的默认深度限制，损坏输入或超出数值范围时报错。JSONL 继续使用 `jsonl`/`batch`/`close` 流式接口，不全量加载；本语法没有 bytes、任意二进制或隐式格式推断。

每次文件操作上限为 64 MiB，并按实际编码字节数消耗同一脚本 operation budget：一个字节计一个操作单位，不表示 FLOP。读取在分配整个文件前检查元数据，并有界读取以防文件增长；上限不代表 JSON 解析、模型准备及序列化的全部 RSS 都被覆盖。失败不退款。启用宿主策略时这些直接路径操作被拒绝，须使用已有授权 capability 文件接口；新语法不会绕过宿主授权。

保存先写同目录临时文件、同步后发布。默认以 hard link 原子发布，任何已存在目标（包括符号链接）都拒绝。显式 overwrite 使用原子 rename，仅允许普通文件，拒绝目录、符号链接和只读目标，保留原文件权限；不会沿目标符号链接写入。发布前失败保留旧内容，临时文件随后清理。这里保证单文件内容发布的原子性，不承诺断电后的目录项持久性或整个脚本的多文件事务。[存储规范](storage.md)仍适用。

`assert CONDITION;` 或 `assert CONDITION, MESSAGE;` 是原生语句。CONDITION 必须是 bool，不做数字/列表真值转换；只求值一次。仅失败才求值 MESSAGE，且失败消息必须为字符串，默认 `assertion failed`。条件可以含有带括号的子表达式，例如 `assert x == (1 + 0);`。失败报告所在行，测试运行器保留逐测试隔离。断言只以 `assert CONDITION;` 或 `assert CONDITION, MESSAGE;` 原生语句形式使用，不是函数调用表达式。源码语句分派不会被同名局部变量遮蔽。
