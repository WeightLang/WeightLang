# PyTorch 专用兼容模型

本地 `target torch` 兼容扩展使用 `wgt 0.5`，尚未发布。
新模型优先采用[共享 WGT 模型](shared-models.md)，由宿主或编译命令选择后端。
此扩展保留 Python/PyTorch 语义，仅由 PyTorch 执行；`wgt check --target rust`、
Rust 模型加载与导出会明确拒绝，不提供 Python 隐式回退。
它面向已有动态 PyTorch 网络的完整架构迁移，使用共享 Rust 前端生成普通 PyTorch 模块。
参数初始化、张量形状和 dtype 采用 PyTorch 运行时语义。

## 源码

每个模型源码目录使用 `model.wgt`。版本头后的第二个有效声明为 `target torch`，
随后声明唯一的 `host MODULE`，并可用 `extern NAME, NAME` 列出宿主符号。
宿主模块名由点分标识符组成，extern 不可重复，声明在模型之前。

```wgt
wgt 0.5
target torch
host weightlang.modules
extern nn, torch

# 输入为 [batch,time,width]，状态为 [1,batch,width]。
# 每次调用共享 cell 参数；宿主显式携带状态，初始化后尚未训练。
model Recurrent extends Module {
    init(self, width=4) {
        super().__init__();
        self.cell = nn.GRU(width, width, batch_first=True);
    }
    fn forward(self, x, hidden=None) {
        return self.cell(x, hidden);
    }
}
```

`model NAME [extends BASE] {` 声明 PyTorch 模型，默认基类 `Module` 对应 `torch.nn.Module`。
`init(...) {` 定义构造器，`fn NAME(...) {` 和 `async fn NAME(...) {` 定义方法。
方法显式声明 `self`。方法签名支持默认值、关键字参数和 Python 类型注解。
构造参数负责确定宽度等尺寸；`--dim` 用于原生模型的维度声明。

控制块包括 `if`、`elif`、`else`、`for`、`while`、`with`、`try`、`except` 和 `finally`。
每个块头以空格和 `{` 结尾，关闭 `}` 独占一行。每条赋值、调用、return、yield、
raise 等语句以 `;` 结尾。一个逻辑语句可以跨行，字符串中的换行应显式转义。
顶层可定义纯张量 helper，helper 名称与模型及 extern 名称唯一。
类的方法支持 `@torch.inference_mode()` 等装饰器，顶层 `fn main` 会被拒绝。
模型之间可继承，helper 可访问同一编译单元的定义及显式 extern。

表达式、索引、字典、推导式、方法调用、布尔 mask 和算术采用 Python/PyTorch 语义。
RustPython 在 Rust 前端验证生成代码的语法。变长输入、GRU 状态、多返回值、
可微 gather、STE 和权重共享由显式模型公式决定。语法检查不提供动态形状证明，
动态模型参数量在实例化后通过 PyTorch 检查。原生模型的 f32/轴合同仍由其前端验证。

## 编译和绑定

```sh
wgt check python/examples/dynamic_recurrent/model.wgt
wgt compile python/examples/dynamic_recurrent/model.wgt --target torch
wgt format --check python/examples/dynamic_recurrent/model.wgt
```

上述命令不初始化模型、不运行 Python、不连接服务。编译生成 `weightlang.torch.v1`，
其中 `kind` 为 `torch_module`，`torch_module` 包含 host、externs、模型/helper 名称
以及生成的 Python 代码。原生 Rust/service 和 native AOT 使用独立的模型计划。

Python SDK 的入口如下，均位于 `weightlang.modules`。

- `bind_models(path, namespace, *, compiler=None)` 编译项目源码并返回名称到模型类/helper 的映射。
- `classes_from_ir(definition, namespace, *, trusted=False)` 绑定已编译定义，执行需传入 `trusted=True`。
- `save_module(model, path, *, kwargs, metadata=None, origin="edited", training=None)` 保存新完整目录。
- `load_module(path, *, trusted=False)` 加载目录，返回含 `path`、`manifest`、`module` 的对象。
- `restore_parameters(cls, values, *, kwargs, frozen=None, training=False, buffers=None)` 完整绑定已有张量。

`namespace` 必须提供每个 extern。构建返回值时执行生成代码的定义，构造器在实例化时执行。
应用显式信任这些生成代码及宿主代码。通用原生模型加载入口持续使用自己的 IR。
编译器按显式 `compiler`、`WGT_COMPILER`、PATH 中的 `wgt` 顺序选择。
不会隐式使用 checkout 中可能过时的 debug 二进制。源码编译所用的编译器需要包含此扩展。

## 保存和恢复

目录内保存源码、编译定义、依赖模型快照、构造配置、冻结标记、权重摘要和宿主文件摘要。
参数与所有注册 buffer 保存在 safetensors，非持久 buffer 使用 manifest 列出的独立键。
参数 alias 的值必须一致，恢复后重新绑定同一参数对象。
模型类需要支持 PyTorch meta device，以便先创建注册布局，再绑定保存张量，避免随机抽样。
`restore_parameters` 的 `buffers` 必须完整提供未包含在 state_dict 中的注册 buffer。
参数/形状不匹配、非有限值、缺失绑定和不一致 alias 均报错。

构造配置支持 JSON 标量、列表、字符串键字典、元组和 dataclass。
完整模型加载可以使用保存的编译定义，宿主包与其摘要仍需匹配。
已有目录默认拒绝覆盖，保存失败清理本次独占预留的目录。
训练记录由调用方提供，优化器与数据采集状态由应用的训练检查点管理。

语法、CLI、formatter、编辑器基础诊断、PyTorch 数值和制品测试分别位于
`crates/wgt-language/tests/torch_model.rs`、`crates/wgt/tests/torch_export.rs` 和
`python/tests/test_torch_modules.py`。LightNest 的独立原实现对照进一步验证完整架构。
编辑器对此目标提供语法诊断，动态张量形状与参数数量在运行时确认。

TextMate 编辑器为该目标提供单引号字符串、宿主声明和 Python 控制关键词高亮。
Tree-sitter 的结构化语法继续覆盖原生 WGT；动态目标使用 Rust/LSP 的基础语法诊断。
