# 原生断言

本地尚未发布的 WGT 0.5 在普通函数中支持 `assert CONDITION;` 和 `assert CONDITION, MESSAGE;`。这属于语句语法，不能被同名局部变量遮蔽；forward/loss 不允许断言副作用。

CONDITION 必须是 bool，不隐式转换数字、列表或张量，只求值一次。MESSAGE 只在条件为 false 时求值，失败消息必须为字符串；省略时使用 `assertion failed`。错误包含语句所在行，原生测试运行器继续隔离各测试的失败。解释执行和 AOT 共用严格布尔契约。

```wgt
wgt 0.5
fn main() {
    let values = [1,2,3];
    assert len(values) == 3, "长度错误";
    assert (values[0] == 1);
    return values;
}
```

兼容函数 `assert(condition,message)` 保留两个实参急求值和返回 null 的契约，包括写成独立语句的旧调用。`assert true, missing();` 不调用 missing；`assert(true,missing());` 仍会因求值第二实参报错。括号包围单个条件属于自然表达式，例如 `assert(true);`、`assert (1) == 1;`。新示例优先使用原生语句。
