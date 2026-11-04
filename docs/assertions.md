# 原生断言

本地 WGT 0.5 在普通函数中支持 `assert CONDITION;` 和 `assert CONDITION, MESSAGE;`。这属于语句语法，不能被同名局部变量遮蔽；forward/loss 不允许断言副作用。

CONDITION 必须是 bool，不隐式转换数字、列表或张量，只求值一次。MESSAGE 只在条件为 false 时求值，失败消息必须为字符串；省略时使用 `assertion failed`。错误包含语句所在行，原生测试运行器继续隔离各测试的失败。解释执行和 AOT 共用严格布尔契约。

```wgt
wgt 0.5
fn main() {
    let values = [1,2,3];
    assert len(values) == 3, "长度错误";
    assert values[0] == 1;
    return values;
}
```

断言只使用 `assert CONDITION;` 或 `assert CONDITION, MESSAGE;` 语句形式。MESSAGE 仅在断言失败时求值，失败消息必须为字符串。断言是语句，不是表达式。
