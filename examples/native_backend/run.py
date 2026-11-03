"""Run with WEIGHTLANG_NATIVE=target/debug/wgt-native PYTHONPATH=src python examples/native_backend/run.py."""
from pathlib import Path
from weightlang.native import NativeSession

# base 与 delta 都是显式提供的示例数值，并非训练结果。
# [out,in] 两个轴分别对应输出分数与输入特征；delta 不会修改 base。
parameters = {
    "base": {"shape": [2, 2], "kind": "resident", "data": [1, 2, 3, 4]},
    "delta": {"shape": [2, 2], "kind": "resident", "data": [0.1, 0, 0, -0.1]},
    "projection.weight": {"kind": "add", "base": "base", "delta": "delta"},
}
with NativeSession.from_source(
    Path(__file__).with_name("model.wgt"), parameters, budget_bytes=64
) as session:
    print(session.report)
    print(session([0.1, 0.2]))
