"""Source sugar must preserve canonical graphs, checkpoints and gradients."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from weightlang.artifact import build_module, load_artifact
from weightlang.compiler import compile_file
from weightlang.syntax import WgtError, parse


def source(body, output="feature=4", shape="time=3, feature=4", version="0.3"):
    return f"wgt {version}\nmodel sugar\ninput x: f32[{shape}]\noutput y: f32[{output}]\n{body}\ninit seed=42\n"


SUGAR = source("""layer gate: linear(4) -> sigmoid from input
layer value: linear(4) from input
layer gated: gate * value
layer skip: gated + value
layer ff: residual(linear(8) -> gelu -> linear(4))
layer swap: transpose(feature, time)
layer pooled: mean(axis=time)""")
CANONICAL = SUGAR.replace("linear(4)", "linear(out=4)").replace("linear(8)", "linear(out=8)").replace(
    "gate * value", "mul(gate, value)").replace("gated + value", "add(gated, value)").replace(
    "transpose(feature, time)", 'rearrange("time feature -> feature time")').replace(
    "mean(axis=time)", 'reduce("feature time -> feature", op=mean)')


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_graph_values_and_gradients_match_explicit_operations(batch, reduction):
    program = parse(SUGAR.replace("mean(axis", f"{reduction}(axis"))
    explicit = parse(CANONICAL.replace("op=mean", f"op={reduction}"))
    assert [x.to_dict() for x in program.layers] == [x.to_dict() for x in explicit.layers]
    model = build_module([x.to_dict() for x in program.layers]).double()
    x = torch.randn(batch + (3, 4), dtype=torch.float64, requires_grad=True)
    value, gate = model.value(x), model.gate(x)
    skip = gate * value + value
    expected = getattr((model.ff.step2(model.ff.step1(model.ff.step0(skip))) + skip), reduction)(dim=-2)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    parameters = (x, *model.parameters())
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), parameters)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_checkpoint_roundtrip_and_stable_weight_names():
    root = Path("/tmp/weightlang")
    root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="sugar-", dir=root) as directory:
        scratch = Path(directory)
        path = scratch / "model.wgt"
        path.write_text(SUGAR)
        built = compile_file(path, scratch / "sugar")
        path.write_text(CANONICAL)
        explicit = compile_file(path, scratch / "explicit")
        loaded = load_artifact(built.path, warn_untrained=False)
        assert built.manifest["layers"] == explicit.manifest["layers"]
        assert built.module.state_dict().keys() == explicit.module.state_dict().keys()
        for key, value in built.module.state_dict().items():
            torch.testing.assert_close(value, explicit.module.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(loaded.module(x), built.module(x), rtol=0, atol=0)


def test_block_constants_from_and_loop_weight_sharing():
    program = parse(source("""const WIDTH = 4
block body(width):
  layer ff: residual(linear(width * 2) -> gelu -> linear(width))
layer stack: repeat(2, body(width=WIDTH))
layer shared: recur(2, body(width=WIDTH)) from input
layer pooled: sum(axis=time)"""))
    stack, shared, _ = program.layers
    assert stack.children[0].children[0].children[0].args["out"] == 8
    assert len(stack.children) == 2
    assert shared.op == "recur" and shared.args == {"steps": 2}


@pytest.mark.parametrize("expression, message", [
    ("input + missing", "unknown or forward"),
    ("input * input + input", "two names"),
    ("input + 1", "tensor name"),
    ("input + input from input", "already specify"),
    ("input + input -> relu", "named layers"),
    ("mean(axis=missing)", "unknown reduction axis"),
    ("transpose(feature, feature)", "each input axis"),
    ("transpose(feature)", "each input axis"),
    ("linear(4, bias=0)", "one positional"),
    ("linear(4 / 2)", "positive integer"),
    ("residual(linear(8) -> relu)", "residual"),
])
def test_invalid_sugar(expression, message):
    with pytest.raises(WgtError, match=message):
        parse(source("layer fc: linear(out=4)\nlayer result: " + expression))


@pytest.mark.parametrize("expression", ["mean(axis=time)", "sum(axis=time)", "transpose(feature, time)"])
def test_axis_sugar_requires_fully_named_input(expression):
    with pytest.raises(WgtError, match="all input axes"):
        parse(source("layer fc: linear(out=4)\nlayer result: " + expression, shape="3, feature=4"))


def test_reduction_output_must_match_and_no_implicit_broadcast():
    with pytest.raises(WgtError, match="shape/dtype mismatch"):
        parse(source("layer fc: linear(4)\nlayer result: mean(axis=feature)", shape="feature=4"))
    with pytest.raises(WgtError, match="shape"):
        parse(source("layer fc: linear(1)\nlayer result: fc + input"))


@pytest.mark.parametrize("version", ["0.1", "0.2"])
@pytest.mark.parametrize("expression", ["linear(4)", "input + input", "input * input", "mean(axis=time)", "sum(axis=time)", "transpose(time, feature)"])
def test_old_versions_reject_sugar(version, expression):
    with pytest.raises(WgtError, match="wgt 0.3"):
        parse(source("layer result: " + expression, shape="4", output="4", version=version))
