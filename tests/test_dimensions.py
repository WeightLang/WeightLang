"""Integration contracts for v0.3 named dimensions, inference and artifacts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn.functional as F

from weightlang.artifact import build_module, load_artifact
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.decompile import decompile
from weightlang.syntax import WgtError, parse


def source(body, *, shape="time=3, feature=8", output="time=3, feature=8", blocks="", version="0.3"):
    return (
        f"wgt {version}\nmodel dimensions\ninput x: f32[{shape}]\n"
        f"output y: f32[{output}]\n{blocks}{body}\ninit seed=17\n"
    )


@pytest.fixture
def scratch():
    root = Path("/tmp/weightlang")
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dimensions-", dir=root) as directory:
        yield Path(directory)


def compile_text(scratch, text):
    path = scratch / "model.wgt"
    path.write_text(text, encoding="utf-8")
    return compile_file(path, scratch / "weights")


def rewrite_manifest(built, mutate):
    path = built.path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_parse_named_and_mixed_ports():
    program = parse(source("layer fc: linear(out=8)"))
    assert program.input.shape == program.output.shape == (3, 8)
    assert program.input.axes == program.output.axes == ("time", "feature")
    assert program.layers[0].args == {"in": 8, "out": 8}
    assert program.layers[0].input_spec == {"dtype": "f32", "shape": [3, 8], "axes": ["time", "feature"]}
    mixed = parse(source("layer fc: linear(out=8)", shape="3, feature=8", output="3, feature=8"))
    assert mixed.input.axes == mixed.output.axes == (None, "feature")


NESTED = """block inner(width):
  layer fc: linear(out=width)
block outer(width):
  layer inner: inner(width=width)
  layer finish: linear(out=3)
"""


def test_inferred_linear_nested_blocks_and_from_branches():
    program = parse(source(
        "layer narrow: linear(out=2)\n"
        "layer body: outer(width=5) from x\n"
        "layer result: linear(out=4) from narrow",
        shape="7", output="4", blocks=NESTED,
    ))
    narrow, body, result = program.layers
    assert narrow.args == {"in": 7, "out": 2}
    assert body.children[0].children[0].args == {"in": 7, "out": 5}
    assert body.children[1].args == {"in": 5, "out": 3}
    assert result.args == {"in": 2, "out": 4}
    model = build_module([layer.to_dict() for layer in program.layers])
    x = torch.randn(2, 7)
    torch.testing.assert_close(model(x), model.result(model.narrow(x)))


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("transform", ["transpose", "split", "merge", "mean", "sum"])
def test_transforms_match_torch_and_parameter_gradients(batch, transform):
    statements = {
        "transpose": ('rearrange("time feature -> feature time")', "feature=8, time=3"),
        "split": ('rearrange("time (head width) -> head time width", head=4)', "head=4, time=3, width=2"),
        "merge": ('rearrange("time feature -> (feature time)")', "24"),
        "mean": ('reduce("time feature -> feature", op=mean)', "feature=8"),
        "sum": ('reduce("time feature -> feature", op=sum)', "feature=8"),
    }
    expression, output = statements[transform]
    program = parse(source(f"layer fc: linear(out=8)\nlayer result: {expression}", output=output))
    model = build_module([layer.to_dict() for layer in program.layers]).double()
    x = torch.randn(batch + (3, 8), dtype=torch.float64, requires_grad=True)
    projected = F.linear(x, model.fc.weight, model.fc.bias)
    if transform == "transpose":
        expected = projected.transpose(-1, -2)
    elif transform == "split":
        expected = projected.reshape(batch + (3, 4, 2)).transpose(-3, -2)
    elif transform == "merge":
        expected = projected.transpose(-1, -2).flatten(-2)
    else:
        expected = getattr(projected, transform)(dim=-2)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    parameters = (x, *model.parameters())
    actual_grads = torch.autograd.grad((actual * upstream).sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


ROUNDTRIP = source(
    'layer body: project()\n'
    'layer split: rearrange("time (head width) -> head time width", head=4)\n'
    'layer pooled: reduce("head time width -> head width", op=mean)',
    output="head=4, width=2",
    blocks="block project():\n  layer fc: linear(out=8)\n",
)


def test_compile_load_preserves_axes_explicit_inference_and_values(scratch):
    built = compile_text(scratch, ROUNDTRIP)
    manifest = built.manifest
    assert manifest["format_version"] == 3
    assert manifest["interface"]["input"]["axes"] == ["time", "feature"]
    assert manifest["interface"]["output"]["axes"] == ["head", "width"]
    assert manifest["layers"][0]["layers"][0]["args"] == {"in": 8, "out": 8}
    assert manifest["layers"][1]["input"]["axes"] == ["time", "feature"]
    assert manifest["layers"][1]["output"]["axes"] == ["head", "time", "width"]
    assert manifest["layers"][1]["args"]["sizes"] == {"head": 4}
    loaded = load_artifact(built.path, warn_untrained=False)
    assert loaded.manifest == manifest
    for batch in [(), (2,), (2, 3)]:
        x = torch.randn(batch + (3, 8))
        projected = F.linear(x, built.module.body.fc.weight, built.module.body.fc.bias)
        expected = projected.reshape(batch + (3, 4, 2)).mean(dim=-3)
        torch.testing.assert_close(built.module(x), expected)
        torch.testing.assert_close(loaded.module(x), expected)


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("location", ["interface", "layer"])
def test_tampered_named_axis_contracts_rejected(scratch, direction, location):
    built = compile_text(scratch, ROUNDTRIP)

    def mutate(manifest):
        contract = manifest["interface"][direction] if location == "interface" else manifest["layers"][1][direction]
        contract["axes"] = list(reversed(contract["axes"]))

    rewrite_manifest(built, mutate)
    with pytest.raises(WgtError, match="axis|contract"):
        load_artifact(built.path, warn_untrained=False)


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("axes", [[], ["row"], ["row", "row"], ["bad-name", None], [7, None]])
def test_invalid_axis_metadata_rejected_even_with_unnamed_contracts(scratch, direction, axes):
    # Wildcard compatibility must not bypass the axis metadata schema itself.
    built = compile_text(scratch, source("layer fc: linear(out=8)", shape="3,8", output="3,8"))
    rewrite_manifest(built, lambda manifest: manifest["interface"][direction].update(axes=axes))
    with pytest.raises(WgtError):
        load_artifact(built.path, warn_untrained=False)


def test_artifact_cannot_omit_resolved_linear_input(scratch):
    built = compile_text(scratch, ROUNDTRIP)
    rewrite_manifest(built, lambda manifest: manifest["layers"][0]["layers"][0]["args"].pop("in"))
    with pytest.raises(WgtError, match="explicit in"):
        load_artifact(built.path, warn_untrained=False)


def test_cli_shapes_displays_names_and_nested_paths(scratch, capsys):
    path = scratch / "model.wgt"
    path.write_text(ROUNDTRIP, encoding="utf-8")
    assert main(["check", str(path), "--shapes"]) == 0
    output = capsys.readouterr().out
    assert "input x: f32[time=3, feature=8]" in output
    assert "body.fc: f32[time=3, feature=8] -> f32[time=3, feature=8]" in output
    assert "split: f32[time=3, feature=8] -> f32[head=4, time=3, width=2]" in output
    assert "pooled: f32[head=4, time=3, width=2] -> f32[head=4, width=2]" in output
    assert main(["check", str(path)]) == 0
    assert "body.fc:" not in capsys.readouterr().out


@pytest.mark.parametrize("version", ["0.1", "0.2"])
@pytest.mark.parametrize("feature", ["named", "inferred", "rearrange", "reduce"])
def test_dimensions_features_reject_old_versions(version, feature):
    shape = "feature=4" if feature == "named" else "4"
    body = "layer fc: linear(in=4, out=4)"
    if feature == "inferred":
        body = "layer fc: linear(out=4)"
    elif feature == "rearrange":
        body += '\nlayer result: rearrange("feature -> feature")'
    elif feature == "reduce":
        body += '\nlayer result: reduce("feature -> feature", op=sum)'
    with pytest.raises(WgtError, match="wgt 0.3"):
        parse(source(body, shape=shape, output="4", version=version))


SWAP = 'block swap():\n  layer swap: rearrange("time feature -> feature time")\n'


@pytest.mark.parametrize("operation", ["add", "mul", "concat", "mix", "where", "residual", "recur"])
def test_equal_size_swapped_axes_rejected(operation):
    body = "layer fc: linear(out=4)\n"
    if operation in {"residual", "recur"}:
        expression = "residual(swap())" if operation == "residual" else "recur(2, swap())"
        body += f"layer result: {expression}"
    else:
        body += 'layer swapped: rearrange("time feature -> feature time") from fc\n'
        refs = "fc, swapped, fc" if operation in {"mix", "where"} else "fc, swapped"
        body += f"layer result: {operation}({refs})"
    output = "4,8" if operation == "concat" else "4,4"
    with pytest.raises(WgtError, match="axis"):
        parse(source(body, shape="time=4, feature=4", output=output, blocks=SWAP))


@pytest.mark.parametrize("expression", [
    'rearrange("time (head width) -> head time width")',
    'rearrange("time (head width channel) -> head time width channel", head=2)',
])
def test_group_inference_ambiguity_is_wgt_error(expression):
    with pytest.raises(WgtError, match="infer"):
        parse(source(f"layer fc: linear(out=8)\nlayer result: {expression}"))


@pytest.mark.parametrize("expression", [
    'reduce("time feature ->", op=mean)',
    'reduce("time feature ->", op=sum)',
    'reduce("time feature -> ()", op=sum)',
])
@pytest.mark.parametrize("following_linear", [False, True])
def test_empty_scalar_reduction_is_wgt_error(expression, following_linear):
    body = f"layer fc: linear(out=8)\nlayer result: {expression}"
    if following_linear:
        body += "\nlayer next: linear(out=8)"
    with pytest.raises(WgtError, match="scalar|group|axis|shape/dtype mismatch"):
        parse(source(body))


@pytest.mark.parametrize("args", ["", ", op=max", ", op=median", ", op=1", ', op="mean"'])
def test_reduce_missing_or_invalid_op(args):
    with pytest.raises(WgtError, match="mean|sum|op"):
        parse(source(f'layer fc: linear(out=8)\nlayer result: reduce("time feature -> feature"{args})', output="8"))


@pytest.mark.parametrize("direction", ["input", "output"])
def test_duplicate_port_axes_rejected(direction):
    options = {"shape" if direction == "input" else "output": "time=3, time=8"}
    with pytest.raises(WgtError, match="duplicate axis"):
        parse(source("layer fc: linear(out=8)", **options))


@pytest.mark.parametrize("pattern", ["time time -> time", "time feature -> time time", "time (head head) -> time head"])
def test_duplicate_pattern_axes_rejected(pattern):
    with pytest.raises(WgtError, match="unique|duplicate"):
        parse(source(f'layer fc: linear(out=8)\nlayer result: reduce("{pattern}", op=sum)'))


@pytest.mark.parametrize("op", ["mean", "sum"])
def test_decompile_axis_transforms_rebuilds_exact_checkpoint(scratch, op):
    built = compile_text(scratch, ROUNDTRIP.replace("op=mean", f"op={op}"))
    recovered = scratch / "recovered.wgt"
    report = decompile(str(built.path), recovered, references=True)
    assert report["kind"] == "checkpoint" and report["executable"]
    program = parse(recovered.read_text(encoding="utf-8"))
    assert program.input.axes == ("time", "feature")
    assert program.output.axes == ("head", "width")
    assert program.layers[0].children[0].args == {"in": 8, "out": 8}
    for recovered_layer, original_layer in zip(program.layers[1:], built.manifest["layers"][1:]):
        assert recovered_layer.op == original_layer["op"]
        assert recovered_layer.args == original_layer["args"]
        assert recovered_layer.input_spec == original_layer["input"]
        assert recovered_layer.output_spec == original_layer["output"]
    rebuilt = compile_file(recovered, scratch / "recovered-weights")
    loaded = load_artifact(rebuilt.path, warn_untrained=False)
    original_state = built.module.state_dict()
    assert loaded.module.state_dict().keys() == original_state.keys()
    for name, tensor in loaded.module.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)
    for batch in [(), (2, 3)]:
        x = torch.randn(batch + (3, 8), requires_grad=True)
        actual, expected = loaded.module(x), built.module(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * upstream).sum(), (x, *loaded.module.parameters()))
        expected_grads = torch.autograd.grad((expected * upstream).sum(), (x, *built.module.parameters()))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["residual", "recur"])
def test_unnamed_boundary_can_acquire_consistent_names(operation):
    expression = "residual(label())" if operation == "residual" else "recur(2, label())"
    program = parse(source(
        f"layer fc: linear(out=4)\nlayer result: {expression}",
        shape="4,4", output="time=4, feature=4",
        blocks='block label():\n  layer label: rearrange("time feature -> time feature")\n',
    ))
    assert program.layers[-1].output_spec["axes"] == ["time", "feature"]
    model = build_module([layer.to_dict() for layer in program.layers])
    x = torch.randn(2, 4, 4)
    projected = model.fc(x)
    torch.testing.assert_close(model(x), projected * (2 if operation == "residual" else 1))


@pytest.mark.parametrize("operation", ["residual", "recur"])
def test_unnamed_boundary_does_not_hide_named_body_axis_swap(operation):
    # The body explicitly calls its input (time, feature), then swaps it.
    # Residual adds mismatched meanings; recur's second iteration feeds
    # (feature, time) into a body expecting (time, feature), despite equal sizes.
    expression = "residual(swap())" if operation == "residual" else "recur(2, swap())"
    with pytest.raises(WgtError, match="axis"):
        parse(source(
            f"layer fc: linear(out=4)\nlayer result: {expression}",
            shape="4,4", output="4,4", blocks=SWAP,
        ))
