"""Behavioral contracts for branches, gating, independent depth and recurrent state."""

import json
from pathlib import Path

import pytest
import torch

from weightlang.artifact import build_module, load_artifact
from weightlang.compiler import compile_file
from weightlang.syntax import WgtError, parse


def source(body, shape="4", output="4", blocks="", version="0.3"):
    return f"wgt {version}\nmodel flow\ninput x: f32[{shape}]\noutput y: f32[{output}]\n{blocks}{body}\ninit seed=4\n"


def module(text):
    program = parse(text)
    return build_module([layer.to_dict() for layer in program.layers])


GATE = """block gated(dim):
  layer a: linear(in=dim, out=dim) from input
  layer g: sigmoid
  layer b: linear(in=dim, out=dim) from input
  layer result: mul(g, b)
"""


def test_gating_matches_equations_and_gradients():
    model = module(source("layer body: gated(dim=4)", blocks=GATE))
    x = torch.randn(3, 4, requires_grad=True)
    actual = model(x)
    body = model.body
    expected = torch.sigmoid(body.a(x)) * body.b(x)
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters()) + (x,)
    actual_grad = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), parameters)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("op", ["add", "mul", "concat", "mix", "where"])
def test_merge_matches_torch(op):
    refs = "g, a, b" if op in ("mix", "where") else "a, b"
    model = module(
        source(
            f"""layer a: linear(in=4, out=4)
layer b: linear(in=4, out=4) from x
layer g: linear(in=4, out=4) from input
layer y: {op}({refs})""",
            output="8" if op == "concat" else "4",
        )
    )
    x = torch.randn(2, 4)
    a, b, g = model.a(x), model.b(x), model.g(x)
    expected = {
        "add": a + b,
        "mul": a * b,
        "concat": torch.cat([a, b], -1),
        "mix": g * a + (1 - g) * b,
        "where": torch.where(g > 0, a, b),
    }[op]
    torch.testing.assert_close(model(x), expected)


def test_repeat_is_independent_recur_shares_parameters():
    repeat = module(source("layer body: repeat(3, gated(dim=4))", blocks=GATE))
    recur = module(source("layer body: recur(3, gated(dim=4))", blocks=GATE))
    assert sum(p.numel() for p in repeat.parameters()) == 3 * sum(p.numel() for p in recur.parameters())
    assert repeat.body.step0.a.weight is not repeat.body.step1.a.weight
    x = torch.randn(2, 4)
    expected = x
    for _ in range(3):
        expected = torch.sigmoid(recur.body.a(expected)) * recur.body.b(expected)
    torch.testing.assert_close(recur(x), expected)
    recur(x).sum().backward()
    assert all(p.grad is not None for p in recur.parameters())


CELL = """block cell():
  layer joined: concat(input, state)
  layer projection: linear(in=6, out=4)
  layer next: tanh
"""


def test_scan_matches_manual_recurrence_and_bptt():
    model = module(source("layer memory: scan(hidden=4, cell())", "5,2", "5,4", CELL))
    x = torch.randn(3, 5, 2, requires_grad=True)
    h = torch.zeros(3, 4)
    expected = []
    for t in range(5):
        h = torch.tanh(model.memory.projection(torch.cat([x[:, t], h], -1)))
        expected.append(h)
    actual = model(x)
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(actual, expected)
    params = tuple(model.parameters()) + (x,)
    grad = torch.autograd.grad(actual[:, -1].sum(), params, retain_graph=True)
    reference = torch.autograd.grad(expected[:, -1].sum(), params)
    for a, b in zip(grad, reference):
        torch.testing.assert_close(a, b)
    assert torch.count_nonzero(grad[-1][:, 0]) > 0  # last state depends on first input
    torch.testing.assert_close(model(x), actual)  # state resets between calls
    torch.testing.assert_close(model(x[0]), actual[0])


@pytest.mark.parametrize(
    "body,blocks,match",
    [
        ("layer a: linear(in=4, out=4) from missing", "", "unknown or forward"),
        ("layer a: linear(in=4, out=4) from a", "", "unknown or forward"),
        ("layer a: linear(in=4, out=4) from state", "", "unknown or forward"),
        ("layer input: linear(in=4, out=4)", "", "reserved"),
        ("layer a: linear(in=4, out=2)\nlayer y: add(input, a)", "", "identical shapes"),
        ("layer a: linear(in=4, out=4)\nlayer y: mix(a, input)", "", "exactly 3"),
        ("layer a: repeat(0, linear(in=4, out=4))", "", "positive integer"),
        ("layer a: recur(1025, linear(in=4, out=4))", "", "steps <= 1024"),
        ("layer a: recur(2, linear(in=4, out=8))", "", "preserve"),
        ("layer a: scan(hidden=4, cell())", CELL, "scan expects"),
    ],
)
def test_invalid_graphs(body, blocks, match):
    with pytest.raises(WgtError, match=match):
        parse(source(body, blocks=blocks))


@pytest.mark.parametrize(
    "body",
    [
        "layer a: linear(in=4, out=4) from input",
        "layer a: repeat(2, linear(in=4, out=4))",
        "layer a: add(input, input)",
    ],
)
def test_new_syntax_requires_v03(body):
    with pytest.raises(WgtError, match="requires? 'wgt 0.3'"):
        parse(source(body, version="0.2"))


@pytest.mark.parametrize("example", ["swiglu", "recurrent", "gated_transformer", "gru"])
def test_examples_round_trip(tmp_path, example):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{example}.wgt"
    built = compile_file(path, tmp_path / "models")
    loaded = load_artifact(built.path, warn_untrained=False)
    shape = built.manifest["interface"]["input"]["shape"]
    x = (
        torch.randint(0, 128, (2, *shape))
        if built.manifest["interface"]["input"]["dtype"] == "i64"
        else torch.randn(2, *shape)
    )
    built.module.eval()
    torch.testing.assert_close(loaded.module(x), built.module(x))
    assert list(loaded.module.state_dict()) == list(built.module.state_dict())


def test_training_and_tampered_wiring(tmp_path):
    path = tmp_path / "train.wgt"
    path.write_text(
        source("layer body: gated(dim=4)", blocks=GATE)
        + """data examples:
  [0,0,0,0] -> [0,0,0,0]
  [1,1,1,1] -> [1,1,1,1]
train data=examples loss=mse optimizer=adam lr=0.03 epochs=60
"""
    )
    built = compile_file(path, tmp_path / "models")
    model = load_artifact(built.path, warn_untrained=False)
    assert torch.mean((model.module(torch.ones(1, 4)) - 1) ** 2) < 0.03
    manifest_path = built.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"][0]["layers"][-1]["inputs"][0] = "not_a_tensor"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WgtError, match="unknown or forward"):
        load_artifact(built.path, warn_untrained=False)


def test_pipeline_and_library_gru_equations():
    model = module(source("use std/recurrent as r\nlayer memory: r.gru(features=2, hidden=4)", "3,2", "3,4"))
    cell = model.memory.sequence
    x = torch.randn(2, 3, 2)
    h = torch.zeros(2, 4)
    outputs = []
    for item in x.unbind(1):
        joined = torch.cat([item, h], -1)
        r = torch.sigmoid(cell.reset.step0(joined))
        z = torch.sigmoid(cell.update.step0(joined))
        candidate = torch.tanh(cell.candidate.step0(torch.cat([item, r * h], -1)))
        h = z * h + (1 - z) * candidate
        outputs.append(h)
    torch.testing.assert_close(model(x), torch.stack(outputs, 1))


def test_bounded_expansion():
    with pytest.raises(WgtError, match="expansion exceeds"):
        parse(source("layer huge: repeat(1024, repeat(1024, linear(in=4, out=4)))"))


def test_top_level_input_alias_can_be_shadowed():
    model = module(source("layer x: linear(in=4, out=4)\nlayer y: identity from x"))
    x = torch.randn(2, 4)
    torch.testing.assert_close(model(x), model.x(x))


def test_scan_training_with_placement_and_reload(tmp_path):
    text = source("layer memory: scan(hidden=4, cell())\nlayer head: last -> linear(in=4, out=1)", "3,2", "1", CELL)
    text += """data train:
  [0,0,0,0,0,0] -> [0]
  [1,1,1,1,1,1] -> [1]
train data=train loss=mse optimizer=adam lr=0.03 epochs=80
"""
    path = tmp_path / "sequence.wgt"
    path.write_text(text)
    built = compile_file(path, tmp_path / "models", placements={"memory.projection": "cpu"})
    artifact = load_artifact(built.path, warn_untrained=False)
    x = torch.tensor([[[0.0, 0.0]] * 3, [[1.0, 1.0]] * 3])
    torch.testing.assert_close(artifact.module(x), built.module(x))
    assert torch.mean((artifact.module(x) - torch.tensor([[0.0], [1.0]])) ** 2) < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_gated_cross_device_gradients():
    from weightlang.devices import apply_placement, placement_plan

    model = module(source("layer body: gated(dim=4)", blocks=GATE))
    apply_placement(model, placement_plan(model, "cpu", {"body.a": "cuda:0"}))
    out = model(torch.randn(2, 4))
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_repeat_merge_uses_each_iterations_input():
    model = module(source("layer projection: linear(in=4, out=4)\nlayer depth: repeat(3, add(input, input))"))
    x = torch.randn(2, 4)
    torch.testing.assert_close(model(x), model.projection(x) * 8)


def test_recurrence_can_name_layer_hidden():
    cell = "block cell():\n  layer joined: concat(input, state)\n  layer hidden: linear(in=6, out=4)\n"
    model = module(source("layer memory: scan(hidden=4, cell())", "3,2", "3,4", cell))
    assert model(torch.randn(2, 3, 2)).shape == (2, 3, 4)


def test_shared_loop_artifact_import_and_freeze(tmp_path):
    text = source("layer body: recur(3, gated(dim=4))", blocks=GATE)
    first = tmp_path / "original.wgt"
    first.write_text(text)
    original = compile_file(first, tmp_path / "original")
    second = tmp_path / "imported.wgt"
    second.write_text(
        text
        + f'import body from "{original.path}" layer=body\nfreeze body.a\n'
        + """data train:
  [0,0,0,0] -> [1,1,1,1]
train data=train loss=mse optimizer=adam lr=0.01 epochs=2
"""
    )
    imported = compile_file(second, tmp_path / "imported")
    assert not any(p.requires_grad for p in imported.module.body.a.parameters())
    torch.testing.assert_close(imported.module.body.a.weight, original.module.body.a.weight)
    loaded = load_artifact(imported.path, warn_untrained=False)
    assert len(loaded.module.state_dict()) == 4  # one copy, despite three iterations
    x = torch.randn(2, 4)
    torch.testing.assert_close(loaded.module(x), imported.module(x))
