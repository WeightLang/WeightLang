import json
from pathlib import Path

import pytest
import torch

from weightlang.artifact import load_artifact, sha256_file
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.syntax import STD_DIR, WgtError, parse, parse_file

ROOT = Path(__file__).resolve().parents[1]


def source(layers="layer fc: linear(in=2, out=2)", prefix="", suffix="", width=2):
    return f"wgt 0.2\n{prefix}model demo\ninput x: f32[2]\noutput y: f32[{width}]\n{layers}\ninit seed=42\n{suffix}"


def build(tmp_path, text, name="m"):
    path = tmp_path / f"{name}.wgt"
    path.write_text(text)
    return compile_file(path, tmp_path / name)


@pytest.mark.parametrize("op", ["gelu", "silu", "softmax", "identity", "layernorm(dim=2)", "dropout(p=0.3)"])
def test_ops_roundtrip_and_reference(tmp_path, op):
    result = build(tmp_path, source(f"layer fc: linear(in=2, out=2)\nlayer x: {op}"))
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    loaded = load_artifact(result.path, warn_untrained=False)
    assert torch.equal(result.module(x), loaded.predict(x))
    assert torch.equal(loaded.predict(x), loaded.predict(x))
    refs = {
        "gelu": torch.nn.GELU(),
        "silu": torch.nn.SiLU(),
        "softmax": torch.nn.Softmax(-1),
        "identity": torch.nn.Identity(),
        "layernorm(dim=2)": torch.nn.LayerNorm(2),
        "dropout(p=0.3)": torch.nn.Identity(),
    }
    assert torch.equal(result.module(x), refs[op](result.module.fc(x)))
    result.module(x).sum().backward()
    assert result.module.fc.weight.grad is not None
    if op.startswith("layernorm"):
        for name in ("x.weight", "x.bias"):
            assert result.manifest["weights"]["tensors"][name]["origin"] == "random-init"


@pytest.mark.parametrize(
    ("op", "message"),
    [
        ("layernorm(dim=3)", "shape mismatch"),
        ("dropout(p=1.5)", "dropout p"),
        ("linear(in=2.5, out=2)", "positive integer"),
    ],
)
def test_op_errors_have_line(op, message):
    text = source(f"layer fc: linear(in=2, out=2)\nlayer bad: {op}")
    with pytest.raises(WgtError, match=message) as exc:
        parse(text)
    assert exc.value.line == 6


def test_version_and_unknown_std():
    with pytest.raises(WgtError, match="requires 'wgt 0.2'") as exc:
        parse(source("layer x: layernorm(dim=2)").replace("wgt 0.2", "wgt 0.1"))
    assert exc.value.line == 5
    with pytest.raises(WgtError, match="available: std/blocks, std/heads, std/nn") as exc:
        parse(source(prefix="use std/missing\n"))
    assert exc.value.line == 2


def test_local_block(tmp_path):
    result = build(tmp_path, source("layer body: pair(n=2)", "block pair(n):\n  layer fc: linear(in=n, out=n)\n"))
    assert set(result.module.state_dict()) == {"body.fc.weight", "body.fc.bias"}


@pytest.mark.parametrize(
    ("prefix", "call", "message"),
    [
        ("block pair(n):\n  layer fc: linear(in=missing, out=n)\n", "pair(n=2)", "unknown parameter"),
        ("block pair(n):\n  layer fc: linear(in=n, out=n)\n", "pair()", "takes"),
        ("block pair(n):\n  layer fc: pair(n=n)\n", "pair(n=2)", "block recursion"),
        ("block pair(n):\n  init seed=1\n", "pair(n=2)", "layer"),
        ("block pair(n):\n", "pair(n=2)", "has no layers"),
        ("block linear(n):\n  layer fc: identity\n", "linear(n=2)", "built-in"),
    ],
)
def test_bad_blocks(prefix, call, message):
    with pytest.raises(WgtError, match=message) as exc:
        parse(source(f"layer body: {call}", prefix))
    assert exc.value.line is not None


def test_use_cycle(tmp_path):
    (tmp_path / "a.wgt").write_text('wgt 0.2\nuse "./b.wgt" as b\n')
    (tmp_path / "b.wgt").write_text('wgt 0.2\nuse "./a.wgt" as a\n')
    p = tmp_path / "main.wgt"
    p.write_text(source(prefix='use "./a.wgt" as a\n'))
    with pytest.raises(WgtError, match="use cycle") as exc:
        parse_file(p)
    assert exc.value.line == 2


@pytest.mark.parametrize(
    ("lib", "call", "width"),
    [
        ("nn", "mlp(in=2, hidden=4, out=2)", 2),
        ("heads", "classifier(in=2, classes=3)", 3),
        ("blocks", "residual_mlp(dim=2, hidden=4)", 2),
    ],
)
def test_std_snapshot(tmp_path, lib, call, width):
    result = build(tmp_path, source(f"layer body: {lib}.{call}", f"use std/{lib}\n", width=width))
    actual = {"layers": result.manifest["layers"], "tensors": sorted(result.module.state_dict())}
    expected = json.loads((ROOT / "tests" / "snapshots" / f"{lib}.json").read_text())
    assert actual == expected
    assert result.manifest["layers"][0]["block"] == f"std/{lib}.{call.split('(')[0]}"
    assert result.manifest["libraries"][lib]["sha256"] == sha256_file(STD_DIR / f"{lib}.wgt")


def test_residual(tmp_path):
    result = build(tmp_path, source("layer body: residual(nn.mlp(in=2, hidden=4, out=2))", "use std/nn\n"))
    x = torch.randn(4, 2)
    child = result.module.body
    assert torch.equal(result.module(x), x + child.fc2(child.act(child.fc1(x))))
    with pytest.raises(WgtError, match="must keep the width"):
        parse(source("layer body: residual(nn.mlp(in=2, hidden=4, out=3))", "use std/nn\n"))
    with pytest.raises(WgtError, match="takes a block call"):
        parse(source("layer body: residual(linear(in=2, out=2))"))


DATA = "data train:\n  [-2, 0] -> [0]\n  [2, 0] -> [1]\n  [0, 2] -> [2]\n"
TRAIN = "train data=train loss=cross_entropy optimizer=adamw lr=0.05 epochs=150 batch=2 weight_decay=0.01\n"


def test_training_and_validation(tmp_path):
    text = source(
        "layer fc: linear(in=2, out=3)",
        suffix=DATA + DATA.replace("train:", "test:") + TRAIN.rstrip() + " validate=test\n",
        width=3,
    )
    a, b = build(tmp_path, text, "a"), build(tmp_path, text, "b")
    assert a.manifest["weights"]["sha256"] == b.manifest["weights"]["sha256"]
    t = a.manifest["training"]
    assert t["train_accuracy"] == 1
    assert t["batch"] == 2 and t["weight_decay"] == 0.01
    assert set(t["validation"]) == {"loss", "accuracy", "dataset"}
    assert t["validation"]["dataset"]["name"] == "test"
    assert main(["audit", str(a.path)]) == 1  # no dataset license


@pytest.mark.parametrize("label", ["0.5", "3", "-1"])
def test_invalid_classes(label):
    with pytest.raises(WgtError, match="class index") as exc:
        parse(source("layer fc: linear(in=2, out=3)", suffix=DATA.replace("[0]", f"[{label}]") + TRAIN, width=3))
    assert exc.value.line == 7


def test_invalid_training():
    with pytest.raises(WgtError, match="raw logits"):
        parse(source("layer fc: linear(in=2, out=3)\nlayer prob: softmax", suffix=DATA + TRAIN, width=3))
    with pytest.raises(WgtError, match="different dataset"):
        parse(source("layer fc: linear(in=2, out=3)", suffix=DATA + TRAIN.rstrip() + " validate=train", width=3))


def test_dropout_training_reproducible(tmp_path):
    text = source("layer fc: linear(in=2, out=3)\nlayer drop: dropout(p=0.2)", suffix=DATA + TRAIN, width=3)
    a, b = build(tmp_path, text, "a"), build(tmp_path, text, "b")
    assert a.manifest["weights"]["sha256"] == b.manifest["weights"]["sha256"]


@pytest.mark.parametrize(
    ("csv", "extra", "error"),
    [
        ("1,2,0\n", "", None),
        ("a,b,c\n1,2,0\n", "skip=1", None),
        ("a,b,c\n", "", "only numbers"),
        ("1,2\n", "", "followed by targets"),
        ("1,2,0\n", "wrong", "source pins"),
        ("1,2,0\n", "missing", "sha256"),
    ],
)
def test_csv(tmp_path, csv, extra, error):
    path = tmp_path / "data.csv"
    path.write_text(csv)
    digest = "0" * 64 if extra == "wrong" else sha256_file(path)
    attrs = "" if extra == "missing" else f"sha256={digest}"
    if extra == "skip=1":
        attrs += " skip=1"
    text = source(suffix=f'data d: file="data.csv" {attrs}\n')
    if error:
        with pytest.raises(WgtError, match=error) as exc:
            build(tmp_path, text)
        assert exc.value.line == 7
    else:
        result = build(tmp_path, text)
        assert result.path.is_dir()


@pytest.mark.parametrize("prefix", ["body.fc1", "body"])
def test_dotted_import_and_freeze(tmp_path, prefix):
    layers = "layer body: nn.mlp(in=2, hidden=4, out=2)\nlayer head: linear(in=2, out=2)"
    original = build(tmp_path, source(layers, "use std/nn\n"), "original")
    suffix = (
        f'import {prefix} from "{original.path}" layer={prefix} freeze\n'
        "data d:\n  [1, 2] -> [0, 1]\ntrain data=d loss=mse optimizer=sgd lr=0.1 epochs=2\n"
    )
    result = build(tmp_path, source(layers, "use std/nn\n", suffix).replace("seed=42", "seed=7"), "reuse")
    for name, param in result.module.named_parameters():
        assert param.requires_grad == (not name.startswith(prefix + "."))
        if name.startswith(prefix + "."):
            assert torch.equal(param, original.module.state_dict()[name])
    bad = source(layers.replace("hidden=4", "hidden=5"), "use std/nn\n", suffix)
    with pytest.raises(WgtError, match="incompatible import"):
        build(tmp_path, bad, "bad")


def test_v1_and_xor_hash(tmp_path):
    result = compile_file(ROOT / "examples/xor.wgt", tmp_path)
    assert result.manifest["weights"]["sha256"] == "7969227a9c0e3528317e1cd22b8da766e5b1a7a705e0ba96078a86cde8296ba2"
    manifest = result.manifest.copy()
    manifest["format_version"] = 1
    manifest.pop("libraries")
    (result.path / "manifest.json").write_text(json.dumps(manifest))
    assert torch.equal(result.module(torch.ones(1, 2)), load_artifact(result.path).predict([[1, 1]]))


def test_cli_std_library_and_nested(tmp_path, capsys):
    assert main(["std"]) == 0
    assert "std/nn.mlp(in, hidden, out)" in capsys.readouterr().out
    assert main(["check", str(STD_DIR / "nn.wgt")]) == 0
    assert "ok: library" in capsys.readouterr().out
    text = source("layer body: blocks.residual_mlp(dim=2, hidden=4)", "use std/blocks\n")
    result = build(tmp_path, text)
    assert main(["check", str(tmp_path / "m.wgt")]) == 0
    assert "26 parameters" in capsys.readouterr().out
    assert main(["inspect", str(result.path)]) == 0
    assert "residual(" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        ("xor", [0, 1, 1, 0]),
        ("and", [0, 0, 0, 1]),
        ("or", [0, 1, 1, 1]),
        ("nand", [1, 1, 1, 0]),
        ("nor", [1, 0, 0, 0]),
        ("xnor", [1, 0, 0, 1]),
    ],
)
def test_all_gates(tmp_path, gate, expected):
    result = compile_file(ROOT / f"packages/gates/{gate}.wgt", tmp_path)
    prediction = load_artifact(result.path).predict([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert (prediction >= 0.5).int().flatten().tolist() == expected


def test_iris_audit_and_reproducibility(tmp_path, capsys):
    a = compile_file(ROOT / "packages/iris/iris.wgt", tmp_path / "a")
    b = compile_file(ROOT / "packages/iris/iris.wgt", tmp_path / "b")
    assert a.manifest["weights"]["sha256"] == b.manifest["weights"]["sha256"]
    assert a.manifest["training"]["validation"]["accuracy"] >= 0.9
    assert main(["audit", str(a.path)]) == 0
    assert capsys.readouterr().out.count("pass:") == 4
    manifest = a.manifest
    manifest["training"]["validation"]["dataset"].pop("license")
    (a.path / "manifest.json").write_text(json.dumps(manifest))
    assert main(["audit", str(a.path)]) == 1
    assert "fail: dataset hashes and licenses" in capsys.readouterr().out


@pytest.mark.parametrize("example", ["std_mlp", "residual"])
def test_std_examples_train(tmp_path, example):
    result = compile_file(ROOT / f"examples/{example}.wgt", tmp_path)
    assert result.manifest["training"]["train_accuracy"] == 1


def test_zero_weight_decay_recorded(tmp_path):
    result = build(
        tmp_path,
        source(
            "layer fc: linear(in=2, out=3)", suffix=DATA + TRAIN.replace("weight_decay=0.01", "weight_decay=0"), width=3
        ),
    )
    assert result.manifest["training"]["weight_decay"] == 0


def test_documented_plain_pytorch_loader(tmp_path):
    result = build(tmp_path, source("layer body: blocks.residual_mlp(dim=2, hidden=4)", "use std/blocks\n"))
    doc = (ROOT / "docs/artifact.md").read_text()
    code = doc.split("Without WeightLang", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    code = code.replace('root = "build/xor_mlp"', f"root = {str(result.path)!r}")
    namespace = {}
    exec(code, namespace)
    x = torch.randn(3, 2)
    assert torch.equal(namespace["net"](x), result.module(x))


def test_cli_trained_import_does_not_claim_random(tmp_path, capsys):
    original = build(tmp_path, source("layer fc: linear(in=2, out=3)", suffix=DATA + TRAIN, width=3), "original")
    text = source("layer fc: linear(in=2, out=3)", suffix=f'import fc from "{original.path}"\n', width=3)
    path = tmp_path / "reuse.wgt"
    path.write_text(text)
    assert main(["build", str(path), "-o", str(tmp_path / "reuse")]) == 0
    output = capsys.readouterr().out
    assert "retained imported trained weights" in output
    assert "WARNING" not in output
