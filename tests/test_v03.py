import json
from pathlib import Path

import pytest
import torch

from weightlang.artifact import load_artifact
from weightlang.compiler import compile_file
from weightlang.syntax import WgtError, parse
from weightlang.weights import evaluate_artifact

ROOT = Path(__file__).resolve().parents[1]


def build(tmp_path, text, name="m"):
    source = tmp_path / f"{name}.wgt"
    source.write_text(text)
    return compile_file(source, tmp_path / name)


def source(layers, inp="f32[3, 4]", out="f32[3, 4]"):
    return f"wgt 0.3\nmodel demo\ninput x: {inp}\noutput y: {out}\n{layers}\ninit seed=42\n"


@pytest.mark.parametrize(
    ("layers", "inp", "out", "x"),
    [
        ("layer e: embedding(vocab=5, dim=4)", "i64[3]", "f32[3, 4]", torch.tensor([[0, 1, 4]])),
        ("layer p: position_embedding(dim=4, max_len=8)", "f32[3, 4]", "f32[3, 4]", torch.randn(2, 3, 4)),
        ("layer a: self_attention(dim=4, heads=2)", "f32[3, 4]", "f32[3, 4]", torch.randn(2, 3, 4)),
        (
            "layer c: conv2d(in=1, out=2, kernel=3, padding=1)\nlayer p: maxpool2d(kernel=2, stride=2)",
            "f32[1, 4, 4]",
            "f32[2, 2, 2]",
            torch.randn(2, 1, 4, 4),
        ),
        (
            "layer c: conv2d(in=1, out=2, kernel=2)\nlayer p: globalavgpool2d",
            "f32[1, 4, 4]",
            "f32[2]",
            torch.randn(2, 1, 4, 4),
        ),
        ("layer fc: linear(in=4, out=4)\nlayer p: meanpool", "f32[3, 4]", "f32[4]", torch.randn(2, 3, 4)),
        ("layer fc: linear(in=4, out=4)\nlayer p: flatten", "f32[3, 4]", "f32[12]", torch.randn(2, 3, 4)),
    ],
)
def test_modern_roundtrip(tmp_path, layers, inp, out, x):
    result = build(tmp_path, source(layers, inp, out))
    loaded = load_artifact(result.path, warn_untrained=False)
    assert result.manifest["format_version"] == 3
    assert torch.equal(result.module(x), loaded.predict(x))
    assert torch.equal(result.module(x[:1])[0], loaded.predict(x[0]))
    assert result.manifest["layers"][-1]["output"]["shape"] == list(result.module(x).shape[1:])
    result.module(x).sum().backward()
    assert all(p.grad is not None for p in result.module.parameters())


def test_causal_attention_reference_and_future_independence(tmp_path):
    result = build(tmp_path, source("layer a: self_attention(dim=4, heads=2, causal=1)"))
    a = result.module.a
    x = torch.randn(2, 3, 4)
    q, k, v = a.qkv(x).reshape(2, 3, 3, 2, 2).unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / (2**0.5)
    scores = scores.masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), float("-inf"))
    reference = a.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 3, 4))
    assert torch.allclose(a(x), reference, atol=1e-6)
    changed = x.clone()
    changed[:, 2] += 100
    assert torch.equal(a(x)[:, :2], a(changed)[:, :2])


@pytest.mark.parametrize(
    ("layers", "inp", "out", "message"),
    [
        ("layer a: self_attention(dim=4, heads=3)", "f32[3, 4]", "f32[3, 4]", "divisible"),
        ("layer a: self_attention(dim=4, heads=2, causal=2)", "f32[3, 4]", "f32[3, 4]", "causal"),
        ("layer p: position_embedding(dim=4, max_len=2)", "f32[3, 4]", "f32[3, 4]", "max_len"),
        ("layer e: embedding(vocab=5, dim=4)", "f32[3]", "f32[3, 4]", "i64"),
        ("layer c: conv2d(in=2, out=2, kernel=2)", "f32[1, 4, 4]", "f32[2, 3, 3]", "channel"),
        ("layer c: conv2d(in=1, out=2, kernel=9)", "f32[1, 4, 4]", "f32[2, 3, 3]", "empty"),
    ],
)
def test_shape_errors(tmp_path, layers, inp, out, message):
    with pytest.raises(WgtError, match=message) as exc:
        parse(source(layers, inp, out))
    assert exc.value.line == 5


def test_version_gate():
    with pytest.raises(WgtError, match="requires 'wgt 0.3'"):
        parse(source("layer e: embedding(vocab=5, dim=4)", "i64[3]").replace("wgt 0.3", "wgt 0.2"))


@pytest.mark.parametrize("x", [[0, 1.5, 2], [-1, 0, 1], [0, 1, float("nan")], [0, 1, 2**63], [0, 1, 5]])
def test_bad_tokens(tmp_path, x):
    result = build(tmp_path, source("layer e: embedding(vocab=5, dim=4)", "i64[3]"))
    with pytest.raises(WgtError):
        load_artifact(result.path, warn_untrained=False).predict(x)


@pytest.mark.parametrize("example", ["transformer", "cnn"])
def test_examples_train_and_reproduce(tmp_path, example):
    text = (ROOT / "examples" / f"{example}.wgt").read_text()
    a = build(tmp_path, text, "a")
    b = build(tmp_path, text, "b")
    assert a.manifest["weights"]["sha256"] == b.manifest["weights"]["sha256"]
    assert a.manifest["training"]["train_accuracy"] == 1
    if example == "transformer":
        data = tmp_path / "evaluation.json"
        data.write_text(json.dumps({"inputs": [[0, 1, 2, 3]], "targets": [[1, 2, 3, 0]]}))
        assert evaluate_artifact(a.path, data, "cross_entropy")["candidate"]["accuracy"] == 1


def test_encoder_and_conv_pool_std(tmp_path):
    text = source("layer body: transformer.encoder(dim=4, heads=2, hidden=8)").replace(
        "model demo", "use std/transformer\nmodel demo"
    )
    result = build(tmp_path, text)
    x = torch.randn(1, 3, 4)
    other = x.clone()
    other[:, 2, 0] += 5
    assert not torch.allclose(result.module(x)[:, 0], result.module(other)[:, 0])
    text = source(
        "layer body: vision.conv_pool(in=1, out=2, kernel=3, pool=2)", "f32[1, 6, 6]", "f32[2, 2, 2]"
    ).replace("model demo", "use std/vision\nmodel demo")
    result = build(tmp_path, text, "cnn")
    assert result.module(torch.ones(1, 1, 6, 6)).shape == (1, 2, 2, 2)


def test_manifest_shape_tampering(tmp_path):
    result = build(tmp_path, source("layer a: self_attention(dim=4, heads=2)"))
    manifest = result.path / "manifest.json"
    record = json.loads(manifest.read_text())
    record["layers"][0]["input"]["shape"] = [5, 4]
    manifest.write_text(json.dumps(record))
    with pytest.raises(WgtError, match="contract mismatch"):
        load_artifact(result.path)
