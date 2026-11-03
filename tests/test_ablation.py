import json

import pytest

from weightlang.ablation import run_ablation
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.syntax import WgtError


@pytest.fixture
def ablation(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(
        "wgt 0.2\nmodel test\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(in=1, out=1)\ninit seed=42\n"
    )
    model = compile_file(source, tmp_path / "weights").path
    (tmp_path / "data.json").write_text(json.dumps({"inputs": [[1], [2]], "targets": [[1], [2]]}))
    suite = {
        "format": "weightlang.ablation",
        "format_version": 1,
        "data": "data.json",
        "loss": "mse",
        "variants": [
            {"name": "zero", "interventions": [{"op": "zero_output", "target": "fc"}]},
            {"name": "skip", "interventions": [{"op": "skip", "target": "fc"}]},
            {
                "name": "weights",
                "interventions": [
                    {"op": "zero_tensor", "target": "fc.weight"},
                    {"op": "zero_tensor", "target": "fc.bias"},
                ],
            },
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    return model, path, suite


def test_independent_variants_and_read_only(ablation, capsys):
    model, path, _ = ablation
    original = {p.name: p.read_bytes() for p in model.iterdir()}
    report = run_ablation(model, path)
    assert [v["metrics"]["loss"] for v in report["variants"]] == [2.5, 0, 2.5]
    for variant in report["variants"]:
        assert variant["delta"]["loss"] == variant["metrics"]["loss"] - report["baseline"]["metrics"]["loss"]
    assert run_ablation(model, path) == report
    assert {p.name: p.read_bytes() for p in model.iterdir()} == original
    assert main(["ablate", str(model), "--suite", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize(
    "edit",
    [
        {"op": "unknown", "target": "fc"},
        {"op": "skip", "target": "missing"},
        {"op": "zero_tensor", "target": "fc"},
        {"op": "zero_output", "target": ""},
        {"op": "skip", "target": []},
    ],
)
def test_invalid_intervention(ablation, edit):
    model, path, suite = ablation
    suite["variants"][0]["interventions"] = [edit]
    path.write_text(json.dumps(suite))
    with pytest.raises(WgtError):
        run_ablation(model, path)


def test_overlapping_and_duplicate_variants(ablation):
    model, path, suite = ablation
    suite["variants"][0]["interventions"].append({"op": "zero_tensor", "target": "fc.weight"})
    path.write_text(json.dumps(suite))
    with pytest.raises(WgtError, match="overlapping"):
        run_ablation(model, path)
    suite["variants"] = [suite["variants"][1]] * 2
    path.write_text(json.dumps(suite))
    with pytest.raises(WgtError, match="unique"):
        run_ablation(model, path)


def test_skip_shape_mismatch(ablation, tmp_path):
    _, path, suite = ablation
    source = tmp_path / "wide.wgt"
    source.write_text(
        "wgt 0.2\nmodel wide\ninput x: f32[1]\noutput y: f32[2]\nlayer fc: linear(in=1, out=2)\ninit seed=42\n"
    )
    model = compile_file(source, tmp_path / "weights").path
    (tmp_path / "data.json").write_text(json.dumps({"inputs": [[1]], "targets": [[1, 2]]}))
    suite["variants"] = [suite["variants"][1]]
    path.write_text(json.dumps(suite))
    with pytest.raises(WgtError, match="variant 'skip'"):
        run_ablation(model, path)


def test_classification_metrics_and_nested_target(tmp_path):
    source = tmp_path / "classifier.wgt"
    source.write_text(
        "wgt 0.2\nuse std/heads\nmodel classifier\ninput x: f32[2]\noutput y: f32[2]\n"
        "layer fc: heads.classifier(in=2, classes=2)\ninit seed=42\n"
    )
    model = compile_file(source, tmp_path / "weights").path
    (tmp_path / "data.json").write_text(json.dumps({"inputs": [[1, 0], [0, 1]], "targets": [0, 1]}))
    suite = {
        "format": "weightlang.ablation",
        "format_version": 1,
        "data": "data.json",
        "loss": "cross_entropy",
        "variants": [{"name": "zero", "interventions": [{"op": "zero_output", "target": "fc.logits"}]}],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    report = run_ablation(model, path)
    assert report["variants"][0]["metrics"]["accuracy"] == 0.5
    assert report["variants"][0]["metrics"]["loss"] == pytest.approx(0.69314718)
    assert report["variants"][0]["delta"]["accuracy"] == 0.5 - report["baseline"]["metrics"]["accuracy"]
