import json

import pytest

from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.experiments import run_experiment
from weightlang.patches import apply_patch, patch_template
from weightlang.syntax import WgtError


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(
        "wgt 0.2\nmodel affine\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(in=1, out=1)\ninit seed=42\n"
    )
    original = compile_file(source, tmp_path / "weights").path

    def edit(base, name, weight, bias):
        patch = patch_template(base, "test")
        patch["reason"] = name
        patch["edits"] = [
            {"tensor": "fc.weight", "index": 0, "value": weight},
            {"tensor": "fc.bias", "index": 0, "value": bias},
        ]
        file = tmp_path / (name + ".patch.json")
        file.write_text(json.dumps(patch))
        apply_patch(base, file, tmp_path / name)
        return tmp_path / name

    base = edit(original, "base", 1, 0)
    candidate = edit(base, "candidate", 2, 0)
    rejected = edit(base, "rejected", 2, 1)
    (tmp_path / "target.json").write_text(json.dumps({"inputs": [[1], [2]], "targets": [[2], [4]]}))
    (tmp_path / "regression.json").write_text(json.dumps({"inputs": [[0]], "targets": [[0]]}))
    spec = {
        "format": "weightlang.experiment",
        "format_version": 1,
        "cases": [
            {"name": "target", "role": "target", "data": "target.json", "loss": "mse", "rules": {"max_loss": 0}},
            {
                "name": "regression",
                "role": "regression",
                "data": "regression.json",
                "loss": "mse",
                "rules": {"max_loss_increase": 0},
            },
        ],
    }
    file = tmp_path / "suite.json"
    file.write_text(json.dumps(spec))
    return base, candidate, rejected, file, spec


def test_success_and_regression_failure(experiment, capsys):
    base, good, bad, file, _ = experiment
    report = run_experiment(good, base, file)
    assert report["passed"] and report["changes"]["changed_elements"] == 1
    assert report["cases"][0]["candidate"]["loss"] == 0
    assert report["cases"][1]["checks"]["max_loss_increase"]["actual"] == 0
    rejected = run_experiment(bad, base, file)
    assert not rejected["passed"]
    assert not rejected["cases"][1]["passed"]
    assert main(["experiment", str(bad), "--baseline", str(base), "--suite", str(file)]) == 1
    assert not json.loads(capsys.readouterr().out)["passed"]
    assert main(["experiment", str(good), "--baseline", str(base), "--suite", str(file)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]
    assert run_experiment(good, base, file) == report


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(format_version=True),
        lambda s: s["cases"].pop(),
        lambda s: s["cases"][0].update(rules={}),
        lambda s: s["cases"][0].update(rules={"magic": 1}),
        lambda s: s["cases"][0].update(rules={"max_loss": float("nan")}),
        lambda s: s["cases"][0].update(rules={"max_loss": True}),
        lambda s: s["cases"][0].update(rules={"min_accuracy": 2}),
        lambda s: s["cases"][1].update(name="target"),
        lambda s: s["cases"][0].update(rules={"min_accuracy": 1}),
    ],
)
def test_invalid_rules(experiment, mutation):
    base, good, _, file, spec = experiment
    mutation(spec)
    file.write_text(json.dumps(spec))
    with pytest.raises(WgtError):
        run_experiment(good, base, file)
