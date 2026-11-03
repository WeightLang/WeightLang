import json
import shutil

import pytest
import torch

from weightlang.artifact import load_artifact, sha256_file, write_safetensors
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.syntax import WgtError
from weightlang.weights import diff_weights, evaluate_artifact, export_weights


@pytest.fixture
def model(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(
        "wgt 0.2\nblock pair(n):\n  layer fc: linear(in=n, out=n)\n"
        "model test\ninput x: f32[2]\noutput y: f32[2]\n"
        "layer body: pair(n=2)\nlayer body2: pair(n=2)\ninit seed=42\n"
    )
    return compile_file(source, tmp_path / "weights").path


def rewrite(path, update):
    """Construct controlled candidates; retain random-init provenance."""
    artifact = load_artifact(path, warn_untrained=False)
    tensors = artifact.module.state_dict()
    update(tensors)
    manifest = artifact.manifest
    file = path / manifest["weights"]["file"]
    write_safetensors(file, tensors, {})
    manifest["weights"]["sha256"] = sha256_file(file)
    (path / "manifest.json").write_text(json.dumps(manifest))


def data_file(tmp_path, targets=None):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"inputs": [[1, 0], [0, 1]], "targets": [0, 1] if targets is None else targets}))
    return path


def test_export_values_prefix_budget_provenance(model):
    report = export_weights(model, limit=3)
    assert sum(len(t["values"]) for t in report["tensors"].values()) == 3
    assert sum(t["omitted"] for t in report["tensors"].values()) == 9
    report = export_weights(model, layer="body", limit=0)
    assert set(report["tensors"]) == {"body.fc.bias", "body.fc.weight"}
    actual = load_artifact(model, warn_untrained=False).module.state_dict()
    for name, record in report["tensors"].items():
        assert torch.equal(torch.tensor(record["values"]).reshape(record["shape"]), actual[name])
        assert record["origin"] == "random-init"
        assert record["omitted"] == 0
    assert report["artifact"]["untrained_tensors"]
    with pytest.raises(WgtError, match="no parameter"):
        export_weights(model, layer="bod")
    with pytest.raises(WgtError, match="non-negative"):
        export_weights(model, limit=-1)


def test_sparse_diff_counts_scalars(model, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(model, other)
    rewrite(other, lambda t: t["body.fc.weight"].flatten().__setitem__(0, t["body.fc.weight"].flatten()[0] + 0.5))
    report = diff_weights(model, other)
    assert report["elements"] == 12
    assert report["changed_elements"] == 1
    assert report["changed_percent"] == pytest.approx(100 / 12)
    assert report["tensors"]["body.fc.weight"]["max_abs_delta"] == pytest.approx(0.5)
    assert report["tensors"]["body.fc.weight"]["l2_delta"] == pytest.approx(0.5)
    assert diff_weights(model, model)["changed_elements"] == 0
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["interface"]["input"]["shape"] = [3]
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(WgtError, match="incompatible"):
        diff_weights(model, other)


def test_evaluation_same_data_exact_metrics_and_no_writes(model, tmp_path):
    def identity(tensors):
        for name, tensor in tensors.items():
            tensor.copy_(torch.eye(2) if name.endswith("weight") else torch.zeros(2))

    rewrite(model, identity)
    data = data_file(tmp_path)
    hashes = {p.name: sha256_file(p) for p in model.iterdir()}
    report = evaluate_artifact(model, data, "cross_entropy", compare=model)
    assert report["candidate"]["accuracy"] == 1.0
    assert report["candidate"]["loss"] == pytest.approx(0.31326166)
    assert report["candidate"]["artifact"]["untrained_tensors"]  # accuracy is not proof of training
    assert report["data"]["sha256"] == sha256_file(data)
    assert report["delta"] == {"loss": 0.0, "accuracy": 0.0}
    assert evaluate_artifact(model, data, "cross_entropy", compare=model) == report
    assert {p.name: sha256_file(p) for p in model.iterdir()} == hashes
    matrix_data = data_file(tmp_path, [[1, 0], [0, 1]])
    assert evaluate_artifact(model, matrix_data, "mse")["candidate"]["loss"] == 0
    assert evaluate_artifact(model, matrix_data, "bce")["candidate"]["accuracy"] == 1


@pytest.mark.parametrize("targets", [[0.5, 1], [0, 2], [False, 1], [0], [float("nan"), 1], [10**500, 1]])
def test_invalid_class_targets(model, tmp_path, targets):
    with pytest.raises(WgtError, match="class index"):
        evaluate_artifact(model, data_file(tmp_path, targets), "cross_entropy")


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"inputs": [], "targets": []},
        {"inputs": [[1, 0]], "targets": [0, 1]},
        {"inputs": [[1, "0"]], "targets": [0]},
        {"inputs": [[1, float("inf")]], "targets": [0]},
        {"inputs": [[1, 2, 3]], "targets": [0]},
        {"inputs": [[True, 0]], "targets": [0]},
    ],
)
def test_bad_data(model, tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(WgtError):
        evaluate_artifact(model, path, "cross_entropy")


def test_cli_and_tamper(model, tmp_path, capsys):
    assert main(["weights", str(model), "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == "weightlang.weights"
    assert main(["diff", str(model), str(model)]) == 0
    assert json.loads(capsys.readouterr().out)["changed_elements"] == 0
    data = data_file(tmp_path)
    assert main(["evaluate", str(model), "--data", str(data), "--loss", "cross_entropy"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["rows"] == 2
    assert main(["weights", str(model), "--limit", "-1"]) == 1
    assert "error:" in capsys.readouterr().err
    manifest = json.loads((model / "manifest.json").read_text())
    file = model / manifest["weights"]["file"]
    file.write_bytes(file.read_bytes() + b"tampered")
    for run in (
        lambda: export_weights(model),
        lambda: diff_weights(model, model),
        lambda: evaluate_artifact(model, data, "cross_entropy"),
    ):
        with pytest.raises(WgtError, match="sha256 mismatch"):
            run()


def test_nonfinite_weights_rejected(model, tmp_path):
    rewrite(model, lambda t: t["body.fc.bias"].__setitem__(0, float("nan")))
    data = data_file(tmp_path)
    for run in (
        lambda: export_weights(model),
        lambda: diff_weights(model, model),
        lambda: evaluate_artifact(model, data, "cross_entropy"),
    ):
        with pytest.raises(WgtError, match="non-finite"):
            run()


def test_evaluation_rejects_final_softmax(tmp_path):
    source = tmp_path / "probabilities.wgt"
    source.write_text(
        "wgt 0.2\nmodel probabilities\ninput x: f32[2]\noutput y: f32[2]\n"
        "layer fc: linear(in=2, out=2)\nlayer probs: softmax\ninit seed=42\n"
    )
    model = compile_file(source, tmp_path / "weights").path
    with pytest.raises(WgtError, match="raw logits"):
        evaluate_artifact(model, data_file(tmp_path), "cross_entropy")


def test_evaluation_reports_candidate_minus_baseline(model, tmp_path):
    baseline = tmp_path / "baseline"
    shutil.copytree(model, baseline)

    def identity(tensors):
        for name, tensor in tensors.items():
            tensor.copy_(torch.eye(2) if name.endswith("weight") else torch.zeros(2))

    rewrite(model, identity)
    report = evaluate_artifact(model, data_file(tmp_path), "cross_entropy", compare=baseline)
    assert report["candidate"]["accuracy"] == 1
    assert report["delta"]["loss"] == pytest.approx(report["candidate"]["loss"] - report["baseline"]["loss"])
    assert report["candidate"]["artifact"]["weights_sha256"] != report["baseline"]["artifact"]["weights_sha256"]
