"""Saved-weight diagnostics must observe the actual checkpoint without modifying it."""
import json

import pytest
import torch

from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.debugger import execute
from weightlang.syntax import WgtError, parse_file


@pytest.fixture
def checkpoint(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(
        "wgt 0.3\nmodel affine\ninput x: f32[1]\noutput y: f32[1]\n"
        "layer fc: linear(in=1,out=1)\ninit seed=42\n"
        "data train:\n  [1] -> [0]\n"
        "train data=train loss=mse optimizer=sgd lr=0.1 epochs=1\n"
    )
    artifact = compile_file(source, tmp_path / "weights").path
    from weightlang.artifact import load_artifact
    loaded = load_artifact(artifact, warn_untrained=False)
    with torch.no_grad():
        loaded.module.fc.weight.fill_(2)
        loaded.module.fc.bias.fill_(1)
    saved = loaded.save(tmp_path / "known")
    return source, saved.path


def test_trace_uses_saved_values_and_records_identity(checkpoint, capsys):
    source, artifact = checkpoint
    assert main(["trace", str(source), "--weights", str(artifact), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output"]["mean"] == 3
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert report["weight_source"]["sha256"] == manifest["weights"]["sha256"]
    assert report["weights"][0]["before"]["mean"] == 2


def test_debug_measures_actual_update_and_never_writes(checkpoint):
    source, artifact = checkpoint
    before = {p.name: p.read_bytes() for p in artifact.iterdir() if p.is_file()}
    rng = torch.get_rng_state().clone()
    report = execute(parse_file(source), mode="debug", weights=artifact, steps=1)
    assert report["losses"] == [9]
    assert report["final_loss"] == pytest.approx(3.24)
    row = next(r for r in report["weights"] if r["parameter"] == "fc.weight")
    assert row["gradient_status"] == "nonzero"
    assert row["gradient"]["norm"] == 6
    assert row["update_norm"] == pytest.approx(0.6)
    assert row["relative_update"] == pytest.approx(0.3)
    assert row["changed_elements"] == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert before == {p.name: p.read_bytes() for p in artifact.iterdir() if p.is_file()}


def test_incompatible_graph_rejected_before_artifact_load(checkpoint, monkeypatch):
    source, artifact = checkpoint
    source.write_text(source.read_text().replace("init seed=42", "layer result: sigmoid\ninit seed=42"))
    def unexpected(*args, **kwargs):
        pytest.fail("incompatible artifact must not allocate weights")
    monkeypatch.setattr("weightlang.artifact.load_artifact", unexpected)
    with pytest.raises(WgtError, match="identical architecture"):
        execute(parse_file(source), mode="trace", weights=artifact)


def test_corrupt_checkpoint_is_rejected(checkpoint):
    source, artifact = checkpoint
    with (artifact / "weights.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(WgtError, match="sha256 mismatch"):
        execute(parse_file(source), mode="trace", weights=artifact)


def test_limit_precedes_saved_weight_allocation(checkpoint, monkeypatch):
    source, artifact = checkpoint
    def unexpected(*args, **kwargs):
        pytest.fail("parameter limit must be checked before loading")
    monkeypatch.setattr("weightlang.artifact.load_artifact", unexpected)
    with pytest.raises(WgtError, match="allocation limit"):
        execute(parse_file(source), mode="trace", weights=artifact, max_parameters=1)


def test_freezing_and_zero_norm_reporting(checkpoint):
    source, artifact = checkpoint
    source.write_text(source.read_text() + "freeze fc.weight\n")
    report = execute(parse_file(source), mode="debug", weights=artifact, steps=1)
    weight = next(r for r in report["weights"] if r["parameter"] == "fc.weight")
    assert weight["gradient_status"] == "frozen"
    assert weight["update_norm"] == 0
    assert weight["gradient"] is None


def test_trace_does_not_need_training_data(checkpoint, capsys):
    source, artifact = checkpoint
    source.write_text(source.read_text() + "freeze all\n")
    assert main(["trace", str(source), "--weights", str(artifact), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert all(not row["trainable"] for row in report["weights"])


def test_zero_gradient_and_zero_initial_norm(checkpoint):
    source, artifact = checkpoint
    from weightlang.artifact import load_artifact
    loaded = load_artifact(artifact, warn_untrained=False)
    with torch.no_grad():
        for p in loaded.module.parameters():
            p.zero_()
    zero = loaded.save(artifact.parent / "zero")
    report = execute(parse_file(source), mode="debug", weights=zero.path, steps=1)
    assert all(r["gradient_status"] == "zero" for r in report["weights"])
    assert all(r["relative_update"] is None for r in report["weights"])
    json.dumps(report, allow_nan=False)


def test_nonfinite_unwatched_parameter_is_detected(tmp_path):
    source = tmp_path / "bad.wgt"
    source.write_text(
        "wgt 0.3\nmodel bad\ninput x: f32[1]\noutput y: f32[1]\n"
        "layer unused: linear(in=1,out=1)\nlayer fc: linear(in=1,out=1) from input\ninit seed=1\n"
    )
    from weightlang.artifact import load_artifact
    artifact = load_artifact(compile_file(source, tmp_path / "weights").path, warn_untrained=False)
    with torch.no_grad():
        artifact.module.unused.weight.fill_(float("nan"))
    # Save bytes and update their integrity hash to test numerical validation,
    # independently of file corruption detection.
    from safetensors.torch import save_file
    from weightlang.artifact import sha256_file
    save_file(artifact.module.state_dict(), str(artifact.path / "weights.safetensors"))
    manifest = artifact.manifest
    manifest["weights"]["sha256"] = sha256_file(artifact.path / "weights.safetensors")
    (artifact.path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(WgtError, match="non-finite initial parameter `unused.weight`"):
        execute(parse_file(source), mode="trace", weights=artifact.path, watch="fc")


def test_trace_explicit_sample_bypasses_missing_csv(checkpoint, capsys):
    source, artifact = checkpoint
    source.write_text(source.read_text().replace(
        "data train:\n  [1] -> [0]",
        'data train: file="missing.csv" sha256=' + '0' * 64,
    ))
    sample = source.parent / "sample.json"
    sample.write_text('[1]')
    assert main(["trace", str(source), "--weights", str(artifact), "--sample", str(sample), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["output"]["mean"] == 3


def test_trace_default_sample_loads_pinned_csv(checkpoint, capsys):
    import hashlib
    source, artifact = checkpoint
    csv = source.parent / "samples.csv"
    csv.write_text("2,0\n")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    source.write_text(source.read_text().replace(
        "data train:\n  [1] -> [0]",
        f'data train: file="samples.csv" sha256={digest}',
    ))
    assert main(["trace", str(source), "--weights", str(artifact), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["output"]["mean"] == 5
