import json

import pytest
import torch

from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.repository import WeightRepository
from weightlang.reuse import ReuseError, preview
from weightlang.syntax import WgtError


def program(name="base", width=4, body="body", extra="", train=""):
    return (
        f"wgt 0.3\nmodel {name}\ninput x: f32[2]\noutput y: f32[1]\n"
        f"layer {body}: linear(in=2, out={width})\nlayer act: tanh\n"
        f"layer head: linear(in={width}, out=1)\n{extra}init seed=42\n{train}"
    )


def build(root, code, **options):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "model.wgt"
    source.write_text(code)
    return compile_file(source, root / "weights", **options)


def test_add_layer_reuses_existing_and_keeps_new_initialization(tmp_path):
    old = build(tmp_path / "a", program())
    text = program("next", extra="layer calibration: linear(in=1, out=1)\n")
    fresh = build(tmp_path / "fresh", text)
    new = build(tmp_path / "b", text, reuse_source=old.path)
    report = new.manifest["reuse"]
    assert report["status"] == "partial" and report["architecture_changed"]
    assert report["quality"] == "not_evaluated"
    assert report["summary"]["copied_values"] == 17
    assert report["summary"]["initialized_values"] == 2
    for name, tensor in new.module.state_dict().items():
        expected = fresh.module.state_dict()[name] if name.startswith("calibration.") else old.module.state_dict()[name]
        assert torch.equal(tensor, expected)
    assert new.manifest["weights"]["tensors"]["body.weight"]["upstream_origin"] == "random-init"
    assert new.manifest["weights"]["tensors"]["calibration.weight"]["origin"] == "random-init"


def test_widen_exact_and_opt_in_overlap(tmp_path):
    old = build(tmp_path / "a", program())
    text = program("wide", width=6)
    exact = build(tmp_path / "exact", text, reuse_source=old.path)
    assert exact.manifest["reuse"]["summary"]["copied_values"] == 1  # stable output bias
    fresh = build(tmp_path / "fresh", text)
    new = build(tmp_path / "overlap", text, reuse_source=old.path, reuse_policy="overlap")
    report = new.manifest["reuse"]
    assert report["status"] == "partial"
    assert report["summary"]["copied_values"] == 17
    assert report["summary"]["initialized_values"] == 8
    assert torch.equal(new.module.body.weight[:4], old.module.body.weight)
    assert torch.equal(new.module.body.weight[4:], fresh.module.body.weight[4:])
    assert torch.equal(new.module.head.weight[:, :4], old.module.head.weight)
    assert torch.equal(new.module.head.weight[:, 4:], fresh.module.head.weight[:, 4:])
    assert new.manifest["weights"]["tensors"]["body.weight"]["origin"] == "random-init"
    assert new.manifest["weights"]["tensors"]["head.bias"]["origin"] == "imported"
    with pytest.raises(ReuseError, match="strict reuse failed") as error:
        build(tmp_path / "strict", text, reuse_source=old.path, reuse_policy="overlap", reuse_strict=True)
    assert error.value.reuse_report["status"] == "failed"
    assert not (tmp_path / "strict" / "weights" / "wide").exists()


def test_rename_mapping_and_removed_source_tensors(tmp_path):
    old = build(tmp_path / "a", program(extra="layer removed: linear(in=1, out=1)\n"))
    text = program("renamed", body="features")
    new = build(tmp_path / "b", text, reuse_source=old.path, reuse_mapping={"features": "body"}, reuse_strict=True)
    assert new.manifest["reuse"]["status"] == "complete"
    assert new.manifest["reuse"]["unused_source_tensors"] == ["removed.bias", "removed.weight"]
    assert torch.equal(new.module.features.weight, old.module.body.weight)
    with pytest.raises(WgtError, match="matches no tensors"):
        build(tmp_path / "typo", text, reuse_source=old.path, reuse_mapping={"featurez": "body"})


def test_attention_head_change_is_not_a_compatible_shape(tmp_path):
    template = (
        "wgt 0.3\nmodel attention\ninput x: f32[2,4]\noutput y: f32[2,4]\n"
        "layer attention: self_attention(dim=4, heads={heads})\ninit seed=42\n"
    )
    old = build(tmp_path / "a", template.format(heads=1))
    new = build(tmp_path / "b", template.format(heads=2), reuse_source=old.path, reuse_policy="overlap")
    report = new.manifest["reuse"]
    assert report["status"] == "no_reuse"
    assert all(row["reason"] == "operator_configuration_changed" for row in report["tensors"])
    assert report["summary"]["copied_values"] == 0


def test_preview_and_strict_do_not_train(tmp_path, monkeypatch):
    from weightlang import compiler

    old = build(tmp_path / "a", program())
    source = tmp_path / "next.wgt"
    source.write_text(
        program(
            "next", width=6, train="data d:\n  [1,2] -> [1]\ntrain data=d loss=mse optimizer=adam lr=0.01 epochs=1\n"
        )
    )

    def fail(*args, **kwargs):
        raise AssertionError("training must not execute")

    monkeypatch.setattr(compiler, "_train", fail)
    report = preview(source, old.path)
    assert report["status"] == "partial"
    with pytest.raises(ReuseError):
        compile_file(source, tmp_path / "out", reuse_source=old.path, reuse_strict=True)
    assert not (tmp_path / "out").exists()


def test_explicit_imports_override_auto_and_frozen_weights_stay_equal(tmp_path):
    old = build(tmp_path / "a", program())
    other = build(tmp_path / "other", program().replace("seed=42", "seed=9"))
    training = (
        f'import body from "{other.path}" layer=body freeze\n'
        "data d:\n  [1,2] -> [1]\ntrain data=d loss=mse optimizer=adamw lr=0.01 epochs=2 weight_decay=0.1\n"
    )
    new = build(tmp_path / "b", program("next", train=training), reuse_source=old.path)
    report = new.manifest["reuse"]
    assert report["summary"]["explicit_import_values"] == 12
    assert report["summary"]["copied_values"] == 5
    assert torch.equal(new.module.body.weight, other.module.body.weight)
    assert new.manifest["weights"]["tensors"]["head.weight"]["origin"] == "imported+trained"
    with pytest.raises(WgtError, match="conflicts with an explicit"):
        build(tmp_path / "bad", program("next", train=training), reuse_source=old.path, reuse_mapping={"body": "body"})


def test_cli_automatic_head_and_failure_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "model.wgt"
    source.write_text(program())
    assert main(["build", str(source)]) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert baseline["reuse"]["status"] == "no_baseline"
    source.write_text(program("next", width=6))
    assert main(["reuse-plan", str(source), "--reuse-report", "plan.json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    repo = WeightRepository("weights/.revisions")
    assert repo.resolve() == baseline["revision"]
    assert plan["summary"]["copied_values"] == 1
    assert main(["build", str(source), "--reuse-strict", "--reuse-report", "failure.json"]) == 1
    assert json.loads((tmp_path / "failure.json").read_text())["status"] == "failed"
    assert "strict_requires_exact_reuse" in capsys.readouterr().err
    assert repo.resolve() == baseline["revision"]
    assert main(["build", str(source), "--reuse", "overlap"]) == 0
    candidate = json.loads(capsys.readouterr().out)
    assert candidate["reuse"]["source"]["revision"] == baseline["revision"]
    assert candidate["parent"] == baseline["revision"]
    assert candidate["reuse"]["summary"]["copied_values"] == 17
    assert not list(repo.root.glob(".reuse-*")) and not list(repo.root.glob(".build-*"))


def test_no_baseline_disabled_and_corruption(tmp_path):
    text = program()
    with pytest.raises(ReuseError, match="strict reuse failed"):
        build(tmp_path / "strict", text, reuse_policy="exact", reuse_strict=True)
    disabled = build(tmp_path / "disabled", text, reuse_policy="none")
    assert disabled.manifest["reuse"]["status"] == "disabled"
    old = build(tmp_path / "a", text)
    (old.path / "weights.safetensors").write_bytes(b"bad")
    with pytest.raises(ReuseError) as error:
        build(tmp_path / "bad", text, reuse_source=old.path)
    assert error.value.reuse_report["failure"] == "baseline_validation_failed"


def test_preview_is_planned_and_build_is_applied(tmp_path):
    old = build(tmp_path / "old", program())
    source = tmp_path / "model.wgt"
    source.write_text(program("next"))
    assert preview(source, old.path)["execution"] == "planned"
    new = build(tmp_path / "new", program("next"), reuse_source=old.path)
    assert new.manifest["reuse"]["execution"] == "applied"


def test_missing_baseline_ref_writes_failure_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "model.wgt"
    source.write_text(program())
    assert main(["build", str(source)]) == 0
    capsys.readouterr()
    repository = WeightRepository("weights/.revisions")
    before = repository.status()["revision"]
    assert main(["build", str(source), "--from", "missing-branch", "--reuse-report", "failure.json"]) != 0
    report = json.loads((tmp_path / "failure.json").read_text())
    assert report["failure"] == "baseline_validation_failed"
    assert repository.status()["revision"] == before


def test_python_strict_without_source_is_not_silently_ignored(tmp_path):
    with pytest.raises(ReuseError, match="strict reuse failed"):
        build(tmp_path, program(), reuse_strict=True)
