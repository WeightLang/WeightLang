"""Construction-only builds preserve migration and never execute training code."""

import pytest
import torch

from weightlang import compile_file, load_artifact, parse_file, WgtError
from weightlang import compiler
from weightlang.cli import main
from weightlang.repository import WeightRepository


def model(width=4, extra="", data=None):
    if data is None:
        data = "data samples:\n  [0,1] -> [1]\n  [1,0] -> [0]\n"
    return (
        "wgt 0.3\nmodel demo\ninput x: f32[2]\noutput y: f32[1]\n"
        f"layer body: linear(in=2,out={width})\nlayer act: tanh\n"
        f"layer head: linear(in={width},out=1)\n{extra}"
        f"init seed=42\n{data}"
        "train data=samples loss=mse optimizer=adam lr=0.01 epochs=3\n"
    )


def forbid_training(*args, **kwargs):
    raise AssertionError("training must not execute")


def test_no_train_matches_initialization_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "model.wgt"
    source.write_text(model())
    expected_source = source.read_bytes()
    trained = compile_file(source, tmp_path / "trained")
    monkeypatch.setattr(compiler, "_train", forbid_training)
    result = compile_file(source, tmp_path / "untrained", no_train=True)
    program = parse_file(source)
    from weightlang.artifact import build_module

    with torch.random.fork_rng():
        torch.manual_seed(program.seed)
        initial = build_module([layer.to_dict() for layer in program.layers])
    assert any(not torch.equal(a, b) for a, b in zip(result.module.parameters(), trained.module.parameters()))
    for name, value in result.module.state_dict().items():
        assert torch.equal(value, initial.state_dict()[name])
        assert result.manifest["weights"]["tensors"][name]["origin"] == "random-init"
    assert result.manifest["training"] is None
    assert result.manifest["build"] == {"no_train": True, "training_declared": True}
    assert (result.path / "source.wgt").read_bytes() == expected_source
    assert load_artifact(result.path, warn_untrained=False).module is not None


def test_no_data_or_trainer_plugin_loaded(tmp_path, monkeypatch, capsys):
    source = tmp_path / "model.wgt"
    data = 'data samples: file="missing.csv" sha256=' + "0" * 64 + "\n"
    source.write_text(model(data=data).replace("optimizer=adam", "trainer=custom"))
    plugin = tmp_path / "plugin.py"
    marker = tmp_path / "executed"
    plugin.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setattr(compiler, "_train", forbid_training)
    assert (
        main(
            [
                "build",
                str(source),
                "--no-train",
                "--no-store",
                "-o",
                str(tmp_path / "weights"),
                "--trainer",
                f"custom={plugin}:train",
            ]
        )
        == 0
    )
    assert "training skipped" in capsys.readouterr().out
    assert not marker.exists()
    with pytest.raises(WgtError, match="cannot read data file"):
        parse_file(source)
    # A dry-run uses the same construction-only checks and writes no model.
    assert main(["build", str(source), "--no-train", "--dry-run", "--no-store", "-o", str(tmp_path / "preview")]) == 0
    assert not (tmp_path / "preview").exists()


def test_build_only_can_freeze_everything_and_ignores_stale_targets(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(model(extra="freeze all\n", data="data samples:\n  [9] -> [8,7,6]\n"))
    result = compile_file(source, tmp_path / "weights", no_train=True)
    assert not any(p.requires_grad for p in result.module.parameters())
    assert set(result.manifest["training_policy"]["frozen"]) == set(result.module.state_dict())
    restored = load_artifact(result.path, warn_untrained=False)
    assert not any(p.requires_grad for p in restored.module.parameters())


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("out=4)", "out=5)", "expects"),
        ("layer act: tanh", "freeze absent\nlayer act: tanh", "matches no parameters"),
    ],
)
def test_structural_errors_still_fail(tmp_path, old, new, message):
    source = tmp_path / "model.wgt"
    source.write_text(model().replace(old, new))
    with pytest.raises(WgtError, match=message):
        compile_file(source, tmp_path / "weights", no_train=True)
    assert not (tmp_path / "weights").exists()


def test_head_migration_without_updates_and_overlap(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "model.wgt"
    source.write_text(model())
    assert main(["build", str(source)]) == 0
    repository = WeightRepository("weights/.revisions")
    original = repository.status()["revision"]
    repository.checkout(original, tmp_path / "baseline")
    baseline = load_artifact(tmp_path / "baseline", warn_untrained=False)
    monkeypatch.setattr(compiler, "_train", forbid_training)
    source.write_text(model(extra="layer calibration: linear(in=1,out=1)\n"))
    assert main(["build", str(source), "--no-train"]) == 0
    assert '"training_skipped": true' in capsys.readouterr().out
    migrated = repository.status()["revision"]
    assert migrated != original
    repository.checkout(migrated, tmp_path / "migrated")
    result = load_artifact(tmp_path / "migrated", warn_untrained=False)
    assert result.manifest["reuse"]["architecture_changed"]
    assert result.manifest["training"] is None
    for name, value in baseline.module.state_dict().items():
        assert torch.equal(value, result.module.state_dict()[name])
        assert result.manifest["weights"]["tensors"][name]["upstream_origin"] == "trained"
    assert result.manifest["weights"]["tensors"]["calibration.weight"]["origin"] == "random-init"
    # Widening uses explicitly requested leading-coordinate migration.
    source.write_text(model(width=6))
    assert main(["build", str(source), "--no-train", "--reuse", "overlap"]) == 0
    repository.checkout("HEAD", tmp_path / "wide")
    wide = load_artifact(tmp_path / "wide", warn_untrained=False)
    assert torch.equal(wide.module.body.weight[:4], baseline.module.body.weight)
    assert torch.equal(wide.module.head.weight[:, :4], baseline.module.head.weight)
    assert wide.manifest["reuse"]["summary"]["initialized_values"] == 8
    assert wide.manifest["training"] is None
    # Older source weights remain intact; incompatible strict migration cannot advance HEAD.
    assert torch.equal(
        load_artifact(tmp_path / "baseline", warn_untrained=False).module.body.weight, baseline.module.body.weight
    )
    before = repository.status()["revision"]
    source.write_text(model(width=7))
    assert main(["build", str(source), "--no-train", "--reuse-strict"]) == 1
    assert repository.status()["revision"] == before


def test_explicit_import_and_no_overwrite(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(model())
    baseline = compile_file(source, tmp_path / "baseline")
    source.write_text(model(extra=f'import body from "{baseline.path}" layer=body freeze\n'))
    target = compile_file(source, tmp_path / "target", no_train=True)
    assert torch.equal(target.module.body.weight, baseline.module.body.weight)
    assert not target.module.body.weight.requires_grad
    assert target.manifest["weights"]["tensors"]["body.weight"]["origin"] == "imported"
    assert target.manifest["training"] is None
    with pytest.raises(WgtError, match="overwrite"):
        compile_file(source, tmp_path / "target", no_train=True)


def test_compile_program_does_not_mutate_training_declaration(tmp_path, monkeypatch):
    source = tmp_path / "model.wgt"
    source.write_text(model())
    program = parse_file(source)
    declaration = program.train
    monkeypatch.setattr(compiler, "_train", forbid_training)
    result = compiler.compile_program(program, tmp_path / "weights", no_train=True)
    assert result.manifest["training"] is None
    assert program.train is declaration


def test_derived_no_train_skips_external_data(tmp_path, monkeypatch):
    from weightlang.artifact import sha256_file

    source = tmp_path / "model.wgt"
    source.write_text(model())
    baseline = compile_file(source, tmp_path / "baseline")
    derived = tmp_path / "child.wgt"
    digest = sha256_file(baseline.path / "manifest.json")
    derived.write_text(
        f'wgt 0.3\nmodel child extends "{baseline.path}" sha256="{digest}":\n'
        "  freeze all\n"
        '  data samples: file="missing.csv" sha256=' + "0" * 64 + "\n"
        "  train data=samples loss=mse optimizer=adam lr=0.01 epochs=2\n"
    )
    monkeypatch.setattr(compiler, "_train", forbid_training)
    result = compile_file(derived, tmp_path / "result", no_train=True)
    assert result.manifest["training"] is None
    assert not any(p.requires_grad for p in result.module.parameters())
    for name, value in baseline.module.state_dict().items():
        assert torch.equal(value, result.module.state_dict()[name])
    assert main(["build", str(derived), "--no-train", "--dry-run", "--no-store"]) == 0
