import json

import pytest

from weightlang.artifact import load_artifact, sha256_file
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.patches import apply_patch, patch_template
from weightlang.syntax import WgtError
from weightlang.weights import diff_weights, evaluate_artifact


@pytest.fixture
def baseline(tmp_path):
    src = tmp_path / "affine.wgt"
    src.write_text("""wgt 0.2
model affine
input x: f32[1]
output y: f32[1]
layer fc: linear(in=1, out=1)
init seed=42
data train:
  [0] -> [0]
  [1] -> [1]
train data=train loss=mse optimizer=adam lr=0.05 epochs=20
""")
    return compile_file(src, tmp_path / "build").path


def patch_file(baseline, tmp_path, update=None):
    patch = patch_template(baseline, "test author")
    patch["reason"] = "Implement y = 2*x + 1 directly, without training"
    patch["edits"] = [
        {"tensor": "fc.weight", "index": 0, "value": 2},
        {"tensor": "fc.bias", "index": 0, "value": 1},
    ]
    if update:
        update(patch)
    path = tmp_path / "patch.json"
    path.write_text(json.dumps(patch))
    return path


def test_complete_edit_compare_evaluate_and_audit(baseline, tmp_path, capsys):
    before = {p.name: sha256_file(p) for p in baseline.iterdir()}
    patch = patch_file(baseline, tmp_path)
    output = tmp_path / "candidate"
    result = apply_patch(baseline, patch, output)
    loaded = load_artifact(output, warn_untrained=False)
    assert loaded.predict([[2], [-2]]).tolist() == [[5], [-3]]
    assert result["changed_elements"] == 2
    assert main(["inspect", str(output)]) == 0
    assert "edit 1: test author (self-reported), 2 scalar changes" in capsys.readouterr().out
    assert set(loaded.untrained_tensors) == {"fc.weight", "fc.bias"}
    assert loaded.manifest["format_version"] == 3
    assert loaded.manifest["training"] is None
    history = loaded.manifest["edit_history"][0]
    assert history["previous_training"]["epochs"] == 20
    assert history["previous_tensor_metadata"]["fc.weight"]["origin"] == "trained"
    assert history["patch_sha256"] == sha256_file(patch)
    assert diff_weights(baseline, output)["changed_elements"] == 2
    data = tmp_path / "evaluation.json"
    data.write_text(json.dumps({"inputs": [[2], [-2]], "targets": [[5], [-3]]}))
    report = evaluate_artifact(output, data, "mse", compare=baseline)
    assert report["candidate"]["loss"] == 0
    assert report["delta"]["loss"] < 0
    assert main(["audit", str(output)]) == 1
    assert "fail: no untrained tensors" in capsys.readouterr().out
    assert {p.name: sha256_file(p) for p in baseline.iterdir()} == before
    again = tmp_path / "same"
    assert apply_patch(baseline, patch, again)["weights_sha256"] == result["weights_sha256"]
    assert (again / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda p: p.update(baseline_weights_sha256="0" * 64), "weights sha256"),
        (lambda p: p.update(baseline_manifest_sha256="0" * 64), "manifest sha256"),
        (lambda p: p.update(format_version=True), "format"),
        (lambda p: p.update(extra="typo"), "documented fields"),
        (lambda p: p.update(author=" "), "author"),
        (lambda p: p.update(edits=[]), "nonempty"),
        (lambda p: p["edits"][0].update(tensor="../escape"), "unknown tensor"),
        (lambda p: p["edits"][0].update(index=-1), "bounds"),
        (lambda p: p["edits"][0].update(index=1), "bounds"),
        (lambda p: p["edits"][0].update(index=True), "bounds"),
        (lambda p: p["edits"][0].update(value=True), "finite float32"),
        (lambda p: p["edits"][0].update(value=1e100), "finite float32"),
        (lambda p: p["edits"][0].update(value=float("nan")), "finite float32"),
        (lambda p: p["edits"][0].update(value=float("inf")), "finite float32"),
        (lambda p: p["edits"][0].update(expected=123), "expected value"),
        (lambda p: p["edits"].append(p["edits"][0]), "duplicate edit"),
    ],
)
def test_invalid_patch_has_no_output(baseline, tmp_path, update, message):
    patch = patch_file(baseline, tmp_path, update)
    output = tmp_path / "candidate"
    with pytest.raises(WgtError, match=message):
        apply_patch(baseline, patch, output)
    assert not output.exists()


def test_duplicate_keys_existing_output_noop_and_stale_manifest(baseline, tmp_path):
    patch = patch_file(baseline, tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(WgtError, match="already exists"):
        apply_patch(baseline, patch, output)
    assert list(output.iterdir()) == []
    raw = patch.read_text()
    patch.write_text(raw.replace('"index": 0', '"index": 0, "index": 1', 1))
    with pytest.raises(WgtError, match="duplicate JSON"):
        apply_patch(baseline, patch, tmp_path / "bad")
    model = load_artifact(baseline, warn_untrained=False)
    patch = patch_file(
        baseline,
        tmp_path,
        lambda p: p.update(edits=[{"tensor": "fc.weight", "index": 0, "value": model.module.fc.weight.item()}]),
    )
    with pytest.raises(WgtError, match="changes no"):
        apply_patch(baseline, patch, tmp_path / "noop")
    patch = patch_file(baseline, tmp_path)
    manifest = baseline / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(WgtError, match="manifest sha256"):
        apply_patch(baseline, patch, tmp_path / "stale")


def test_history_chain_and_imports_keep_edited_origin(baseline, tmp_path):
    patch = patch_file(baseline, tmp_path)
    first = tmp_path / "first"
    apply_patch(baseline, patch, first)
    second_patch = patch_file(first, tmp_path, lambda p: p["edits"][1].update(value=2, expected=1))
    second = tmp_path / "second"
    apply_patch(first, second_patch, second)
    loaded = load_artifact(second, warn_untrained=False)
    assert len(loaded.manifest["edit_history"]) == 2
    assert loaded.predict([2]).item() == 6
    for n in range(2):
        src = tmp_path / f"reuse{n}.wgt"
        src.write_text(f"""wgt 0.2
model reuse
input x: f32[1]
output y: f32[1]
layer fc: linear(in=1, out=1)
import fc from "{second}" layer=fc
init seed=42
""")
        second = compile_file(src, tmp_path / f"build{n}").path
        loaded = load_artifact(second, warn_untrained=False)
        assert set(loaded.untrained_tensors) == {"fc.weight", "fc.bias"}
        assert loaded.manifest["weights"]["tensors"]["fc.weight"]["upstream_origin"] == "edited"
        assert loaded.manifest["format_version"] == 3


def test_cli_and_failed_write_cleanup(baseline, tmp_path, capsys, monkeypatch):
    assert main(["patch-template", str(baseline), "--author", "tester"]) == 0
    assert json.loads(capsys.readouterr().out)["edits"] == []
    patch = patch_file(baseline, tmp_path)
    output = tmp_path / "candidate"
    assert main(["patch", str(baseline), "--patch", str(patch), "-o", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["changed_elements"] == 2
    assert main(["patch", str(baseline), "--patch", str(patch), "-o", str(output)]) == 1
    assert "already exists" in capsys.readouterr().err

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("weightlang.patches.write_safetensors", fail)
    with pytest.raises(WgtError, match="disk failure"):
        apply_patch(baseline, patch, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_output_cannot_mutate_baseline_directory(baseline, tmp_path):
    patch = patch_file(baseline, tmp_path)
    with pytest.raises(WgtError, match="outside the baseline"):
        apply_patch(baseline, patch, baseline / "nested")


def test_sparse_edit_retains_other_origins(baseline, tmp_path):
    patch = patch_file(baseline, tmp_path, lambda p: p.update(edits=p["edits"][:1]))
    candidate = tmp_path / "candidate"
    apply_patch(baseline, patch, candidate)
    loaded = load_artifact(candidate, warn_untrained=False)
    assert loaded.untrained_tensors == ["fc.weight"]
    assert loaded.manifest["weights"]["tensors"]["fc.bias"]["origin"] == "trained"


def test_inspect_reports_token_dtype(tmp_path, capsys):
    src = tmp_path / "tokens.wgt"
    src.write_text(
        "wgt 0.3\nmodel tokens\ninput x: i64[2]\noutput y: f32[2, 3]\n"
        "layer embed: embedding(vocab=4, dim=3)\ninit seed=42\n"
    )
    artifact = compile_file(src, tmp_path / "build").path
    assert main(["inspect", str(artifact)]) == 0
    assert "input  x: i64[2]" in capsys.readouterr().out
