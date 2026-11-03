import json

import pytest
import torch

from weightlang import WgtError, compile_file, load_artifact, save_artifact


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text("wgt 0.2\nmodel demo\ninput x: f32[1]\noutput y: f32[1]\n"
                      "layer fc: linear(in=1, out=1)\ninit seed=42\n")
    built = compile_file(source, tmp_path / "weights")
    return load_artifact(built.path, warn_untrained=False)


def test_training_roundtrip(artifact, tmp_path):
    original = (artifact.path / "weights.safetensors").read_bytes()
    model = artifact.module.train()
    x = torch.tensor([[-1.], [0.], [1.]])
    target = 2 * x + 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    saved = artifact.save(tmp_path / "fine_tuned", training={"method": "SGD", "steps": 100})
    assert torch.allclose(saved.predict(x), target, atol=1e-4)
    assert torch.equal(saved.predict(x), model(x))
    assert all(m["origin"] == "trained" for m in saved.manifest["weights"]["tensors"].values())
    assert saved.manifest["training"]["provenance"] == "caller-reported"
    assert saved.manifest["python_history"][0]["baseline_weights_sha256"] == artifact.manifest["weights"]["sha256"]
    assert (artifact.path / "weights.safetensors").read_bytes() == original
    assert model.training  # Saving does not change the caller's mode.


def test_edit_and_noop_provenance(artifact, tmp_path):
    with torch.no_grad():
        artifact.module.fc.weight.add_(1)
    saved = save_artifact(artifact, tmp_path / "edited")
    assert saved.manifest["training"] is None
    assert saved.manifest["weights"]["tensors"]["fc.weight"]["origin"] == "edited"
    assert saved.manifest["weights"]["tensors"]["fc.bias"]["origin"] == "random-init"
    again = saved.save(tmp_path / "unchanged", training={"method": "no updates"})
    assert again.manifest["weights"]["tensors"] == saved.manifest["weights"]["tensors"]
    assert again.manifest["weights"]["sha256"] == saved.manifest["weights"]["sha256"]


@pytest.mark.parametrize("mutation, message", [
    (lambda m: m.fc.weight.data.fill_(float("nan")), "non-finite"),
    (lambda m: m.double(), "dtype"),
    (lambda m: setattr(m, "fc", torch.nn.ReLU()), "structure"),
    (lambda m: setattr(m.fc, "forward", lambda x: x), "forward"),
    (lambda m: m.fc.register_forward_hook(lambda m, i, o: o), "hooks"),
])
def test_reject_invalid_updates(artifact, tmp_path, mutation, message):
    mutation(artifact.module)
    with pytest.raises(WgtError, match=message):
        artifact.save(tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_protect_paths_and_stale_baseline(artifact, tmp_path):
    for path in (artifact.path, artifact.path / "nested"):
        with pytest.raises(WgtError):
            artifact.save(path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(WgtError):
        artifact.save(link)
    artifact.manifest["model"] = "changed"
    with pytest.raises(WgtError, match="manifest changed"):
        artifact.save(tmp_path / "stale")


@pytest.mark.parametrize("record", [{}, {"method": ""}, {"method": "SGD", "loss": float("nan")}, {"method": "SGD", "x": object()}])
def test_training_record_validation(artifact, tmp_path, record):
    with pytest.raises(WgtError):
        artifact.save(tmp_path / "invalid", training=record)


def test_freeze_restored_and_enforced(artifact, tmp_path):
    manifest = artifact.manifest
    manifest["training_policy"] = {"frozen": ["fc.weight"], "trainable": ["fc.bias"]}
    (artifact.path / "manifest.json").write_text(json.dumps(manifest))
    loaded = load_artifact(artifact.path, warn_untrained=False)
    assert not loaded.module.fc.weight.requires_grad
    assert loaded.module.fc.bias.requires_grad
    with torch.no_grad():
        loaded.module.fc.weight.add_(1)
    with pytest.raises(WgtError, match="frozen parameter"):
        loaded.save(tmp_path / "invalid", training={"method": "SGD"})


def test_failed_write_cleans_only_new_output(artifact, tmp_path, monkeypatch):
    import weightlang.interop as interop

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(interop, "write_safetensors", fail)
    with pytest.raises(OSError, match="disk failure"):
        artifact.save(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert artifact.path.is_dir()


def test_imported_training_preserves_lineage(artifact, tmp_path):
    source = tmp_path / "import.wgt"
    source.write_text(
        'wgt 0.2\nmodel imported\ninput x: f32[1]\noutput y: f32[1]\n'
        'layer fc: linear(in=1, out=1)\n'
        'import fc from "./weights/demo" layer=fc\ninit seed=1\n'
    )
    built = compile_file(source, tmp_path / "weights")
    imported = load_artifact(built.path, warn_untrained=False)
    with torch.no_grad():
        imported.module.fc.weight.add_(1)
    saved = imported.save(tmp_path / "trained_import", training={"method": "external"})
    meta = saved.manifest["weights"]["tensors"]
    assert meta["fc.weight"]["origin"] == "imported+trained"
    assert meta["fc.weight"]["upstream_origin"] == "random-init"
    assert meta["fc.bias"]["origin"] == "imported"


def test_stateless_configuration_change_rejected(artifact, tmp_path):
    artifact.module.fc.in_features = 100
    with pytest.raises(WgtError, match="structure"):
        artifact.save(tmp_path / "invalid")
