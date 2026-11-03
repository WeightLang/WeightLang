import json

import pytest
import torch

from weightlang.artifact import load_artifact
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.revisions import RevisionStore
from weightlang.syntax import WgtError, parse


def source(model="base", rules="", train=True, version="0.3"):
    return (
        f"wgt {version}\nmodel {model}\ninput x: f32[2]\noutput y: f32[1]\n"
        "layer body: linear(in=2, out=128)\nlayer head: linear(in=128, out=1)\ninit seed=7\n"
        + rules
        + (
            "data d:\n  [1, 2] -> [1]\ntrain data=d loss=mse optimizer=adamw lr=0.001 epochs=3 weight_decay=0.1\n"
            if train
            else ""
        )
    )


def build(tmp_path, text, filename="model.wgt"):
    path = tmp_path / filename
    path.write_text(text)
    return compile_file(path, tmp_path / "weights")


def test_freeze_exact_parameter_and_adamw(tmp_path):
    initial = build(tmp_path, source("base", train=False))
    trained = build(tmp_path, source("trained", "freeze all\nunfreeze head.bias\n"))
    before, after = initial.module.state_dict(), trained.module.state_dict()
    for name in before:
        assert torch.equal(before[name], after[name]) == (name != "head.bias")
    assert trained.manifest["training_policy"]["trainable"] == ["head.bias"]
    assert trained.manifest["weights"]["tensors"]["body.weight"]["origin"] == "random-init"


@pytest.mark.parametrize(
    "rules,match", [("freeze missing\n", "matches no parameters"), ("freeze all\n", "nothing to update")]
)
def test_invalid_freezing(rules, match):
    with pytest.raises(WgtError, match=match):
        parse(source(rules=rules))


def test_freezing_order_and_version(tmp_path):
    result = build(tmp_path, source(rules="freeze all\nunfreeze head\nfreeze head.weight\n"))
    assert result.manifest["training_policy"]["trainable"] == ["head.bias"]
    with pytest.raises(WgtError, match="requires 'wgt 0.3'"):
        parse(source(rules="freeze body\n", version="0.2"))
    with pytest.raises(WgtError, match="train statement"):
        parse(source(rules="freeze body\n", train=False))


def test_lossless_versions_sharing_and_independent_checkout(tmp_path):
    first = build(tmp_path, source("base", train=False))
    second = build(tmp_path, source("trained", "freeze all\nunfreeze head.bias\n"))
    store = RevisionStore(tmp_path / "store")
    a = store.save(first.path)
    b = store.save(second.path, parent=a["revision"])
    assert b["new_bytes"] < b["artifact_bytes"]
    assert store.save(second.path, parent=a["revision"])["new_bytes"] == 0
    record = store.read(b["revision"])
    assert record["parent"] == a["revision"]
    # Parent is history, not a replay dependency.
    (store.root / "revisions" / a["revision"]).unlink()
    out = tmp_path / "restored"
    store.checkout(b["revision"], out)
    for name in ("manifest.json", "source.wgt", "weights.safetensors"):
        assert (out / name).read_bytes() == (second.path / name).read_bytes()
    assert torch.equal(
        load_artifact(out, warn_untrained=False).predict([[1, 2]]), second.module(torch.tensor([[1.0, 2.0]]))
    )
    assert store.verify(b["revision"])["verified"]
    with pytest.raises(WgtError, match="overwrite"):
        store.checkout(b["revision"], out)


def test_corruption_missing_parent_and_cli(tmp_path, capsys):
    built = build(tmp_path, source(train=False))
    store = RevisionStore(tmp_path / "store")
    with pytest.raises(WgtError, match="missing/unreadable"):
        store.save(built.path, parent="0" * 64)
    assert main(["revision", "save", str(built.path), "--store", str(store.root)]) == 0
    revision = json.loads(capsys.readouterr().out)["revision"]
    assert main(["revision", "verify", revision, "--store", str(store.root)]) == 0
    rec = store.read(revision)
    obj = rec["files"]["weights.safetensors"]["objects"][-1]["sha256"]
    (store.root / "objects" / obj).write_bytes(b"corrupted")
    with pytest.raises(WgtError, match="sha256 mismatch"):
        store.checkout(revision, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(WgtError, match="64 lowercase"):
        store.read("../../other")


def test_inherit_and_direct_build_store(tmp_path, capsys):
    first = build(tmp_path, source(train=False))
    sha = first.manifest["weights"]["sha256"]
    code = source("next", f'inherit from "{first.path}" sha256={sha}\nfreeze all\nunfreeze head.bias\n')
    file = tmp_path / "next.wgt"
    file.write_text(code)
    store = tmp_path / "store"
    assert main(["build", str(file), "--store", str(store)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not list(store.glob(".build-*"))
    out = tmp_path / "export"
    RevisionStore(store).checkout(report["revision"], out)
    model = load_artifact(out, warn_untrained=False)
    assert torch.equal(model.module.state_dict()["body.weight"], first.module.state_dict()["body.weight"])
    assert model.manifest["weights"]["tensors"]["head.bias"]["origin"] == "imported+trained"
    assert model.manifest["weights"]["tensors"]["head.weight"]["origin"] == "imported"
    file.write_text(code.replace(sha, "0" * 64))
    with pytest.raises(WgtError, match="inherit pin mismatch"):
        compile_file(file, tmp_path / "bad")


def test_gradients_cross_a_frozen_layer(tmp_path):
    initial = build(tmp_path, source("initial", train=False))
    trained = build(tmp_path, source("trained", "freeze head\n"))
    assert torch.equal(initial.module.head.weight, trained.module.head.weight)
    assert not torch.equal(initial.module.body.weight, trained.module.body.weight)


def test_chunks_reuse_large_tensor_and_preserve_signed_zero(tmp_path):
    from weightlang.artifact import sha256_file, write_safetensors

    built = build(tmp_path, source(train=False).replace("128", "32768"))
    store = RevisionStore(tmp_path / "store")
    a = store.save(built.path)
    tensors = built.module.state_dict()
    tensors["body.weight"].view(-1)[20000] = -0.0
    file = built.path / "weights.safetensors"
    write_safetensors(file, tensors, {"format": "weightlang.artifact", "model": "base"})
    built.manifest["weights"]["sha256"] = sha256_file(file)
    (built.path / "manifest.json").write_text(json.dumps(built.manifest))
    b = store.save(built.path, parent=a["revision"])
    old = store.read(a["revision"])["files"]["weights.safetensors"]["objects"]
    new = store.read(b["revision"])["files"]["weights.safetensors"]["objects"]
    assert len(old) == len(new)
    assert sum(x != y for x, y in zip(old, new)) == 1
    out = tmp_path / "restored"
    store.checkout(b["revision"], out)
    assert (out / "weights.safetensors").read_bytes() == file.read_bytes()
    assert torch.signbit(load_artifact(out, warn_untrained=False).module.body.weight.view(-1)[20000])


@pytest.mark.parametrize("mutation", ["unknown-field", "bool-version", "wrong-size", "unknown-precision"])
def test_malformed_revision_rejected(tmp_path, mutation):
    from weightlang.revisions import _canonical

    built = build(tmp_path, source(train=False))
    store = RevisionStore(tmp_path / "store")
    record = store.read(store.save(built.path)["revision"])
    if mutation == "unknown-field":
        record["extra"] = 1
    elif mutation == "bool-version":
        record["format_version"] = True
    elif mutation == "wrong-size":
        record["files"]["manifest.json"]["size"] += 1
    else:
        record["precision"] = "int8"
    revision, _ = store._put("revisions", _canonical(record))
    with pytest.raises(WgtError, match="invalid revision"):
        store.read(revision)
