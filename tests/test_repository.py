import json
from pathlib import Path

import pytest

from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.repository import WeightRepository
from weightlang.revisions import RevisionStore
from weightlang.syntax import WgtError


def model_source(seed):
    return f"wgt 0.3\nmodel demo\ninput x: f32[2]\noutput y: f32[1]\nlayer fc: linear(in=2, out=1)\ninit seed={seed}\n"


@pytest.fixture
def models(tmp_path):
    result = []
    for seed in range(3):
        source = tmp_path / f"{seed}.wgt"
        source.write_text(model_source(seed))
        result.append(compile_file(source, tmp_path / str(seed)).path)
    return result


def test_auto_parent_messages_and_noop(tmp_path, models):
    repo = WeightRepository(tmp_path / "repo")
    assert not repo.status()["initialized"]
    a = repo.commit(models[0], message="基线")
    b = repo.commit(models[1], message="iteration")
    assert a["parent"] is None and b["parent"] == a["revision"]
    assert b["branch"] == "main" and repo.resolve() == b["revision"]
    assert [entry["message"] for entry in repo.log()] == ["iteration", "基线"]
    before = {str(p.relative_to(repo.root)): p.read_bytes() for p in repo.root.rglob("*") if p.is_file()}
    repeated = repo.commit(models[1], message="no artificial empty commits")
    assert repeated["unchanged"] and repeated["new_bytes"] == 0
    assert repeated["revision"] == b["revision"]
    assert before == {str(p.relative_to(repo.root)): p.read_bytes() for p in repo.root.rglob("*") if p.is_file()}
    assert repo.resolve("HEAD~1") == a["revision"]
    assert repo.resolve(a["revision"][:8]) == a["revision"]
    assert repo.diff("HEAD~1")["changed_files"]


def test_branch_divergence_detached_and_export(tmp_path, models):
    repo = WeightRepository(tmp_path / "repo")
    a = repo.commit(models[0])
    blobs = sorted((repo.root / "objects").iterdir())
    repo.branch("experiment/head", switch=True)
    assert sorted((repo.root / "objects").iterdir()) == blobs
    b = repo.commit(models[1])
    repo.switch("main")
    assert repo.resolve() == a["revision"]
    c = repo.commit(models[2])
    assert c["parent"] == b["parent"] == a["revision"]
    repo.switch(b["revision"], detach=True)
    assert repo.status()["branch"] is None
    detached = repo.commit(models[2])
    assert detached["parent"] == b["revision"]
    assert repo.status()["branches"]["main"] == c["revision"]
    repo.branch("rescue", switch=True)
    assert repo.resolve("rescue") == detached["revision"]
    repo.store.checkout(repo.resolve("experiment/head"), tmp_path / "out")
    for name in ("weights.safetensors", "manifest.json", "source.wgt"):
        assert (tmp_path / "out" / name).read_bytes() == (models[1] / name).read_bytes()


def test_status_dirty_artifact_failure_preserves_head(tmp_path, models):
    repo = WeightRepository(tmp_path / "repo")
    before = repo.commit(models[0])["revision"]
    assert repo.status(models[0])["changed_files"] == []
    (models[0] / "source.wgt").write_text("changed")
    assert repo.status(models[0])["changed_files"] == ["source.wgt"]
    with pytest.raises(WgtError, match="source sha256 mismatch"):
        repo.commit(models[0])
    assert repo.resolve() == before
    assert not (repo.root / "refs.lock").exists()


def test_lock_and_conflicting_build_retains_result(tmp_path, models):
    repo = WeightRepository(tmp_path / "repo")
    a = repo.commit(models[0])
    expected = repo.prepare()
    b = repo.commit(models[1])
    with repo._lock():
        with pytest.raises(WgtError, match="locked"):
            repo.branch("busy")
    with pytest.raises(WgtError, match="Result preserved as revision") as error:
        repo.commit(models[2], message="long training", expected=expected)
    saved = str(error.value).split("Result preserved as revision ")[1].split(";")[0]
    assert repo.resolve() == b["revision"]
    assert repo.store.read(saved)["parent"] == a["revision"]
    repo.store.verify(saved)
    repo.branch("recovered", start=saved)
    assert repo.resolve("recovered") == saved


def test_migrate_v1_without_guessing_tip(tmp_path, models):
    store = RevisionStore(tmp_path / "repo")
    a = store.save(models[0])
    store.save(models[2])
    repo = WeightRepository(store.root)
    with pytest.raises(WgtError, match="initialize once with --start"):
        repo.commit(models[1])
    repo.init(start=a["revision"])
    b = repo.commit(models[1], message="v2")
    assert b["parent"] == a["revision"]
    assert store.read(a["revision"])["format_version"] == 1
    assert store.read(b["revision"])["format_version"] == 2
    assert store.verify(b["revision"])["verified"]


@pytest.mark.parametrize("name", ["../escape", "HEAD", "a.lock", "/root", "", "a//b"])
def test_invalid_branch_names(tmp_path, name):
    repo = WeightRepository(tmp_path / "repo")
    repo.init()
    before = (repo.root / "refs.json").read_bytes()
    with pytest.raises(WgtError, match="invalid branch"):
        repo.branch(name)
    assert (repo.root / "refs.json").read_bytes() == before


def test_corrupt_refs_and_failed_write(tmp_path, models, monkeypatch):
    repo = WeightRepository(tmp_path / "repo")
    a = repo.commit(models[0])
    before = (repo.root / "refs.json").read_bytes()

    def fail(state):
        raise OSError("disk write failure")

    monkeypatch.setattr(repo, "_write", fail)
    with pytest.raises(OSError, match="disk write failure"):
        repo.commit(models[1])
    assert (repo.root / "refs.json").read_bytes() == before
    assert repo.resolve() == a["revision"]
    (repo.root / "refs.json").write_text('{"format": "bad"}')
    with pytest.raises(WgtError, match="invalid refs"):
        repo.status()


def test_cli_auto_build_history_and_switch(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = Path("model.wgt")
    source.write_text(model_source(0))
    assert main(["build", str(source), "-m", "first"]) == 0
    a = json.loads(capsys.readouterr().out)
    source.write_text(model_source(1))
    assert main(["build", str(source), "-m", "second"]) == 0
    b = json.loads(capsys.readouterr().out)
    assert b["parent"] == a["revision"]
    assert main(["revision", "log"]) == 0
    assert [r["message"] for r in json.loads(capsys.readouterr().out)] == ["second", "first"]
    assert main(["revision", "switch", "-c", "experiment"]) == 0
    assert json.loads(capsys.readouterr().out)["branch"] == "experiment"
    assert main(["revision", "checkout", "HEAD~1", "-o", "export"]) == 0
    capsys.readouterr()
    assert main(["revision", "commit", "export", "-m", "restore older weights"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["parent"] == b["revision"]
    assert main(["revision", "status", "export"]) == 0
    assert json.loads(capsys.readouterr().out)["changed_files"] == []
    assert not list(Path("weights/.revisions").glob(".build-*"))


def test_adopt_single_history_tip_and_stats(tmp_path, models):
    store = RevisionStore(tmp_path / "repo")
    a = store.save(models[0])
    b = store.save(models[1], parent=a["revision"])
    repo = WeightRepository(store.root)
    repo.init()
    assert repo.resolve() == b["revision"]
    stats = repo.stats()
    assert stats["revision_count"] == 2
    assert stats["stored_payload_bytes"] == stats["object_bytes"] + stats["revision_bytes"] + stats["refs_bytes"]
    assert stats["saved_payload_bytes"] == stats["full_snapshot_bytes"] - stats["stored_payload_bytes"]


def test_unborn_branches_and_parent_mismatch(tmp_path, models):
    repo = WeightRepository(tmp_path / "repo")
    repo.init()
    assert repo.log() == []
    repo.branch("other", switch=True)
    a = repo.commit(models[0])
    repo.switch("main")
    b = repo.commit(models[1])
    assert a["parent"] is b["parent"] is None
    with pytest.raises(WgtError, match="differs from HEAD"):
        repo.commit(models[2], parent=a["revision"])
    assert repo.resolve() == b["revision"]
    with pytest.raises(WgtError, match="no such ancestor"):
        repo.resolve("HEAD~1")
    with pytest.raises(WgtError, match="missing or ambiguous"):
        repo.resolve("0" * 8)


def test_build_conflict_exports_recoverable_result(tmp_path, models, monkeypatch, capsys):
    from weightlang import compiler

    root = tmp_path / "repo"
    repo = WeightRepository(root)
    a = repo.commit(models[0])
    original = compiler.compile_file

    def concurrent_build(*args, **kwargs):
        result = original(*args, **kwargs)
        repo.commit(models[1])
        return result

    monkeypatch.setattr(compiler, "compile_file", concurrent_build)
    source = tmp_path / "next.wgt"
    source.write_text(model_source(2))
    assert main(["build", str(source), "--store", str(root)]) == 1
    error = capsys.readouterr().err
    saved = error.split("Result preserved as revision ")[1].split(";")[0]
    assert repo.store.read(saved)["parent"] == a["revision"]
    assert repo.resolve() != saved
    assert repo.store.verify(saved)["verified"]
    assert not list(root.glob(".build-*"))


def test_explicit_no_store_and_flag_conflicts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("model.wgt").write_text(model_source(0))
    assert main(["build", "model.wgt", "--no-store"]) == 0
    assert Path("weights/demo/weights.safetensors").is_file()
    assert not Path("weights/.revisions").exists()
    assert main(["build", "model.wgt", "--no-store", "-m", "unused"]) == 1
    assert main(["build", "model.wgt", "--no-store", "--parent", "0" * 64]) == 1
    assert not Path("weights/.revisions").exists()
    with pytest.raises(SystemExit) as error:
        main(["build", "model.wgt", "--store", "--no-store"])
    assert error.value.code == 2
    assert main(["build", "model.wgt", "-o", "custom"]) == 1
    assert not Path("custom").exists()
    assert main(["build", "model.wgt", "-m", "tracked"]) == 0
    assert Path("weights/.revisions/refs.json").is_file()
    # The default build must not overwrite an independently retained model.
    assert Path("weights/demo/weights.safetensors").is_file()
