"""Package publish/get/build flow against an in-memory fake of GitHub (no network)."""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from weightlang import WgtError, load_artifact
from weightlang.compiler import compile_file
from weightlang.packages import (
    INDEX_ASSET,
    ModulePath,
    build_package,
    get,
    parse_mod,
    read_sum,
    resolve_module_import,
)

ROOT = Path(__file__).resolve().parent.parent
MODULE = "github.com/WeightLang/WeightLang/packages/gates"
COMMIT = "a" * 40
XNOR = [[0, 0], [0, 1], [1, 0], [1, 1]]


class FakeGitHub:
    def __init__(self):
        self.tags: dict[tuple[str, str], str] = {}
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []

    def publish(self, dist: Path, module: str, version: str, commit: str, pkg_dir: Path):
        m = ModulePath.parse(module)
        self.tags[(m.repo_url, m.tag(version))] = commit
        for f in dist.iterdir():
            self.files[m.release_asset_url(version, f.name)] = f.read_bytes()
        for src in pkg_dir.glob("*.wgt"):
            self.files[m.raw_url(commit, src.name)] = src.read_bytes()

    def resolve_tag(self, repo_url, tag):
        try:
            return self.tags[(repo_url, tag)]
        except KeyError:
            raise WgtError(f"tag {tag} not found in {repo_url}") from None

    def download(self, url, max_bytes):
        self.requests.append(url)
        if url not in self.files:
            raise WgtError(f"download failed: {url}: 404")
        data = self.files[url]
        if len(data) > max_bytes:
            raise WgtError("too large")
        return data


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTLANG_CACHE", str(tmp_path / "cache"))
    pkg = tmp_path / "gates"
    shutil.copytree(ROOT / "packages" / "gates", pkg)
    dist = tmp_path / "dist"
    build_package(pkg, "v0.1.0", COMMIT, dist)
    gh = FakeGitHub()
    gh.publish(dist, MODULE, "v0.1.0", COMMIT, pkg)
    consumer = tmp_path / "consumer"
    shutil.copytree(ROOT / "examples" / "remote_reuse", consumer)
    (consumer / "wgt.sum").unlink(missing_ok=True)
    return gh, consumer, dist, pkg


def test_mod_and_module_paths():
    mod = parse_mod(ROOT / "examples" / "remote_reuse" / "wgt.mod")
    assert mod.requires == {MODULE: "v0.1.0"}
    m = ModulePath.parse(MODULE)
    assert m.tag("v0.1.0") == "packages/gates/v0.1.0"
    assert m.release_asset_url("v0.1.0", "x") == (
        "https://github.com/WeightLang/WeightLang/releases/download/packages/gates/v0.1.0/x"
    )
    with pytest.raises(WgtError):
        ModulePath.parse("gitlab.com/a/b")


def test_index_lists_hashes_and_commit(env):
    _, _, dist, _ = env
    index = json.loads((dist / INDEX_ASSET).read_text())
    assert index["module"] == MODULE and index["commit"] == COMMIT and index["tag"] == "packages/gates/v0.1.0"
    files = index["artifacts"]["xor_mlp"]["files"]
    assert set(files) == {"manifest.json", "weights.safetensors", "source.wgt"}
    assert index["artifacts"]["xor_mlp"]["untrained_tensors"] == []


def test_publish_refuses_untrained(tmp_path):
    pkg = tmp_path / "p"
    pkg.mkdir()
    (pkg / "wgt.mod").write_text("module github.com/o/r\nwgt 0.1\n")
    shutil.copy(ROOT / "examples" / "untrained.wgt", pkg)
    with pytest.raises(WgtError, match="untrained tensors"):
        build_package(pkg, "v0.1.0", COMMIT, tmp_path / "d")
    build_package(pkg, "v0.1.0", COMMIT, tmp_path / "d2", allow_untrained=True)


def test_get_then_offline_build_reuses_published_weights(env, tmp_path):
    gh, consumer, dist, _ = env
    assert get(consumer, gh) == [f"{MODULE}@v0.1.0"]
    sums = read_sum(consumer / "wgt.sum")
    assert sums[(MODULE, "v0.1.0", "commit")] == COMMIT
    assert sums[(MODULE, "v0.1.0", "xor_mlp/weights.safetensors")].startswith("sha256:")

    gh.files.clear()  # build must not need the network
    res = compile_file(consumer / "xnor.wgt", tmp_path / "out")
    imp = res.manifest["imports"][0]
    assert imp["package"] == {"module": MODULE, "version": "v0.1.0", "commit": COMMIT}
    assert res.manifest["weights"]["tensors"]["hidden.weight"]["upstream_origin"] == "trained"

    published = json.loads((dist / INDEX_ASSET).read_text())["artifacts"]["xor_mlp"]["files"]
    assert imp["source_weights_sha256"] == published["weights.safetensors"]["sha256"]
    from safetensors.torch import load_file

    src = load_file(str(dist / "xor_mlp.weights.safetensors"))
    dst = load_artifact(res.path).module.state_dict()
    assert torch.equal(dst["hidden.weight"], src["hidden.weight"])
    assert (load_artifact(res.path).predict(XNOR) >= 0.5).int().flatten().tolist() == [1, 0, 0, 1]


def test_build_without_get_explains(env, tmp_path):
    _, consumer, _, _ = env
    with pytest.raises(WgtError, match="run: weightc get"):
        compile_file(consumer / "xnor.wgt", tmp_path / "out")


def test_tampered_asset_rejected(env):
    gh, consumer, _, _ = env
    url = ModulePath.parse(MODULE).release_asset_url("v0.1.0", "xor_mlp.weights.safetensors")
    gh.files[url] = gh.files[url][:-1] + bytes([gh.files[url][-1] ^ 1])
    with pytest.raises(WgtError, match="sha256 mismatch"):
        get(consumer, gh)
    assert not (consumer / "wgt.sum").exists()


def test_assets_must_match_tagged_source(env):
    gh, consumer, _, _ = env
    url = ModulePath.parse(MODULE).raw_url(COMMIT, "xor.wgt")
    gh.files[url] += b"# changed after build\n"
    with pytest.raises(WgtError, match="xor.wgt at"):
        get(consumer, gh)


def test_index_commit_must_match_tag(env):
    gh, consumer, _, _ = env
    gh.tags[(ModulePath.parse(MODULE).repo_url, "packages/gates/v0.1.0")] = "b" * 40
    with pytest.raises(WgtError, match="tag points to"):
        get(consumer, gh)


def test_moved_tag_detected_by_wgt_sum(env):
    gh, consumer, _, _ = env
    get(consumer, gh)
    gh.tags[(ModulePath.parse(MODULE).repo_url, "packages/gates/v0.1.0")] = "c" * 40
    with pytest.raises(WgtError, match="published version changed"):
        get(consumer, gh)


def test_tampered_cache_rejected_at_build(env, tmp_path):
    gh, consumer, _, _ = env
    get(consumer, gh)
    cached = resolve_module_import(f"{MODULE}/xor_mlp", consumer / "xnor.wgt").path / "source.wgt"
    cached.write_text(cached.read_text() + "\n")
    with pytest.raises(WgtError, match="does not match wgt.sum"):
        compile_file(consumer / "xnor.wgt", tmp_path / "out")


def test_unknown_module_and_artifact(env, tmp_path):
    gh, consumer, _, _ = env
    get(consumer, gh)
    with pytest.raises(WgtError, match="not provided by any require"):
        resolve_module_import("github.com/x/y/z", consumer / "xnor.wgt")
    with pytest.raises(WgtError, match="no artifact 'nope'"):
        resolve_module_import(f"{MODULE}/nope", consumer / "xnor.wgt")


@pytest.mark.skipif(not os.environ.get("WEIGHTLANG_NETWORK_TESTS"), reason="set WEIGHTLANG_NETWORK_TESTS=1")
def test_real_github_release(tmp_path, monkeypatch):
    """Fetch the actually published gates@v0.1.0 from GitHub and check it against the committed wgt.sum."""
    monkeypatch.setenv("WEIGHTLANG_CACHE", str(tmp_path / "cache"))
    consumer = tmp_path / "consumer"
    shutil.copytree(ROOT / "examples" / "remote_reuse", consumer)
    committed = read_sum(consumer / "wgt.sum")
    assert committed, "examples/remote_reuse/wgt.sum must be committed"
    get(consumer)
    assert read_sum(consumer / "wgt.sum") == committed
    res = compile_file(consumer / "xnor.wgt", tmp_path / "out")
    assert (load_artifact(res.path).predict(XNOR) >= 0.5).int().flatten().tolist() == [1, 0, 0, 1]
