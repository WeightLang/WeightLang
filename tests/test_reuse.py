import re

import pytest
import torch

from weightlang import WgtError, load_artifact
from weightlang.compiler import compile_file

XNOR = [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_frozen_import_reuses_exact_tensors(workspace, xor_build):
    res = compile_file(workspace / "examples" / "xnor_reuse.wgt", workspace / "weights")
    src = load_artifact(xor_build.path).module.state_dict()
    dst = load_artifact(res.path).module.state_dict()
    assert torch.equal(dst["hidden.weight"], src["hidden.weight"])
    assert torch.equal(dst["hidden.bias"], src["hidden.bias"])
    assert not torch.equal(dst["head.weight"], src["head.weight"])

    tensors = res.manifest["weights"]["tensors"]
    assert tensors["hidden.weight"]["origin"] == "imported"
    assert tensors["hidden.weight"]["upstream_origin"] == "trained"
    assert tensors["head.weight"]["origin"] == "trained"
    imp = res.manifest["imports"][0]
    assert imp["source_model"] == "xor_mlp" and imp["frozen"]
    assert imp["source_weights_sha256"] == xor_build.manifest["weights"]["sha256"]

    pred = load_artifact(res.path).predict(XNOR)
    assert (pred >= 0.5).int().flatten().tolist() == [1, 0, 0, 1]


def _edit(workspace, old, new):
    path = workspace / "examples" / "xnor_reuse.wgt"
    path.write_text(path.read_text().replace(old, new))
    return path


def test_pinned_hash_must_match(workspace, xor_build):
    sha = xor_build.manifest["weights"]["sha256"]
    ok = _edit(workspace, "layer=hidden freeze", f"layer=hidden sha256={sha} freeze")
    assert compile_file(ok, workspace / "weights").manifest["imports"][0]["pinned"]
    bad = _edit(workspace, sha, "0" * 64)
    with pytest.raises(WgtError, match="pin mismatch"):
        compile_file(bad, workspace / "weights", force=True)


def test_incompatible_import_shape(workspace, xor_build):
    path = _edit(workspace, "layer=hidden freeze", "layer=head freeze")
    with pytest.raises(WgtError, match=re.escape("incompatible import")):
        compile_file(path, workspace / "weights")


def test_missing_source_artifact(workspace):
    with pytest.raises(WgtError, match="cannot import 'hidden'") as exc:
        compile_file(workspace / "examples" / "xnor_reuse.wgt", workspace / "weights")
    assert exc.value.line is not None


def test_import_from_untrained_artifact_stays_flagged(workspace):
    (workspace / "examples" / "u_src.wgt").write_text(
        "wgt 0.1\nmodel src\ninput x: f32[2]\noutput y: f32[1]\n"
        "layer hidden: linear(in=2, out=8)\nlayer act: tanh\nlayer head: linear(in=8, out=1)\ninit seed=3\n"
    )
    compile_file(workspace / "examples" / "u_src.wgt", workspace / "weights")
    (workspace / "examples" / "u_dst.wgt").write_text(
        "wgt 0.1\nmodel dst\ninput x: f32[2]\noutput y: f32[1]\n"
        "layer hidden: linear(in=2, out=8)\nlayer act: tanh\nlayer head: linear(in=8, out=1)\ninit seed=4\n"
        'import hidden from "../weights/src"\n'
    )
    res = compile_file(workspace / "examples" / "u_dst.wgt", workspace / "weights")
    art = load_artifact(res.path, warn_untrained=False)
    assert set(art.untrained_tensors) == {"hidden.weight", "hidden.bias", "head.weight", "head.bias"}
