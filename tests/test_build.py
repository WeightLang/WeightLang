import json
import warnings

import pytest
import torch

from weightlang import UntrainedWeightsWarning, WgtError, load_artifact
from weightlang.cli import main
from weightlang.compiler import compile_file

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_xor_is_trained_and_correct_on_training_rows(xor_build):
    m = xor_build.manifest
    assert {t["origin"] for t in m["weights"]["tensors"].values()} == {"trained"}
    assert m["training"]["train_accuracy"] == 1.0
    assert m["training"]["final_loss"] < m["training"]["initial_loss"]
    with torch.no_grad():
        pred = xor_build.module(torch.tensor(XOR_X, dtype=torch.float32))
    assert ((pred >= 0.5).int().flatten().tolist()) == [0, 1, 1, 0]


def test_reload_gives_identical_outputs(xor_build):
    x = torch.tensor(XOR_X, dtype=torch.float32)
    with torch.no_grad():
        before = xor_build.module(x)
    loaded = load_artifact(xor_build.path)
    after = loaded.predict(x)
    assert torch.equal(before, after)  # bit-for-bit, not approximately
    for name, tensor in xor_build.module.state_dict().items():
        assert torch.equal(tensor, loaded.module.state_dict()[name])
    assert loaded.predict([0, 1]).shape == (1,)


def test_manifest_records_contract_and_versions(xor_build):
    m = json.loads((xor_build.path / "manifest.json").read_text())
    assert m["format"] == "weightlang.artifact" and m["format_version"] == 1
    assert m["language_version"] == "0.1"
    assert m["compiler"]["name"] == "weightc"
    assert m["built_with"]["torch"] == torch.__version__
    assert m["interface"]["input"] == {"name": "bits", "dtype": "f32", "shape": [2]}
    assert m["interface"]["output"] == {"name": "prob", "dtype": "f32", "shape": [1]}
    assert m["init"]["seed"] == 42
    assert (xor_build.path / "source.wgt").read_text().startswith("wgt 0.1")


def test_build_is_byte_stable(workspace, xor_build):
    # Same source, same machine/torch: identical weight bytes, hence identical sha256.
    for i in range(5):
        again = compile_file(workspace / "examples" / "xor.wgt", workspace / f"other{i}")
        assert again.manifest["weights"]["sha256"] == xor_build.manifest["weights"]["sha256"]


def test_artifact_loads_with_plain_pytorch(xor_build):
    """The documented loader-free path: safetensors + manifest layer list."""
    from collections import OrderedDict

    from safetensors.torch import load_file

    manifest = json.loads((xor_build.path / "manifest.json").read_text())
    ops = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid}
    net = torch.nn.Sequential(
        OrderedDict(
            (
                layer["name"],
                torch.nn.Linear(layer["args"]["in"], layer["args"]["out"])
                if layer["op"] == "linear"
                else ops[layer["op"]](),
            )
            for layer in manifest["layers"]
        )
    )
    net.load_state_dict(load_file(str(xor_build.path / manifest["weights"]["file"])))
    x = torch.tensor(XOR_X, dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(net(x), load_artifact(xor_build.path).predict(x))


def test_tampered_weights_are_rejected(xor_build):
    w = xor_build.path / "weights.safetensors"
    data = bytearray(w.read_bytes())
    data[-1] ^= 0xFF
    w.write_bytes(bytes(data))
    with pytest.raises(WgtError, match="sha256 mismatch"):
        load_artifact(xor_build.path)


def test_wrong_input_shape_is_rejected(xor_build):
    with pytest.raises(WgtError, match="expects shape"):
        load_artifact(xor_build.path).predict([[1, 2, 3]])


def test_untrained_artifact_is_labelled(workspace):
    res = compile_file(workspace / "examples" / "untrained.wgt", workspace / "build")
    assert res.manifest["training"] is None
    assert {t["origin"] for t in res.manifest["weights"]["tensors"].values()} == {"random-init"}
    with pytest.warns(UntrainedWeightsWarning):
        art = load_artifact(res.path)
    assert len(art.untrained_tensors) == 4


def test_trained_artifact_does_not_warn(xor_build):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_artifact(xor_build.path)


def test_refuses_to_overwrite_foreign_directory(workspace):
    (workspace / "build" / "xor_mlp").mkdir()
    (workspace / "build" / "xor_mlp" / "keep.txt").write_text("mine")
    with pytest.raises(WgtError, match="refusing to overwrite"):
        compile_file(workspace / "examples" / "xor.wgt", workspace / "build")
    assert (workspace / "build" / "xor_mlp" / "keep.txt").exists()


def test_cli_build_run_inspect_verify(workspace, capsys):
    src = str(workspace / "examples" / "xor.wgt")
    out = str(workspace / "build")
    assert main(["check", src]) == 0
    assert main(["build", src, "-o", out]) == 0
    capsys.readouterr()
    assert main(["run", f"{out}/xor_mlp", "--input", json.dumps(XOR_X)]) == 0
    probs = json.loads(capsys.readouterr().out)["prob"]
    assert [round(p[0]) for p in probs] == [0, 1, 1, 0]
    assert main(["inspect", f"{out}/xor_mlp"]) == 0
    assert "hidden.weight" in capsys.readouterr().out
    assert main(["verify", f"{out}/xor_mlp"]) == 0
    assert main(["run", f"{out}/missing", "--input", "[0, 1]"]) == 1
    assert "not a WeightLang artifact" in capsys.readouterr().err
