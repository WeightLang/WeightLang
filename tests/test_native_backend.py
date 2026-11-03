"""Native execution agrees with Torch; invalid contracts fail before inference."""
import json
import os
from pathlib import Path
import struct

import pytest
import torch

from weightlang import WgtError, load
from weightlang.native import NativeSession, lower

BINARY = Path(os.environ.get("WEIGHTLANG_NATIVE", Path(__file__).resolve().parents[1] / "target/debug/wgt-native"))
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build cargo build -p weightlang-runtime first")

SOURCE = """wgt 0.3
model native_demo
input x: f32[2]
output y: f32[2]
layer first: linear(in=2,out=2)
layer act: tanh
layer second: linear(in=2,out=2,bias=0)
init seed=7
"""

@pytest.fixture
def source(tmp_path):
    path = tmp_path / "model.wgt"
    path.write_text(SOURCE)
    return path


def test_native_matches_torch_and_budget(source):
    assert BINARY.exists(), "build cargo build -p weightlang-runtime before these integration tests"
    model = load(source)
    weights = {name: p.detach().flatten().tolist() for name, p in model.named_parameters()}
    with NativeSession.from_source(source, weights, budget_bytes=4096, executable=BINARY, device="cpu") as native:
        report = native.report
        for x in ([1., 2.], [-2., 0.5], [0., 0.]):
            expected = model(torch.tensor(x)).detach()
            torch.testing.assert_close(torch.tensor(native(x)), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        NativeSession.from_source(source, weights, budget_bytes=report["managed_bytes"]-1, executable=BINARY, device="cpu")
    with pytest.raises(RuntimeError, match="closed"):
        native([1., 2.])


def test_file_base_delta_and_slice(source, tmp_path):
    weight_file = tmp_path / "base.f32"
    weight_file.write_bytes(struct.pack("<5f", 99., 1., 2., 3., 4.))
    weights = {
        "base": dict(shape=[2,2], kind="file", path=str(weight_file), offset=4),
        "delta": dict(shape=[2,2], kind="resident", data=[0.5,0,0,-0.5]),
        "first.weight": dict(kind="add", base="base", delta="delta"),
        "first.bias": dict(kind="view", source="base", start=1),
        "second.weight": [1.,0.,0.,1.],
    }
    with NativeSession.from_source(source, weights, budget_bytes=1024, executable=BINARY, device="cpu") as native:
        result = native([0.1, 0.2])
        torch.testing.assert_close(torch.tensor(result), torch.tanh(torch.tensor([2.55,4.])))
        assert native.report["streamed_parameters"] == 1
    assert weight_file.read_bytes() == struct.pack("<5f", 99., 1., 2., 3., 4.)


def test_wgt_sharing(source):
    source.write_text(SOURCE + "parameter base = first.weight\nshare second.weight = base\n")
    model = load(source)
    weights = {name: p.detach().flatten().tolist() for name, p in model.named_parameters()}
    with NativeSession.from_source(source, weights, budget_bytes=4096, executable=BINARY, device="cpu") as native:
        expected = model(torch.tensor([0.2,0.4]))["y"].detach()
        torch.testing.assert_close(torch.tensor(native([0.2,0.4])), expected)
        assert native.report["resident_bytes"] == 24


def test_reject_unsupported_and_missing(source):
    with pytest.raises(WgtError, match="missing native parameter"):
        lower(source, {})
    source.write_text(SOURCE.replace("tanh", "gelu"))
    with pytest.raises(WgtError, match="unsupported operator"):
        lower(source, {})


def test_bad_ir_unknown_fields_and_shape(source, tmp_path):
    model = lower(source, {"first.weight": [1,2,3,4], "first.bias": [0,0], "second.weight": [1,0,0,1]})
    model["parameters"][0]["device"] = "cuda"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(model))
    with pytest.raises(RuntimeError, match="unknown field"):
        NativeSession(path, budget_bytes=4096, executable=BINARY, device="cpu")
    del model["parameters"][0]["device"]
    model["parameters"][0]["shape"] = [4,1]
    path.write_text(json.dumps(model))
    with pytest.raises(RuntimeError, match="linear weight"):
        NativeSession(path, budget_bytes=4096, executable=BINARY, device="cpu")


def test_reject_branch(source):
    source.write_text(SOURCE.replace("linear(in=2,out=2,bias=0)", "linear(in=2,out=2,bias=0) from input"))
    with pytest.raises(WgtError, match="sequential chain"):
        lower(source, {})


def test_native_cli_runs_without_python(tmp_path):
    import subprocess
    example = Path(__file__).resolve().parents[1] / "examples/native_backend/model.json"
    result = subprocess.run([str(BINARY), str(example), "64"], input="[0.1,0.2]\n[0,0]\n", text=True,
                            capture_output=True, check=True, env={"PATH": str(tmp_path)})
    report, first, second = map(json.loads, result.stdout.splitlines())
    assert report["managed_bytes"] == 64
    assert first == pytest.approx([0.4699452, 0.7931991])
    assert second == [0, 0]


def test_streaming_multiple_chunks_and_truncation(tmp_path):
    path = tmp_path / "weights.f32"
    path.write_bytes(struct.pack("<1105f", *range(1105)))
    model = dict(version=1, input=1100, parameters=[
        dict(name="stored", shape=[1105], kind="file", path="weights.f32", offset=0),
        dict(name="window", shape=[1,1100], kind="view", source="stored", start=5),
    ], layers=[dict(op="linear", weight="window", bias=None)])
    ir = tmp_path / "model.json"
    ir.write_text(json.dumps(model))
    with NativeSession(ir, budget_bytes=13200, executable=BINARY, device="cpu") as session:
        assert session([1.] * 1100) == [sum(range(5,1105))]
        assert session.report["resident_bytes"] == 0
        path.write_bytes(b"")
        with pytest.raises(RuntimeError, match="parameter read failed"):
            session([1.] * 1100)
