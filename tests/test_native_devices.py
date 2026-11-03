"""Device selection is measured, explicit requests never silently fall back."""
import json
import os
from pathlib import Path
import struct

import pytest
import torch

from weightlang import load
from weightlang.native import NativeSession, available_devices

BINARY = Path(os.environ.get("WEIGHTLANG_NATIVE", Path(__file__).resolve().parents[1] / "target/debug/wgt-native"))
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build native binary first")
DEVICES = available_devices(executable=BINARY) if BINARY.exists() else []
GPU_IDS = [d["id"] for d in DEVICES if d["backend"] != "cpu"]
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/native_backend/model.json"


def test_discovery_and_required_hardware():
    assert DEVICES[0]["id"] == "cpu"
    assert len({d["id"] for d in DEVICES}) == len(DEVICES)
    if os.environ.get("WEIGHTLANG_REQUIRE_GPU"):
        assert GPU_IDS, "GPU required for this local acceptance run"


def test_auto_uses_fastest_valid_measurement():
    with NativeSession(EXAMPLE, budget_bytes=4096, executable=BINARY) as session:
        candidates = [c for c in session.report["candidates"] if c["rejected"] is None]
        winner = min(candidates, key=lambda c: c["median_ns"])
        assert session.report["device"] == winner["device"]["id"]
        assert session([0.1, 0.2]) == pytest.approx([0.4699452, 0.7931991], abs=1e-5)
        assert session.report["managed_bytes"] == session.report["host_managed_bytes"] + session.report["gpu_buffer_bytes"]


def test_strict_zero_alloc_and_invalid_requests():
    with NativeSession(EXAMPLE, budget_bytes=64, executable=BINARY, strict_zero_alloc=True) as session:
        assert session.report["device"] == "cpu"
        assert session.report["zero_heap_allocation"]
        assert len(session.report["candidates"]) == 1
    for device, message in [("cuda:0", "invalid native device"), ("vulkan:99999", "unavailable|not compiled")]:
        with pytest.raises(RuntimeError, match=message):
            NativeSession(EXAMPLE, budget_bytes=4096, executable=BINARY, device=device)
    with pytest.raises(RuntimeError, match="strict zero-allocation"):
        NativeSession(EXAMPLE, budget_bytes=4096, executable=BINARY, device="gpu", strict_zero_alloc=True)


def test_auto_budget_fallback_is_reported():
    with NativeSession(EXAMPLE, budget_bytes=64, executable=BINARY) as session:
        assert session.report["device"] == "cpu"
        for candidate in session.report["candidates"][1:]:
            assert "budget exceeded" in candidate["rejected"]


@pytest.mark.parametrize("gpu_device", GPU_IDS or [None])
def test_hardware_all_operators_match_torch(tmp_path, gpu_device):
    if gpu_device is None:
        pytest.skip("no GPU adapter available")
    source = tmp_path / "model.wgt"
    source.write_text('''wgt 0.3
model gpu_check
input x: f32[67]
output y: f32[7]
layer first: linear(in=67,out=65)
layer a: tanh
layer second: linear(in=65,out=7)
layer b: relu
layer c: sigmoid
layer d: identity
init seed=17
''')
    model = load(source)
    parameters = {n: p.detach().flatten().tolist() for n, p in model.named_parameters()}
    with NativeSession.from_source(source, parameters, budget_bytes=100000, executable=BINARY, device=gpu_device) as gpu:
        assert gpu.report["device"] == gpu_device
        assert not gpu.report["zero_heap_allocation"]
        for scale in [0., 0.1, 1., -3., 100.]:
            x = torch.linspace(-1, 1, 67) * scale
            expected = model(x).detach()
            torch.testing.assert_close(torch.tensor(gpu(x.tolist())), expected, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("gpu_device", GPU_IDS or [None])
def test_gpu_streaming_reuses_scratch_in_queue_order(tmp_path, gpu_device):
    if gpu_device is None:
        pytest.skip("no GPU adapter available")
    (tmp_path / "weights.f32").write_bytes(struct.pack("<21f", *[i/30 for i in range(21)]))
    ir = dict(version=1, input=5, parameters=[
        dict(name="base", shape=[3,5], kind="file", path="weights.f32", offset=0),
        dict(name="delta", shape=[3,5], kind="resident", data=[0.01]*15),
        dict(name="sum", shape=[3,5], kind="add", base="base", delta="delta"),
        dict(name="bias", shape=[3], kind="view", source="base", start=1),
        dict(name="second", shape=[2,3], kind="file", path="weights.f32", offset=60),
    ], layers=[dict(op="linear", weight="sum", bias="bias"), dict(op="tanh"),
               dict(op="linear", weight="second"), dict(op="sigmoid"), dict(op="identity")])
    path = tmp_path / "model.json"
    path.write_text(json.dumps(ir))
    with NativeSession(path, budget_bytes=4096, executable=BINARY, device="cpu") as cpu, \
         NativeSession(path, budget_bytes=4096, executable=BINARY, device=gpu_device) as gpu:
        for i in range(8):
            x = [i/10, -0.5, 0.3, 0.2, -1.]
            assert gpu(x) == pytest.approx(cpu(x), rel=1e-4, abs=1e-5)
        assert gpu.report["streamed_parameters"] == 2
        (tmp_path / "weights.f32").write_bytes(b"")
        with pytest.raises(RuntimeError, match="parameter read failed"):
            gpu([0.]*5)


@pytest.mark.parametrize("gpu_device", GPU_IDS or [None])
def test_explicit_gpu_budget_failure_and_no_fallback(gpu_device):
    if gpu_device is None:
        pytest.skip("no GPU adapter available")
    with pytest.raises(RuntimeError, match="no compatible device"):
        NativeSession(EXAMPLE, budget_bytes=64, executable=BINARY, device=gpu_device)
