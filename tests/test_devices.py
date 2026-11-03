import copy
import json

import pytest
import torch

from weightlang import WgtError, load_artifact
from weightlang.artifact import Residual
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.devices import apply_placement, placement_plan, resolve_device
from weightlang.weights import evaluate_artifact

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
SOURCE = """wgt 0.2
model device_test
input x: f32[2]
output y: f32[1]
layer hidden: linear(in=2, out=4)
layer act: tanh
layer head: linear(in=4, out=1)
data train:
  [0, 0] -> [0]
  [1, 1] -> [1]
data valid:
  [0, 1] -> [0.5]
init seed=42
train data=train loss=mse optimizer=adam lr=0.03 epochs=20 batch=1 validate=valid
"""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "model.wgt"
    path.write_text(SOURCE)
    return path


@pytest.mark.parametrize("device", ["auto", "xpu", "cpu:0", "cuda:-1", "cuda:no", "meta", "cuda:1:2"])
def test_invalid_devices(device):
    with pytest.raises(WgtError, match="unsupported device"):
        resolve_device(device)


def test_unavailable_cuda(monkeypatch, source, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(WgtError, match="CUDA-enabled"):
        compile_file(source, tmp_path / "out", device="cuda")
    assert not (tmp_path / "out").exists()


def test_invalid_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(WgtError, match="out of range"):
        resolve_device("cuda:1")


def test_unknown_parameter_rejected_before_write(source, tmp_path):
    with pytest.raises(WgtError, match="unknown placement target"):
        compile_file(source, tmp_path / "out", placements={"hidden.weigth": "cpu"})
    assert not (tmp_path / "out").exists()


def test_cpu_training_reload_rng_and_evaluation(source, tmp_path):
    rng = torch.get_rng_state().clone()
    result = compile_file(source, tmp_path / "one", placements={"hidden.weight": "cpu"})
    assert torch.equal(rng, torch.get_rng_state())
    again = compile_file(source, tmp_path / "two", device="cpu")
    assert result.manifest["weights"]["sha256"] == again.manifest["weights"]["sha256"]
    execution = result.manifest["built_with"]["execution"]
    assert execution["parameters"]["hidden.weight"] == "cpu"
    assert result.manifest["training"]["final_loss"] < result.manifest["training"]["initial_loss"]
    loaded = load_artifact(result.path, placements={"head": "cpu"})
    prediction = loaded.predict([[0, 1]])
    assert torch.equal(prediction, copy.deepcopy(loaded).predict([[0, 1]]))
    data = tmp_path / "evaluation.json"
    data.write_text(json.dumps({"inputs": [[0, 1]], "targets": [[0.5]]}))
    report = evaluate_artifact(result.path, data, "mse", compare=result.path, placements={"head": "cpu"})
    assert report["delta"]["loss"] == 0


def test_cli_device_map_and_override(source, tmp_path, capsys):
    mapping = tmp_path / "map.json"
    mapping.write_text('{"hidden": "cuda:999"}')
    out = tmp_path / "weights"
    assert (
        main(
            [
                "legacy-build",
                "--no-store",
                str(source),
                "-o",
                str(out),
                "--device-map",
                str(mapping),
                "--place",
                "hidden=cpu",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["run", str(out / "device_test"), "--input", "[[0,1]]", "--place", "head.weight=cpu"]) == 0
    assert len(json.loads(capsys.readouterr().out)["y"]) == 1


@pytest.mark.parametrize(
    "options",
    [
        ["--place", "hidden"],
        ["--place", "=cpu"],
        ["--place", "hidden=cpu", "--place", "hidden=cpu"],
    ],
)
def test_cli_bad_placement(source, tmp_path, options):
    assert main(["legacy-build", "--no-store", str(source), "-o", str(tmp_path / "out"), *options]) == 1
    assert not (tmp_path / "out").exists()


def test_shell_copies_preserve_gradients_and_names():
    # cpu:0 compares differently to cpu, exercising the functional-copy path on CPU-only hosts.
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    plan = placement_plan(model)
    plan["layers"]["0"] = "cpu:0"
    apply_placement(model, plan)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[0].weight.detach().clone()
    model(torch.ones(1, 2)).sum().backward()
    assert model[0].weight.grad is not None
    optimizer.step()
    assert not torch.equal(before, model[0].weight)
    assert set(model.state_dict()) == {"0.weight", "0.bias"}
    clone = copy.deepcopy(model)
    assert torch.equal(model(torch.ones(1, 2)), clone(torch.ones(1, 2)))


@CUDA
@pytest.mark.parametrize("placements", [{}, {"head": "cpu"}, {"hidden.weight": "cpu", "head.bias": "cpu"}])
def test_cuda_training_validation_reload(source, tmp_path, placements):
    rng = torch.cuda.get_rng_state().clone()
    result = compile_file(source, tmp_path / "weights", device="cuda:0", placements=placements)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert result.manifest["training"]["final_loss"] < result.manifest["training"]["initial_loss"]
    parameters = dict(result.module.named_parameters())
    for name, expected in result.manifest["built_with"]["execution"]["parameters"].items():
        assert str(parameters[name].device) == expected
        assert parameters[name].grad is not None
        assert parameters[name].grad.device == parameters[name].device
    gpu = load_artifact(result.path, device="cuda:0", placements=placements)
    cpu = load_artifact(result.path)
    assert torch.allclose(gpu.predict([0, 1]).cpu(), cpu.predict([0, 1]), atol=1e-5)
    data = tmp_path / "evaluation.json"
    data.write_text('{"inputs": [[0,1]], "targets": [[0.5]]}')
    assert evaluate_artifact(result.path, data, "mse", device="cuda:0")["candidate"]["loss"] >= 0


@CUDA
def test_residual_cross_device_and_freezing():
    model = torch.nn.Sequential(Residual(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)))
    model[0][0].weight.requires_grad_(False)
    plan = placement_plan(model, "cuda:0", {"0.1": "cpu", "0.0.bias": "cpu"})
    apply_placement(model, plan)
    x = torch.ones(1, 2, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert model[0][0].weight.grad is None
    assert model[0][0].bias.grad.device.type == "cpu"


@CUDA
def test_nested_rule_precedence():
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 2)))
    plan = placement_plan(model, "cpu", {"0": "cuda:0", "0.0.bias": "cpu"})
    assert plan["layers"]["0.0"] == "cuda:0"
    assert plan["parameters"] == {"0.0.weight": "cuda:0", "0.0.bias": "cpu"}


def test_unavailable_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(WgtError, match="macOS MPS"):
        resolve_device("mps")
