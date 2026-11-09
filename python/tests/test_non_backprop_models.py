"""Real standard-library models preserve native state and local updates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch.nn import functional as F

import weightlang as wgt


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/native_training/non_backprop"


@pytest.fixture(autouse=True)
def compiler(monkeypatch):
    monkeypatch.setenv("WGT_COMPILER", os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt")))


CASES = [
    ("snn", [
        {"name": "step", "inputs": [[0.6, 0.2]]},
        {"name": "step", "inputs": [[0.6, 1.4]]},
        {"name": "observe", "inputs": []},
        {"name": "reset", "inputs": []},
        {"name": "step", "inputs": [[0.6, 0.2]]},
        {"name": "step", "inputs": [[0.6, 1.4]], "max_work": 1},
        {"name": "observe", "inputs": []},
    ]),
    ("ff", [
        {"name": "forward", "inputs": [[1., 0.]]},
        {"name": "adapt", "inputs": [[1., 0.], [0., 1.], [0.1]]},
        {"name": "goodness", "inputs": [[1., 0.]]},
        {"name": "forward", "inputs": [[0., 1.]]},
        {"name": "adapt", "inputs": [[0.8, 0.2], [0.3, 0.7], [0.1]]},
        {"name": "adapt", "inputs": [[1., 0.], [0., 1.], [0.1]], "max_work": 1},
    ]),
    ("stdp", [
        {"name": "adapt", "inputs": [[1., 0.], [0., 0.]]},
        {"name": "adapt", "inputs": [[0., 0.], [0., 1.]]},
        {"name": "forward", "inputs": [[1., 0.]]},
        {"name": "adapt", "inputs": [[1., 0.], [0., 0.]]},
        {"name": "reset", "inputs": []},
        {"name": "adapt", "inputs": [[1., 1.], [1., 1.]], "max_work": 1},
        {"name": "forward", "inputs": [[1., 0.]]},
    ]),
]


@pytest.mark.parametrize("name,calls", CASES)
def test_standard_models_match_native_after_every_transaction(name, calls, tmp_path, monkeypatch):
    definition = wgt.compile(EXAMPLES / name / "model.wgt")
    executable = os.environ.get("WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference"))
    process = subprocess.run([executable], input=json.dumps({"source": definition["source"], "calls": calls}),
                             text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    expected = json.loads(process.stdout)
    model = wgt.from_ir(definition, device="cpu")

    def forbidden(*_, **__):
        pytest.fail("typed standard-library updates must not invoke autograd")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    for call, result in zip(calls, expected["results"]):
        inputs = [torch.tensor(values) for values in call["inputs"]]
        if "error" in result:
            with pytest.raises((ValueError, RuntimeError)):
                model.call(call["name"], *inputs, max_work=call.get("max_work"))
        else:
            actual = model.call(call["name"], *inputs)
            if result["output"] is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, torch.tensor(result["output"]), rtol=2e-6, atol=1e-7)
                assert not actual.requires_grad
        for parameter, values in result["parameters"].items():
            torch.testing.assert_close(model.parameter_values()[parameter].flatten(), torch.tensor(values), rtol=2e-6, atol=1e-7)
        for state, values in result["states"].items():
            torch.testing.assert_close(getattr(model, model.state_names[state]).flatten(), torch.tensor(values), rtol=2e-6, atol=1e-7)
    before = deepcopy(model.state_dict())
    saved = wgt.save(model, tmp_path / f"{name}.pt")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "absent"))
    restored = wgt.load(saved, device="cpu")
    for key, value in before.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_real_ff_rule_matches_independent_local_objective_derivative():
    model = wgt.load(EXAMPLES / "ff/model.wgt", device="cpu")
    initial = model.parameter_values()
    weight = initial["weight"].double().requires_grad_()
    bias = initial["bias"].double().requires_grad_()
    positive, negative = torch.tensor([1., 0.5]), torch.tensor([0.2, 1.])
    pos = torch.relu(weight @ positive.double().reshape(2, 1) + bias)
    neg = torch.relu(weight @ negative.double().reshape(2, 1) + bias)
    threshold = initial["threshold"].item()
    objective = (F.logsigmoid(pos.square().sum() - threshold)
                 + F.logsigmoid(threshold - neg.square().sum())) / 2
    dw, db = torch.autograd.grad(objective, (weight, bias))
    wgt.train_forward_forward(model, [positive], [negative], rate=0.1)
    actual = model.parameter_values()
    torch.testing.assert_close(actual["weight"].double(), weight + 0.1 * dw, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(actual["bias"].double(), bias + 0.1 * db, rtol=2e-6, atol=1e-7)
    for name in ("threshold", "epsilon"):
        torch.testing.assert_close(actual[name], initial[name], rtol=0, atol=0)
    activity = model(positive)
    torch.testing.assert_close(activity.norm(), torch.ones(()), rtol=1e-6, atol=0.)
