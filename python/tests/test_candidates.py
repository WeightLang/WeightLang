"""External candidate updates keep typed-operation storage atomic."""

from copy import deepcopy
from pathlib import Path
import os

import pytest
import torch

import weightlang as wgt


SOURCE = """wgt 0.5
model CandidateCell
parameter parameters: f32[2] = [1,2]
learn count: int in -2..=2 = 0;
learn enabled: bool = false;
learn bounded: float in -2..=2 = 0;
learn positive: float > 0 = 1;
frozen fixed: f32[1] = [5]
parameter probability: prob = 0.5
state activity: f32[2] = [0,0];
fn forward(x: f32[2]) -> f32[2] { return x * parameters + activity; }
fn step(x: f32[2]) -> f32[2] { activity <- activity + x; return activity; }
fn reset() { activity <- [0,0]; }
"""


@pytest.fixture
def model(monkeypatch):
    executable = os.environ.get("WGT_COMPILER", str(Path(__file__).resolve().parents[2] / "target/debug/wgt"))
    monkeypatch.setenv("WGT_COMPILER", executable)
    return wgt.loads(SOURCE, device="cpu")


def unchanged(model, expected):
    assert model.state_dict().keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_candidates_use_original_names_and_snapshot_inputs(model):
    assert model.parameter_names["parameters"] != "parameters"
    assert callable(model.parameters)
    model.step(torch.tensor([3., 4.]))
    value = torch.tensor([7., 8.])
    model.update_parameters({"parameters": value, "count": torch.tensor([-2.]),
                             "enabled": torch.ones(1)})
    value.zero_()
    torch.testing.assert_close(model(torch.ones(2)), torch.tensor([10., 12.]))
    snapshot = model.parameter_values()
    snapshot["parameters"].zero_()
    torch.testing.assert_close(model.parameter_values()["parameters"], torch.tensor([7., 8.]))
    assert not model.parameter_schema["fixed"]["candidate_updatable"]
    assert not model.parameter_schema["probability"]["candidate_updatable"]
    assert model.parameter_schema["parameters"]["candidate_updatable"]


@pytest.mark.parametrize("name,value", [
    ("count", torch.tensor([0.5])), ("enabled", torch.tensor([0.4])),
    ("bounded", torch.tensor([3.])), ("positive", torch.zeros(1)),
    ("fixed", torch.ones(1)), ("probability", torch.zeros(1)),
    ("activity", torch.zeros(2)), ("unknown", torch.zeros(1)),
    ("count", torch.tensor([float("nan")])), ("count", torch.zeros(2)),
    ("count", torch.zeros(1, dtype=torch.float64)),
    ("count", torch.zeros(1, requires_grad=True)),
])
def test_bad_candidate_rolls_back_whole_batch(model, name, value):
    model.step(torch.ones(2))
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        model.update_parameters({"parameters": torch.tensor([9., 9.]), name: value})
    unchanged(model, before)


def test_candidate_dtype_is_explicit_and_zero_updates_rejected(model):
    model.double()
    model.update_parameters({"parameters": torch.tensor([2., 3.], dtype=torch.float64)})
    assert model.parameter_values()["parameters"].dtype == torch.float64
    with pytest.raises(ValueError, match="nonempty"):
        model.update_parameters({})
    with pytest.raises(ValueError, match="Tensor"):
        model.update_parameters({"parameters": [2., 3.]})


def test_new_eligibility_metadata_cannot_enable_frozen_or_probability(model):
    definition = deepcopy(model.definition)
    for spec in definition["parameters"]:
        if spec["name"] == "probability":
            spec["candidate_updatable"] = True
    with pytest.raises(ValueError, match="candidate_updatable"):
        wgt.from_ir(definition, device="cpu")
    # Old exports remain safe because their full storage specs include prob.
    for spec in definition["parameters"]:
        spec.pop("candidate_updatable", None)
    restored = wgt.from_ir(definition, device="cpu")
    assert not restored.parameter_schema["probability"]["candidate_updatable"]
