"""The host scheduler delegates every local update to the model's adapt entry."""

from copy import deepcopy
from pathlib import Path
import os

import pytest
import torch

from weightlang import torch_api as wgt


SOURCE = """wgt 0.5
model ExplicitLocalRule
parameter weight:f32[1]=[0]
frozen other:f32[1]=[7]
state count:f32[1]=[0];
fn adapt(positive:f32[1],negative:f32[1],rate:f32[1]) {
    weight <- weight + (positive-negative)*rate;
    count <- count + [1];
}
fn observe()->f32[1] { return weight; }
fn calls()->f32[1] { return count; }
"""


@pytest.fixture
def model(monkeypatch):
    compiler = os.environ.get("WGT_COMPILER", str(Path(__file__).resolve().parents[2] / "target/debug/wgt"))
    monkeypatch.setenv("WGT_COMPILER", compiler)
    return wgt.loads(SOURCE, device="cpu")


def test_pair_scheduler_is_ordered_no_ad_and_keeps_other_weights(model, monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("production FF scheduling must not invoke autograd")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    report = wgt.train_forward_forward(model, [torch.tensor([3.]), torch.tensor([5.])],
                                      [torch.tensor([1.]), torch.tensor([2.])], rate=0.5, epochs=2)
    assert report == wgt.FFReport(pairs=2, epochs=2, updates=4)
    torch.testing.assert_close(model.observe(), torch.tensor([5.]))
    torch.testing.assert_close(model.calls(), torch.tensor([4.]))
    assert model.parameter_values()["other"].item() == 7.
    assert all(value.grad is None for value in model.parameters())


@pytest.mark.parametrize("positive,negative", [
    ([torch.ones(1)], []),
    ([torch.ones(1), torch.ones(2)], [torch.zeros(1), torch.zeros(1)]),
    ([torch.ones(1), torch.tensor([float("nan")])], [torch.zeros(1), torch.zeros(1)]),
    ([torch.ones(1, requires_grad=True)], [torch.zeros(1)]),
    ([torch.ones(1, dtype=torch.float64)], [torch.zeros(1)]),
])
def test_all_dataset_validation_precedes_first_update(model, positive, negative):
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        wgt.train_forward_forward(model, positive, negative, rate=0.1)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_invalid_adapt_contract_and_rate_are_explicit(model):
    with pytest.raises(ValueError, match="adapt must"):
        wgt.train_forward_forward(model, [torch.ones(1)], [torch.zeros(1)], rate=0.1, adapt_operation="observe")
    with pytest.raises(ValueError, match="storage precision"):
        wgt.train_forward_forward(model, [torch.ones(1)], [torch.zeros(1)], rate=1e-100)


def test_restored_float64_model_uses_actual_storage_precision(model, tmp_path):
    model.double()
    restored = wgt.load(wgt.save(model, tmp_path / "ff.pt"), device="cpu")
    wgt.train_forward_forward(restored, [torch.ones(1, dtype=torch.float64)],
                             [torch.zeros(1, dtype=torch.float64)], rate=0.1)
    torch.testing.assert_close(restored.observe(), torch.tensor([0.1], dtype=torch.float64))
