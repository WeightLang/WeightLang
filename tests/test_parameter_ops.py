from pathlib import Path

import pytest
import torch

from weightlang import (WgtError, call_with_parameters, compile_file, load,
                       load_artifact, read_parameters, write_parameters)
from test_primitives import variants

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/parameter_updates/model.wgt").read_text()


def test_functional_binding_preserves_values_identity_and_gradient(tmp_path):
    for model in variants(tmp_path, SOURCE):
        original = model.projection.weight
        before = original.detach().clone()
        supplied = torch.tensor([[2., 3.]], requires_grad=True)
        x = torch.tensor([4., 5.])
        out = call_with_parameters(model, {"projection.weight": supplied}, x)
        torch.testing.assert_close(out, torch.tensor([23.]))
        torch.testing.assert_close(torch.autograd.grad(out.sum(), supplied)[0], x[None])
        assert model.projection.weight is original
        torch.testing.assert_close(original, before)
        with pytest.raises(Exception):
            call_with_parameters(model, {"projection.weight": supplied}, torch.ones(3))
        assert model.projection.weight is original
        torch.testing.assert_close(original, before)


def test_sharing_alias_conflict_and_snapshot(tmp_path):
    source = SOURCE.replace("out=1", "out=2").replace("f32[1]", "f32[2]") + '''layer second: linear(in=2,out=2,bias=0)
parameter shared = projection.weight
share second.weight = shared
'''
    for model in variants(tmp_path, source):
        snapshot = read_parameters(model)
        assert list(snapshot) == ["projection.weight"]
        snapshot["projection.weight"].zero_()
        assert torch.count_nonzero(model.projection.weight)
        weight = torch.eye(2) * 2
        assert model.projection.weight is model.second.weight
        out = call_with_parameters(model, {"second.weight": weight}, torch.ones(2))['y']
        torch.testing.assert_close(out, torch.ones(2) * 4)
        with pytest.raises(WgtError, match="same shared"):
            write_parameters(model, {"projection.weight": weight, "second.weight": weight})
        original = model.projection.weight
        write_parameters(model, {"second.weight": weight})
        assert model.projection.weight is original is model.second.weight
        torch.testing.assert_close(model(torch.ones(2))['y'], out)


def test_validation_before_write_swaps_and_freezing():
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    before = read_parameters(model)
    invalid = [
        ({"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}, "frozen"),
        ({"weight": torch.zeros(2, 2), "unknown": torch.zeros(1)}, "unknown"),
        ({"weight": torch.zeros(4)}, "shape"),
        ({"weight": torch.zeros(2, 2, dtype=torch.float64)}, "dtype"),
        ({"weight": torch.full((2, 2), float('nan'))}, "nonfinite"),
    ]
    for values, message in invalid:
        with pytest.raises(WgtError, match=message):
            write_parameters(model, values)
        for name, value in read_parameters(model).items():
            torch.testing.assert_close(value, before[name])
    # Distinct source/destination aliases must be snapshotted before any copy.
    pair = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.])),
                                   torch.nn.Parameter(torch.tensor([2.]))])
    write_parameters(pair, {'0': pair[1], '1': pair[0]})
    assert pair[0].item() == 2 and pair[1].item() == 1


def test_model_driven_update_without_autograd_and_artifact_roundtrip(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("autograd must not be used")
    monkeypatch.setattr(torch.autograd, 'backward', forbidden)
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    path = tmp_path / 'model.wgt'
    path.write_text(SOURCE)
    built = compile_file(path, tmp_path / 'baseline', no_train=True)
    artifact = load_artifact(built.path, warn_untrained=False)
    updater_code = (ROOT / 'examples/parameter_updates/updater.wgt').read_text()
    (tmp_path / 'updater').mkdir()
    for updater in variants(tmp_path / 'updater', updater_code):
        write_parameters(updater, {'direction.weight': torch.eye(2) * 0.25})
        write_parameters(artifact.module, {'projection.weight': torch.zeros(1, 2)})
        state = torch.zeros(1, 2)
        desired = torch.tensor([[2., -1.]])
        with torch.no_grad():
            for _ in range(12):
                old = read_parameters(artifact.module)['projection.weight']
                result = updater(old, desired-old, state)
                write_parameters(artifact.module, {'projection.weight': result['next_weights']})
                state = result['next_memory']
        actual = read_parameters(artifact.module)['projection.weight']
        assert (actual-desired).norm() < 0.06
        assert all(p.grad is None for p in artifact.module.parameters())
    saved = artifact.save(tmp_path / 'updated')
    torch.testing.assert_close(saved.module.projection.weight, actual)
    assert saved.manifest['weights']['tensors']['projection.weight']['origin'] == 'edited'


def test_weight_domain_and_buffer_isolation(tmp_path):
    from test_weight_types import CODE
    for model in variants(tmp_path, CODE):
        value = torch.full((3, 3), 0.8, requires_grad=True)
        out = call_with_parameters(model, {'projection.weight': value}, torch.eye(3))['y']
        torch.testing.assert_close(out, torch.tensor([[1., .5, .8]]).expand(3, 3))
        assert torch.autograd.grad(out.sum(), value)[0].shape == (3, 3)
    bn = torch.nn.BatchNorm1d(2)
    before = bn.running_mean.clone()
    call_with_parameters(bn, {}, torch.ones(4, 2))
    torch.testing.assert_close(bn.running_mean, before)
