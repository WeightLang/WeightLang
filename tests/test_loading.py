from pathlib import Path

import pytest
import torch

from weightlang import WgtError, compile_file, load, load_artifact

SOURCE = Path(__file__).resolve().parents[1] / 'examples/python_first/model.wgt'


def test_load_is_in_memory_deterministic_and_differentiable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = torch.random.get_rng_state().clone()
    model = load(SOURCE)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []
    assert isinstance(model, torch.nn.Module) and not model.training
    other = load(SOURCE)
    built = compile_file(SOURCE, tmp_path / 'built', no_train=True)
    x = torch.eye(4)[:3]
    torch.testing.assert_close(model(x), built.module(x), rtol=0, atol=0)
    torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.hidden.step0.weight.detach().clone()
    torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2])).backward()
    optimizer.step()
    assert not torch.equal(before, model.hidden.step0.weight)


def test_source_load_never_executes_data_or_training(tmp_path, monkeypatch):
    source = tmp_path / 'model.wgt'
    source.write_text(SOURCE.read_text() + '\ndata samples: python="missing.py:load"\n'
                      'train data=samples loss=cross_entropy epochs=1 python="missing.py:train"\n')
    import weightlang.compiler as compiler
    def forbidden(*args, **kwargs):
        raise AssertionError('training must not run')
    monkeypatch.setattr(compiler, '_train', forbidden)
    model = load(source)
    assert model(torch.ones(2, 4)).shape == (2, 3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['model.wgt']


def test_freezing_and_saved_artifact_roundtrip(tmp_path):
    source = tmp_path / 'model.wgt'
    source.write_text(SOURCE.read_text() + '\nfreeze hidden\n')
    model = load(source)
    assert all(not p.requires_grad for p in model.hidden.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    built = compile_file(source, tmp_path / 'baseline', no_train=True)
    artifact = load_artifact(built.path, warn_untrained=False)
    with torch.no_grad():
        model.head.bias.add_(1)
    artifact.module.load_state_dict(model.state_dict())
    saved = artifact.save(tmp_path / 'trained', training={'method': 'test update'})
    restored = load(saved.path)
    assert all(not p.requires_grad for p in restored.hidden.parameters())
    torch.testing.assert_close(restored(torch.ones(2, 4)), model(torch.ones(2, 4)), rtol=0, atol=0)
    weights = saved.path / 'weights.safetensors'
    weights.write_bytes(weights.read_bytes() + b'corrupt')
    with pytest.raises(WgtError):
        load(saved.path)


def test_invalid_source_rejected_without_output(tmp_path):
    source = tmp_path / 'bad.wgt'
    source.write_text(SOURCE.read_text().replace('f32[3]', 'f32[5]'))
    with pytest.raises(WgtError):
        load(source)
    assert list(tmp_path.iterdir()) == [source]
