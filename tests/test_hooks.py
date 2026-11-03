from pathlib import Path

import pytest
import torch

from weightlang import WgtError, compile_file, forward_hooks, load, load_artifact

SOURCE = Path(__file__).resolve().parents[1] / 'examples/python_first/model.wgt'


def test_observation_preserves_forward_and_gradients():
    model = load(SOURCE)
    x = torch.ones(2, 4, requires_grad=True)
    expected = model(x)
    params = (x, *model.parameters())
    reference_grad = torch.autograd.grad(expected.sum(), params)
    captured = []
    def observe(module, inputs, output):
        captured.append((inputs[0].detach().clone(), output.detach().clone()))
    with forward_hooks(model, {'hidden.step0': observe, 'head': observe, '': observe}):
        actual = model(x)
        gradients = torch.autograd.grad(actual.sum(), params)
    assert len(captured) == 3
    assert captured[0][0].shape == (2, 4)
    assert captured[0][1].shape == (2, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(gradients, reference_grad):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    model(x)
    assert len(captured) == 3


def test_nested_scopes_and_native_hook_are_preserved():
    model = load(SOURCE)
    counts = [0, 0, 0]
    def callback(index):
        def observe(module, inputs, output):
            counts[index] += 1
        return observe
    handle = model.head.register_forward_hook(callback(0))
    try:
        with forward_hooks(model, {'head': callback(1)}):
            with forward_hooks(model, {'head': callback(2)}):
                model(torch.ones(1, 4))
            model(torch.ones(1, 4))
        model(torch.ones(1, 4))
        assert counts == [3, 2, 1]
    finally:
        handle.remove()


def test_callback_exception_removes_all_owned_hooks():
    model = load(SOURCE)
    def fail(module, inputs, output):
        raise RuntimeError('callback failed')
    with pytest.raises(RuntimeError, match='callback failed'):
        with forward_hooks(model, {'hidden': fail, 'head': fail}):
            model(torch.ones(1, 4))
    assert not model.hidden._forward_hooks and not model.head._forward_hooks
    model(torch.ones(1, 4))


@pytest.mark.parametrize('invalid, exception', [({'missing': print}, AttributeError),
                                                ({'head': None}, TypeError), ({1: print}, TypeError)])
def test_invalid_targets_do_not_install_partial_hooks(invalid, exception):
    model = load(SOURCE)
    with pytest.raises(exception):
        with forward_hooks(model, {'hidden': print, **invalid}):
            pytest.fail('invalid hook should not enter the context')
    assert not model.hidden._forward_hooks


def test_output_replacement_is_scoped_and_differentiable():
    model = load(SOURCE)
    x = torch.ones(2, 4)
    expected = model(x)
    with forward_hooks(model, {'head': lambda module, inputs, output: output * 2}):
        actual = model(x)
        actual.sum().backward()
    torch.testing.assert_close(actual, expected * 2)
    assert model.head.weight.grad is not None
    torch.testing.assert_close(model(x), expected)


def test_shared_scan_hooks_fire_each_step_and_merge_inputs_are_nested():
    model = load(SOURCE.parents[1] / 'recurrent.wgt')
    steps, merges = [], []
    with forward_hooks(model, {
        'memory.gate': lambda module, inputs, output: steps.append(output.detach().clone()),
        'memory.joined': lambda module, inputs, output: merges.append(inputs),
    }):
        model(torch.ones(2, 8, 4)).sum().backward()
    assert len(steps) == len(merges) == 8
    assert steps[0].shape == (2, 16)
    assert isinstance(merges[0][0], list) and len(merges[0][0]) == 2


def test_saving_after_context_removes_runtime_dependency(tmp_path):
    built = compile_file(SOURCE, tmp_path / 'baseline', no_train=True)
    artifact = load_artifact(built.path, warn_untrained=False)
    with forward_hooks(artifact.module, {'head': lambda module, inputs, output: None}):
        with pytest.raises(WgtError, match='hooks'):
            artifact.save(tmp_path / 'hooked')
    saved = artifact.save(tmp_path / 'saved')
    x = torch.ones(1, 4)
    torch.testing.assert_close(saved.predict(x), artifact.predict(x))
