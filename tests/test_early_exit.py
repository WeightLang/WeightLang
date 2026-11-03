"""Actual stopping, per-sample retention, gradients, and portable loop contracts."""
import importlib.util
from pathlib import Path

import pytest
import torch

from weightlang import WgtError, compile_file, export_python, load_artifact, parse
from weightlang.artifact import build_module
from weightlang.decompile import decompile

SOURCE = '''wgt 0.3
model halt
input x: f32[2]
output y: f32[2]
block cell():
  layer next: linear(2) from input
  layer stop: linear(1) -> sigmoid from next
  layer result: identity from next
layer loop: recur(5, cell(), until=stop)
init seed=4
'''


def model(text=SOURCE):
    program = parse(text)
    return build_module([layer.to_dict() for layer in program.layers])


def configure(m, bias):
    with torch.no_grad():
        m.loop.next.weight.copy_(torch.eye(2))
        m.loop.next.bias.fill_(1)
        m.loop.stop.step0.weight.zero_()
        m.loop.stop.step0.bias.fill_(bias)


@pytest.mark.parametrize('bias,steps', [(100., 1), (-100., 5)])
def test_true_exit_and_upper_bound(bias, steps):
    m = model()
    configure(m, bias)
    calls = []
    hook = m.loop.next.register_forward_hook(lambda *args: calls.append(1))
    x = torch.zeros(3, 2, requires_grad=True)
    y = m(x)
    hook.remove()
    torch.testing.assert_close(y, x + steps)
    assert len(calls) == steps
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(m.loop.next.bias.grad, torch.full((2,), 3. * steps))
    assert m.loop.stop.step0.weight.grad is None
    assert len(m.state_dict()) == 4


def test_independent_samples_freeze_and_reset(monkeypatch):
    m = model()
    configure(m, 0.)  # p = 0.5 at every step
    draws = iter([[[.1], [.9]], [[.9], [.1]]])
    monkeypatch.setattr(torch, 'rand_like', lambda p: p.new_tensor(next(draws)))
    x = torch.zeros(2, 2, requires_grad=True)
    y = m(x)
    torch.testing.assert_close(y, torch.tensor([[1., 1.], [2., 2.]]))
    y.sum().backward()
    torch.testing.assert_close(m.loop.next.bias.grad, torch.full((2,), 3.))
    # A fresh invocation has no persistent stopping state.
    monkeypatch.setattr(torch, 'rand_like', lambda p: torch.zeros_like(p))
    torch.testing.assert_close(m(x), torch.ones_like(x))
    # The unbatched core module has the same semantics.
    torch.testing.assert_close(m(torch.zeros(2)), torch.ones(2))


@pytest.mark.parametrize('old,new', [
    ('until=stop', 'until=missing'),
    ('linear(1)', 'linear(2)'),
    ('recur(5,', 'recur(0,'),
    ('recur(5,', 'repeat(5,'),
    ('until=stop', 'until=stop.step1'),
])
def test_invalid_contracts(old, new):
    with pytest.raises(WgtError):
        parse(SOURCE.replace(old, new))


@pytest.mark.parametrize('p', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_probabilities_fail(p):
    m = model()
    hook = m.loop.stop.register_forward_hook(lambda module, args, result: torch.full_like(result, p))
    with pytest.raises(ValueError, match='probability'):
        m(torch.zeros(2, 2))
    hook.remove()


def test_roundtrip_export_decompile_and_seed(tmp_path):
    path = tmp_path / 'halt.wgt'
    path.write_text(SOURCE)
    built = compile_file(path, tmp_path / 'models', no_train=True)
    restored = load_artifact(built.path, warn_untrained=False).module
    target = tmp_path / 'standalone.py'
    export_python(built.path, target)
    spec = importlib.util.spec_from_file_location('halt_export', target)
    exported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported)
    standalone = exported.load_model()
    x = torch.randn(12, 2)
    results = []
    for m in (built.module, restored, standalone):
        torch.manual_seed(123)
        results.append(m(x))
    for result in results[1:]:
        torch.testing.assert_close(result, results[0])
    recovered = tmp_path / 'recovered.wgt'
    decompile(str(built.path), recovered)
    parsed = parse(recovered.read_text())
    assert parsed.layers[0].args['until'] == 'stop'
    assert len(build_module([l.to_dict() for l in parsed.layers]).state_dict()) == 4


def test_documented_example():
    m = model(Path('examples/early_exit.wgt').read_text())
    x = torch.randn(3, 8, requires_grad=True)
    assert m(x).shape == x.shape
    m(x).sum().backward()
    assert x.grad is not None


def test_stop_is_a_control_dependency():
    from weightlang.debugger import lint
    assert not lint(parse(SOURCE).layers)
    unused = SOURCE.replace('  layer result:', '  layer unused: linear(2) from next\n  layer result:')
    warnings = lint(parse(unused).layers)
    assert len(warnings) == 1 and 'unused' in warnings[0]['message']
