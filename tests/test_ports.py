import importlib.util
import json

import pytest
import torch

from weightlang import WgtError, compile_file, export_python, load, load_artifact, parse

SOURCE = '''wgt 0.3
model streaming
input samples: f32[3,2]
input old: f32[4]
output states: f32[3,4] from memory
output new: f32[4] from final
block cell():
  layer joined: concat(input, state)
  layer next: linear(in=6, out=4) -> tanh
layer memory: scan(hidden=4, cell()) from samples, old
layer final: last from memory
init seed=7
'''


def test_streaming_values_gradients_and_roundtrip(tmp_path):
    path = tmp_path / 'stream.wgt'
    path.write_text(SOURCE)
    module = load(path)
    x = torch.randn(2, 6, 2, requires_grad=True)
    initial = torch.randn(2, 4, requires_grad=True)
    first = module(samples=x[:, :3], old=initial)
    second = module(x[:, 3:], first['new'])
    expected = []
    state = initial
    for item in x.unbind(1):
        state = torch.tanh(module.memory.next.step0(torch.cat([item, state], -1)))
        expected.append(state)
    actual = torch.cat([first['states'], second['states']], 1)
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(actual, expected)
    parameters = tuple(module.parameters()) + (x, initial)
    ga = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), parameters)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
    built = compile_file(path, tmp_path / 'models', no_train=True)
    restored = load_artifact(built.path, warn_untrained=False).module
    assert built.manifest['format_version'] == 4
    assert set(restored.state_dict()) == set(module.state_dict())
    out = restored(x[:, :3], initial)
    torch.testing.assert_close(out['new'], first['new'])
    exported = tmp_path / 'standalone.py'
    export_python(built.path, exported)
    spec = importlib.util.spec_from_file_location('port_export', exported)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.load_model()(samples=x[:, :3], old=initial)
    torch.testing.assert_close(out['states'], first['states'])
    assert not hasattr(restored, 'state')


def test_boundary_rejects_invalid_calls(tmp_path):
    path = tmp_path / 'stream.wgt'
    path.write_text(SOURCE)
    module = load(path)
    x, state = torch.randn(2, 3, 2), torch.randn(2, 4)
    for args, kwargs, error in [
        ((x,), {}, TypeError),
        ((x, state), {'old': state}, TypeError),
        ((x, state, state), {}, TypeError),
        ((x, state[0]), {}, ValueError),
        ((x.double(), state), {}, ValueError),
        ((x[:, :2], state), {}, ValueError),
    ]:
        with pytest.raises(error):
            module(*args, **kwargs)
    assert module(x[0], state[0])['new'].shape == (4,)


@pytest.mark.parametrize('old,new', [
    ('from final', 'from missing'),
    ('output new: f32[4]', 'output new: f32[5]'),
    ('input old: f32[4]', 'input old: f32[5]'),
    ('input old:', 'input samples:'),
    (' from final', ''),
])
def test_bad_static_contracts(old, new):
    with pytest.raises(WgtError):
        parse(SOURCE.replace(old, new))


def test_tampered_output_rejected(tmp_path):
    path = tmp_path / 'stream.wgt'
    path.write_text(SOURCE)
    built = compile_file(path, tmp_path / 'models', no_train=True)
    manifest_path = built.path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['interface']['outputs'][1]['from'] = 'memory'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WgtError, match='mismatch'):
        load_artifact(built.path, warn_untrained=False)


def test_bool_mask_and_branch_outputs(tmp_path):
    source = '''wgt 0.3
model conditional
input x: f32[4]
input mask: bool[4]
input fallback: f32[4]
output chosen: f32[4] from selected
output raw: f32[4] from projected
layer projected: linear(in=4, out=4)
layer selected: where(mask, projected, fallback)
init seed=9
'''
    path = tmp_path / 'mask.wgt'
    path.write_text(source)
    built = compile_file(path, tmp_path / 'models', no_train=True)
    module = load_artifact(built.path, warn_untrained=False).module
    x, other = torch.randn(3, 4), torch.randn(3, 4)
    mask = torch.rand(3, 4) > .5
    result = module(x=x, mask=mask, fallback=other)
    torch.testing.assert_close(result['chosen'], torch.where(mask, module.projected(x), other))
    torch.testing.assert_close(result['raw'], module.projected(x))


def test_save_keeps_boundary_and_rejects_rewiring(tmp_path):
    source = tmp_path / 'stream.wgt'
    source.write_text(SOURCE)
    built = compile_file(source, tmp_path / 'models', no_train=True)
    built = load_artifact(built.path, warn_untrained=False)
    with torch.no_grad():
        next(built.module.parameters()).add_(0.2)
    saved = built.save(tmp_path / 'edited')
    assert saved.manifest['format_version'] == 4
    x, state = torch.randn(2, 3, 2), torch.randn(2, 4)
    torch.testing.assert_close(saved.predict({'samples': x, 'old': state})['new'], built.module(x, state)['new'])
    saved.module._port_routes[0] = ('memory', ['input'], 'scan')
    with pytest.raises(WgtError, match='structure'):
        saved.save(tmp_path / 'bad')


def test_named_recur_and_axis_mismatch(tmp_path):
    source = '''wgt 0.3
model refine
input x: f32[features=4]
input other: f32[features=4]
output y: f32[features=4] from result
output original: f32[features=4] from other
block step():
  layer dense: linear(in=4, out=4) -> tanh
layer merged: add(x, other)
layer result: recur(3, step())
init seed=1
'''
    path = tmp_path / 'refine.wgt'
    path.write_text(source)
    module = load(path)
    x, other = torch.randn(2, 4), torch.randn(2, 4)
    expected = x + other
    for _ in range(3):
        expected = module.result.dense(expected)
    torch.testing.assert_close(module(x, other)['y'], expected)
    with pytest.raises(WgtError, match='axis mismatch'):
        parse(source.replace('input other: f32[features=4]', 'input other: f32[channels=4]'))


def test_decompile_keeps_named_boundary_and_scan_wiring(tmp_path):
    from weightlang.decompile import decompile
    path = tmp_path / 'stream.wgt'
    path.write_text(SOURCE)
    built = compile_file(path, tmp_path / 'models', no_train=True)
    recovered = tmp_path / 'recovered.wgt'
    decompile(str(built.path), recovered, references=True)
    model = load(recovered)
    x, h = torch.randn(2, 3, 2), torch.randn(2, 4)
    for name, value in built.module(x, h).items():
        torch.testing.assert_close(model(x, h)[name], value)


def test_scan_preserves_time_axis_without_inventing_hidden_name():
    program = parse(SOURCE.replace('input samples: f32[3,2]', 'input samples: f32[time=3,features=2]'))
    assert program.layers[0].output_spec['axes'] == ['time', None]
