import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weightlang import WgtError, parse
from weightlang.compiler import compile_file
from weightlang.cli import main

HEADER = '''wgt 0.3
model external
input x: f32[1]
output y: f32[1]
layer fit: linear(out=1)
init seed=42
'''
ROWS = {"inputs": [[0], [1]], "targets": [[1], [3]]}
TRAIN = 'train data=samples loss=mse optimizer=sgd lr=0.1 epochs=20\n'


def pin(path, **options):
    return f'data samples: file="{path.name}" sha256={hashlib.sha256(path.read_bytes()).hexdigest()} ' + ' '.join(f'{k}="{v}"' for k, v in options.items()) + '\n'


def build(tmp_path, text, **kwargs):
    source = tmp_path / 'model.wgt'
    source.write_text(HEADER + text)
    return compile_file(source, tmp_path / 'out', **kwargs)


@pytest.mark.parametrize('fmt', ['csv', 'tsv', 'json', 'jsonl', 'npz', 'wgt'])
def test_formats_train_and_reload(tmp_path, fmt):
    path = tmp_path / f'data.{fmt}'
    if fmt == 'csv':
        path.write_text('"0","1"\n1,3\n')
    elif fmt == 'tsv':
        path.write_text('0\t1\n1\t3\n')
    elif fmt == 'json':
        path.write_text(json.dumps(ROWS))
    elif fmt == 'jsonl':
        path.write_text('\n'.join(json.dumps(dict(inputs=x, targets=y)) for x, y in zip(ROWS['inputs'], ROWS['targets'])))
    elif fmt == 'npz':
        np.savez(path, **ROWS)
    else:
        path.write_text('wgt 0.3\ndata samples:\n  [0] -> [1]\n  [1] -> [3]\n')
        assert main(['check', str(path)]) == 0
    result = build(tmp_path, pin(path) + TRAIN)
    record = result.manifest['training']
    assert record['final_loss'] < record['initial_loss']
    from weightlang import load_artifact
    assert load_artifact(result.path).predict([[0]]).shape == (1, 1)


def test_python_callbacks_deferred_and_trained(tmp_path):
    plugin = tmp_path / 'callbacks.py'
    plugin.write_text('''import torch
from pathlib import Path
Path(__file__).with_suffix('.ran').touch()
def load(ctx):
    assert ctx.input_shape == (1,)
    assert ctx.source_dir == Path(__file__).parent
    return {"inputs": [[0], [1]], "targets": [[1], [3]]}
def train(ctx):
    with torch.no_grad():
        ctx.module.fit.weight.fill_(2)
        ctx.module.fit.bias.fill_(1)
    return {"updates": 1}
''')
    text = 'data samples: python="callbacks.py"\ntrain data=samples loss=mse epochs=1 python="callbacks.py:train"\n'
    parse(HEADER + text, str(tmp_path / 'model.wgt'))
    assert not plugin.with_suffix('.ran').exists()
    result = build(tmp_path, text, no_train=True)
    assert result.manifest['training'] is None
    assert not plugin.with_suffix('.ran').exists()
    result = build(tmp_path, text, force=True)
    assert result.manifest['training']['final_loss'] == 0
    assert result.manifest['training']['dataset']['loader']['sha256']
    assert result.manifest['training']['trainer']['sha256']


def test_installed_entrypoints(tmp_path, monkeypatch):
    from importlib import metadata
    def load(ctx):
        return ROWS
    def train(ctx):
        import torch
        with torch.no_grad():
            ctx.module.fit.weight.fill_(2)
            ctx.module.fit.bias.fill_(1)
    calls = []
    def entries(*, group, name):
        calls.append((group, name))
        return [SimpleNamespace(load=lambda: load if group.endswith('data_loaders') else train,
                                value='package:callback', dist=SimpleNamespace(name='package', version='1'))]
    monkeypatch.setattr(metadata, 'entry_points', entries)
    text = 'data samples: entrypoint="fixture"\ntrain data=samples loss=mse epochs=1 entrypoint="fit"\n'
    parse(HEADER + text)
    assert calls == []
    result = build(tmp_path, text)
    assert result.manifest['training']['final_loss'] == 0
    assert calls == [('weightlang.data_loaders', 'fixture'), ('weightlang.trainers', 'fit')]


@pytest.mark.parametrize('value,error', [
    ({'inputs': [[1]], 'targets': []}, 'equal row counts'),
    ({'inputs': [[float('nan')]], 'targets': [[1]]}, 'finite'),
    ({'inputs': [[1, 2]], 'targets': [[1]]}, 'rows must'),
    ({'inputs': [], 'targets': []}, 'no rows'),
])
def test_invalid_loader_results(tmp_path, value, error):
    plugin = tmp_path / 'bad.py'
    plugin.write_text('import json\ndef load(ctx):\n    return json.loads(' + repr(json.dumps(value)) + ')\n')
    with pytest.raises(WgtError, match=error):
        build(tmp_path, 'data samples: python="bad.py"\n' + TRAIN)
    assert not (tmp_path / 'out').exists()


def test_nested_relative_data_and_selection(tmp_path):
    sub = tmp_path / 'nested'
    sub.mkdir()
    payload = sub / 'rows.json'
    payload.write_text(json.dumps(ROWS))
    data = sub / 'data.wgt'
    data.write_text('wgt 0.3\n' + pin(payload).replace('data samples:', 'data chosen:'))
    text = pin(data, dataset='chosen').replace('file="data.wgt"', 'file="nested/data.wgt"')
    result = build(tmp_path, text + TRAIN)
    assert result.manifest['training']['rows'] == 2
    payload.write_text('{}')
    with pytest.raises(WgtError, match='source pins'):
        parse(HEADER + text + TRAIN, str(tmp_path / 'model.wgt'))


@pytest.mark.parametrize('statement', [
    'data samples: python="x.py" entrypoint="x"',
    'data samples: python="x.py" file="x.csv"',
    'data samples: format=json',
    'data samples: python=""',
    'train data=samples loss=mse epochs=1 python="x.py" trainer=x',
    'train data=samples loss=mse epochs=1 python="x.py" entrypoint=x',
])
def test_exclusive_sources(statement):
    with pytest.raises(WgtError):
        parse(HEADER + statement + '\n')


@pytest.mark.parametrize('count', [0, 2])
def test_missing_or_ambiguous_entrypoint(tmp_path, monkeypatch, count):
    from importlib import metadata
    monkeypatch.setattr(metadata, 'entry_points', lambda **kw: [object()] * count)
    with pytest.raises(WgtError, match='expected one'):
        build(tmp_path, 'data samples: entrypoint="missing"\n' + TRAIN)


def test_explicit_format_and_skip(tmp_path):
    path = tmp_path / 'rows.dat'
    path.write_text('header\n0\t1\n1\t3\n')
    result = build(tmp_path, pin(path, format='tsv', skip='1') + TRAIN)
    assert result.manifest['training']['rows'] == 2


def test_no_train_skips_missing_external_and_scripts(tmp_path):
    result = build(tmp_path, 'data samples: file="absent.wgt" sha256=' + '0' * 64 + '\n'
                   'train data=samples loss=mse epochs=1 python="absent.py"\n', no_train=True)
    assert result.manifest['training'] is None


def test_data_file_rejects_model_and_missing_selection(tmp_path):
    path = tmp_path / 'data.wgt'
    path.write_text('wgt 0.3\nmodel forbidden\n')
    with pytest.raises(WgtError, match='only allow'):
        build(tmp_path, pin(path) + TRAIN)
    path.write_text('wgt 0.3\ndata other:\n  [0] -> [1]\n')
    with pytest.raises(WgtError, match='no dataset'):
        build(tmp_path, pin(path) + TRAIN)


def test_real_package_metadata_entrypoints(tmp_path, monkeypatch):
    package = tmp_path / 'wgt_fixture_hooks.py'
    package.write_text('''import torch
def load(context):
    return {"inputs": [[0], [1]], "targets": [[1], [3]]}
def train(context):
    with torch.no_grad():
        context.module.fit.weight.fill_(2)
        context.module.fit.bias.fill_(1)
''')
    dist = tmp_path / 'wgt_fixture_hooks-1.0.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: wgt-fixture-hooks\nVersion: 1.0\n')
    (dist / 'entry_points.txt').write_text(
        '[weightlang.data_loaders]\nfixture_rows = wgt_fixture_hooks:load\n'
        '[weightlang.trainers]\nfixture_fit = wgt_fixture_hooks:train\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    result = build(tmp_path, 'data samples: entrypoint="fixture_rows"\n'
                   'train data=samples loss=mse epochs=1 entrypoint="fixture_fit"\n')
    record = result.manifest['training']
    assert record['final_loss'] == 0
    assert record['trainer']['entrypoint']['version'] == '1.0'
    assert record['dataset']['loader']['distribution'] == 'wgt-fixture-hooks'


def test_documented_examples_execute(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'examples' / 'external_data'
    for name in ('model.wgt', 'python.wgt'):
        result = compile_file(root / name, tmp_path / 'examples')
        assert result.manifest['training']['final_loss'] < result.manifest['training']['initial_loss']


def test_standalone_data_cannot_execute_python(tmp_path):
    path = tmp_path / 'data.wgt'
    path.write_text('wgt 0.3\ndata samples: python="loader.py"\n')
    with pytest.raises(WgtError, match='model file'):
        build(tmp_path, pin(path) + TRAIN)
