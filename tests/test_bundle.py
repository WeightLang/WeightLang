import json
from pathlib import Path
import zipfile

import pytest
import torch

from weightlang import load, compile_file, WgtError
from weightlang.bundle import build_bundle, install_bundle, METADATA
from weightlang.cli import main

SOURCE = '''wgt 0.3
model demo
input x: f32[2]
output y: f32[1]
layer head: linear(in=2, out=1)
init seed=42
'''


def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'model.wgt').write_text(SOURCE)
    return root


def test_source_build_never_trains_or_allocates_weights(tmp_path, monkeypatch):
    root = project(tmp_path)
    (root / 'model.wgt').write_text(SOURCE + '\ndata data: python="missing.py:load"\ntrain data=data loss=mse epochs=1 python="missing.py:train"\n')
    import weightlang.compiler as compiler
    monkeypatch.setattr(compiler, '_prepare_program', lambda *a, **k: pytest.fail('must not initialize'))
    archive = tmp_path / 'model.wgtpkg'
    build_bundle(root, archive)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'sources/model.wgt', METADATA}
    install_bundle(archive, tmp_path / 'installed')
    assert not (root / 'weights').exists()


def test_multifile_and_embedded_external_roundtrip(tmp_path):
    root = project(tmp_path)
    (root / 'lib').mkdir()
    (root / 'lib/layers.wgt').write_text('wgt 0.3\nblock head(n):\n  layer projection: linear(in=2, out=n)\n')
    (root / 'model.wgt').write_text(SOURCE.replace('model demo', 'use "./lib/layers.wgt" as local\nmodel demo').replace('linear(in=2, out=1)', 'local.head(n=1)'))
    # Actual import syntax is checked by the current parser in the test.
    baseline = compile_file(root / 'model.wgt', tmp_path / 'baseline', no_train=True)
    for mode in ('source', 'embedded', 'external'):
        archive = tmp_path / f'{mode}.wgtpkg'
        kwargs = {} if mode == 'source' else {('weights' if mode == 'embedded' else 'references'): [f'model.wgt={baseline.path}']}
        build_bundle(root, archive, **kwargs)
        target = tmp_path / mode
        install_bundle(archive, target)
        if mode == 'external':
            with pytest.raises(WgtError, match='external weights required'):
                load(target)
        module = load(target, **({'external_weights': baseline.path} if mode == 'external' else {}))
        torch.testing.assert_close(module(torch.ones(1, 2)), baseline.module(torch.ones(1, 2)))
        if mode == 'embedded':
            with zipfile.ZipFile(archive) as z:
                assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist() if i.filename.endswith('.safetensors'))


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', 'a\\b', 'C:/file'])
def test_unsafe_archive_rejected(tmp_path, name):
    path = tmp_path / 'bad.wgtpkg'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, 'bad')
    with pytest.raises(WgtError):
        install_bundle(path, tmp_path / 'dest')
    assert not (tmp_path / 'dest').exists()


def test_corruption_and_overwrite(tmp_path):
    root = project(tmp_path)
    archive = tmp_path / 'model.wgtpkg'
    build_bundle(root, archive)
    with pytest.raises(WgtError):
        build_bundle(root, archive)
    corrupted = tmp_path / 'corrupt.wgtpkg'
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(corrupted, 'w') as dst:
        for name in src.namelist():
            dst.writestr(name, b'bad' if name.endswith('.wgt') else src.read(name))
    with pytest.raises(WgtError, match='hash/size'):
        install_bundle(corrupted, tmp_path / 'dest')
    assert not (tmp_path / 'dest').exists()


def test_train_resumes_repository_and_requires_explicit_fresh(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    (root / 'wgt.toml').write_text('[model]\nsource="model.wgt"\n[train]\npython="train.py:train"\n[revisions]\nmax_versions=0\n')
    (root / 'train.py').write_text('''import torch
from pathlib import Path
def train(ctx):
    Path("before.txt").write_text(str(float(ctx.model.head.bias[0].detach())))
    with torch.no_grad():
        ctx.model.head.bias.add_(1)
    return {"updated": 1}
''')
    assert main(['train']) != 0
    assert main(['train', '--fresh']) == 0
    from weightlang.repository import WeightRepository
    repo = WeightRepository(root / 'weights/.revisions')
    first = repo.resolve('HEAD')
    repo.checkout(first, root / 'first')
    bias = load(root / 'first').head.bias[0].item()
    assert main(['train']) == 0
    second = repo.resolve('HEAD')
    assert second != first
    assert float((root / 'before.txt').read_text()) == bias
    repo.checkout(second, root / 'second')
    assert load(root / 'second').head.bias[0].item() == pytest.approx(bias + 1)
    assert repo.store.read(second)['parent'] == first


def test_cli_build_and_export_do_not_train(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    (root / 'model.wgt').write_text(SOURCE + '\ndata samples: python="missing.py:load"\ntrain data=samples loss=mse epochs=1 python="missing.py:train"\n')
    assert main(['build', '.', '--entry', 'model.wgt']) == 0
    assert not (root / 'weights').exists()
    assert main(['export', 'model.wgt']) == 0
    from weightlang.repository import WeightRepository
    repo = WeightRepository(root / 'weights/.revisions')
    repo.checkout('HEAD', root / 'exported')
    manifest = json.loads((root / 'exported/manifest.json').read_text())
    assert manifest['training'] is None


def test_changed_external_weights_fail_closed(tmp_path):
    root = project(tmp_path)
    result = compile_file(root / 'model.wgt', tmp_path / 'baseline', no_train=True)
    archive = tmp_path / 'external.wgtpkg'
    build_bundle(root, archive, references=[f'model.wgt={result.path}'])
    install_bundle(archive, tmp_path / 'installed')
    (result.path / 'weights.safetensors').write_bytes(b'changed')
    with pytest.raises(WgtError, match='changed'):
        load(tmp_path / 'installed', external_weights=result.path)


def test_metadata_types_and_multi_model_entry(tmp_path):
    root = project(tmp_path)
    (root / 'other.wgt').write_text(SOURCE.replace('model demo', 'model other'))
    with pytest.raises(WgtError, match='multiple models'):
        build_bundle(root, tmp_path / 'out.wgtpkg')
    (root / 'wgt.toml').write_text('[package]\nauthors="not-an-array"\n')
    with pytest.raises(WgtError, match='metadata type'):
        build_bundle(root, tmp_path / 'out.wgtpkg', entry='model.wgt')


def test_failed_task_keeps_head_and_frozen_weights(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    (root / 'model.wgt').write_text(SOURCE.replace('layer head:', 'layer hidden: linear(in=2, out=2)\nlayer head:') + '\nfreeze hidden\n')
    assert main(['export', 'model.wgt']) == 0
    from weightlang.repository import WeightRepository
    repo = WeightRepository(root / 'weights/.revisions')
    before = repo.resolve('HEAD')
    (root / 'wgt.toml').write_text('[model]\nsource="model.wgt"\n[train]\npython="bad.py:train"\n')
    (root / 'bad.py').write_text('import torch\ndef train(ctx):\n    with torch.no_grad():\n        ctx.model.hidden.weight.add_(1)\n')
    assert main(['train']) == 1
    assert repo.resolve('HEAD') == before
    (root / 'model.wgt').write_text((root / 'model.wgt').read_text().replace('freeze hidden', ''))
    assert main(['train']) == 1
    assert repo.resolve('HEAD') == before
