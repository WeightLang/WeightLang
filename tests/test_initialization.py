import pytest
import torch

from weightlang import WgtError, load, compile_file, load_artifact
from weightlang.syntax import parse_file
from weightlang.bundle import build_bundle, install_bundle

SOURCE = '''wgt 0.3
model custom
input x: f32[3]
output y: f32[3]
layer head: linear(3)
init file="init.wgt"
'''


def model(tmp_path, rules='tensor "*.weight" eye\ntensor "*.bias" zeros\n'):
    source = tmp_path / 'model.wgt'
    source.write_text(SOURCE)
    (tmp_path / 'init.wgt').write_text('wgt init 1\n' + rules)
    return source


def test_identity_forward_backward_artifact_and_rng(tmp_path, monkeypatch):
    source = model(tmp_path)
    rng = torch.random.get_rng_state().clone()
    # Meta constructors may describe defaults; no real random default allocation.
    original = torch.nn.init.kaiming_uniform_
    def guarded(tensor, *args, **kwargs):
        assert tensor.device.type == 'meta'
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.nn.init, 'kaiming_uniform_', guarded)
    module = load(source)
    assert torch.equal(rng, torch.random.get_rng_state())
    x = torch.tensor([[1., 2., 3.]], requires_grad=True)
    torch.testing.assert_close(module(x), x)
    module(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    artifact = compile_file(source, tmp_path / 'out', no_train=True)
    assert artifact.manifest['init']['method'] == 'wgt-rules'
    assert len(artifact.manifest['init']['sha256']) == 64
    monkeypatch.undo()
    (tmp_path / 'init.wgt').unlink()
    torch.testing.assert_close(load_artifact(artifact.path).module(x), x)


@pytest.mark.parametrize('rows,cols', [(3, 3), (5, 3), (3, 5)])
def test_dct_is_deterministic_orthogonal(tmp_path, rows, cols):
    source = model(tmp_path, 'tensor "*.weight" dct gain=2\ntensor "*.bias" zeros\n')
    source.write_text(SOURCE.replace('input x: f32[3]', f'input x: f32[{cols}]')
                      .replace('output y: f32[3]', f'output y: f32[{rows}]')
                      .replace('linear(3)', f'linear({rows})'))
    weight = load(source).head.weight.detach()
    gram = weight @ weight.T if rows <= cols else weight.T @ weight
    torch.testing.assert_close(gram, 4 * torch.eye(min(rows, cols)))
    source.write_text(source.read_text().replace('init file=', 'init seed=17 file='))
    assert torch.equal(weight, load(source).head.weight)


def test_python_static_check_no_execution_then_full_generation(tmp_path):
    source = model(tmp_path)
    source.write_text(SOURCE.replace('file="init.wgt"', 'python="generate.py" seed=9'))
    parse_file(source)  # The Python file does not even exist yet.
    (tmp_path / 'generate.py').write_text('''import torch
def initialize(context):
    assert context.seed == 9
    return {s.name: torch.full(s.shape, 2., dtype=s.dtype) for s in context.tensors}
''')
    module = load(source)
    torch.testing.assert_close(module(torch.ones(1, 3)), torch.full((1, 3), 8.))
    built = compile_file(source, tmp_path / 'saved', no_train=True)
    (tmp_path / 'generate.py').unlink()
    torch.testing.assert_close(load(built.path)(torch.ones(1, 3)), torch.full((1, 3), 8.))
    assert built.manifest['init']['method'] == 'python'


@pytest.mark.parametrize('expression', [
    '{}', '{s.name: torch.zeros(1) for s in context.tensors}',
    '{s.name: torch.full(s.shape, float("nan")) for s in context.tensors}',
    '{s.name: torch.zeros(s.shape, dtype=torch.float64) for s in context.tensors}',
])
def test_python_rejects_invalid_results(tmp_path, expression):
    source = model(tmp_path)
    source.write_text(SOURCE.replace('file="init.wgt"', 'python="bad.py"'))
    (tmp_path / 'bad.py').write_text(f'import torch\ndef initialize(context):\n    return {expression}\n')
    with pytest.raises(WgtError, match='initializ'):
        load(source)


@pytest.mark.parametrize('rules', [
    'tensor "*.weight" eye',
    'tensor "missing" zeros',
    'tensor "*" eye',
    'tensor "*" normal std=-1',
    'tensor "*" zeros value=2',
    'tensor "*" constant value=nan',
])
def test_invalid_rules_rejected(tmp_path, rules):
    with pytest.raises(WgtError):
        load(model(tmp_path, rules))


def test_rules_bundle_roundtrip_and_python_requires_weights(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    source = model(root)
    archive = tmp_path / 'source.wgtpkg'
    build_bundle(root, archive)
    installed = tmp_path / 'installed'
    install_bundle(archive, installed)
    torch.testing.assert_close(load(installed)(torch.eye(3)), torch.eye(3))
    source.write_text(SOURCE.replace('file="init.wgt"', 'python="missing.py"'))
    with pytest.raises(WgtError, match='Python initialization'):
        build_bundle(root, tmp_path / 'python.wgtpkg')


def test_seeded_algorithms_restore_rng(tmp_path):
    source = model(tmp_path, 'tensor "*.weight" orthogonal gain=0.5\ntensor "*.bias" normal std=0.1\n')
    rng = torch.random.get_rng_state().clone()
    a, b = load(source), load(source)
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name])
    torch.testing.assert_close(a.head.weight @ a.head.weight.T, 0.25 * torch.eye(3), atol=1e-6, rtol=1e-6)


def test_version_duplicate_and_mixed_sources(tmp_path):
    source = model(tmp_path)
    for text in [SOURCE.replace('0.3', '0.2'), SOURCE + 'init seed=1\n',
                 SOURCE.replace('init file=', 'init python="gen.py" file=')]:
        source.write_text(text)
        with pytest.raises(WgtError):
            parse_file(source)


def test_tied_weights_preserved_and_conflicting_rules_rejected(tmp_path):
    source = model(tmp_path, 'tensor "*" dct\n')
    source.write_text('wgt 0.3\nmodel tied\ninput tokens: i64[2]\noutput logits: f32[2, 5]\nlayer embed: embedding(vocab=5, dim=3)\nlayer head: linear(in=3, out=5, bias=0, tie_embedding=1)\ninit file="init.wgt"\n')
    module = load(source)
    assert module.embed.weight is module.head.weight
    built = compile_file(source, tmp_path / 'saved', no_train=True)
    restored = load(built.path)
    assert restored.embed.weight is restored.head.weight
    (tmp_path / 'init.wgt').write_text('wgt init 1\ntensor "*" dct\ntensor "head.weight" zeros\n')
    with pytest.raises(WgtError, match='conflicting tied'):
        load(source)


def test_import_overrides_initializer(tmp_path):
    source = model(tmp_path)
    baseline = compile_file(source, tmp_path / 'base', no_train=True)
    (tmp_path / 'init.wgt').write_text('wgt init 1\ntensor "*" constant value=7\n')
    source.write_text(SOURCE + f'\nimport head from "{baseline.path}"\n')
    torch.testing.assert_close(load(source)(torch.eye(3)), torch.eye(3))
