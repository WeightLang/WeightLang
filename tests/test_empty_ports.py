import importlib.util

import pytest
import torch

from weightlang import WgtError, compile_file, export_python, load, load_artifact, parse
from weightlang.decompile import decompile

SCALAR = '''wgt 0.3
model scalar_ports
input x: f32[features=4]
input gain: f32[]
output loss: f32[] from scaled
layer head: linear(in=4,out=4)
layer total: reduce("features ->",op=sum)
layer scaled: mul(total,gain)
init seed=7
'''
EMPTY = '''wgt 0.3
model empty_prefix
dim P min=0 max=4
input prefix: f32[P,2]
input initial: f32[3]
output values: f32[P,3] from memory
output retained: f32[3] from initial
block cell():
  layer combined: concat(input,state)
  layer next: linear(in=5,out=3) -> tanh
layer memory: scan(hidden=3,cell()) from prefix,initial
init seed=7
'''


def models(tmp_path, source):
    path = tmp_path / 'model.wgt'
    path.write_text(source)
    model = load(path)
    artifact = compile_file(path, tmp_path/'compiled', no_train=True)
    restored = load_artifact(artifact.path, warn_untrained=False)
    target = tmp_path/'export.py'
    export_python(artifact.path, target)
    spec = importlib.util.spec_from_file_location('empty_export', target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovered = tmp_path/'recovered.wgt'
    decompile(str(artifact.path), recovered, references=True)
    return (model, restored.module, module.load_model(), load(recovered))


def test_scalar_ports_reduction_batch_and_gradient(tmp_path):
    variants = models(tmp_path, SCALAR)
    for batched in (False, True):
        x = torch.randn((2,4) if batched else (4,), requires_grad=True)
        gain = torch.randn((2,) if batched else (), requires_grad=True)
        reference = None
        for model in variants:
            out = model(x, gain)['loss']
            expected = model.head(x).sum(-1) * gain
            torch.testing.assert_close(out, expected)
            actual = torch.autograd.grad(out.sum(), (x,gain), retain_graph=True)
            wanted = torch.autograd.grad(expected.sum(), (x,gain))
            for a,b in zip(actual, wanted):
                torch.testing.assert_close(a,b)
            if reference is not None:
                torch.testing.assert_close(out, reference)
            reference = out
    with pytest.raises(ValueError, match='batch'):
        variants[0](torch.randn(2,4), torch.tensor(1.))


def test_empty_scan_roundtrip_and_autograd(tmp_path):
    for model in models(tmp_path, EMPTY):
        for batched in (False, True):
            x = torch.empty((2,0,2) if batched else (0,2), requires_grad=True)
            state = torch.randn((2,3) if batched else (3,), requires_grad=True)
            out = model(x,state)
            assert out['values'].shape == ((2,0,3) if batched else (0,3))
            torch.testing.assert_close(out['retained'],state)
            gx, gs = torch.autograd.grad(out['values'].sum()+out['retained'].sum(), (x,state))
            assert gx.shape == x.shape
            torch.testing.assert_close(gs,torch.ones_like(state))
            assert set(model.state_dict()) == {'memory.next.step0.weight','memory.next.step0.bias'}


@pytest.mark.parametrize('operation',['last','meanpool','softmax','self_attention(dim=2,heads=1,causal=0)'])
def test_empty_undefined_operations_rejected(operation):
    source = EMPTY.replace('output values: f32[P,3] from memory','output values: f32[2] from invalid')
    source += f'layer invalid: {operation} from prefix\n'
    with pytest.raises(WgtError, match='empty sample'):
        parse(source)


def test_scalar_feature_operator_rejected():
    source = SCALAR.replace('layer scaled: mul(total,gain)','layer scaled: linear(out=1) from gain')
    with pytest.raises(WgtError, match='non-scalar'):
        parse(source)
