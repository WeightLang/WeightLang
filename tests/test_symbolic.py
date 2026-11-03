import importlib.util
import json

import pytest
import torch

from weightlang import WgtError, compile_file, export_python, load, load_artifact, parse
from weightlang.decompile import decompile

SOURCE = '''wgt 0.3
model variable_scan
dim T min=1 max=8
dim P min=1 max=4
require T % 2 == 0
require P + T <= 8
input samples: f32[time=T,features=2]
input old: f32[4]
input prefix: bool[P]
input mask: bool[P+T]
output states: f32[time=T,4] from memory
output new: f32[4] from final
output checked_mask: bool[P+T] from mask
block cell():
  layer joined: concat(input, state)
  layer next: linear(in=6,out=4) -> tanh
layer memory: scan(hidden=4,cell()) from samples, old
layer final: last from memory
init seed=7
'''


def call(model, x, state, past=1):
    return model(samples=x, old=state,
                 prefix=torch.ones(x.shape[0], past, dtype=torch.bool),
                 mask=torch.ones(x.shape[0], past+x.shape[1], dtype=torch.bool))


def test_variable_scan_full_chunk_values_and_gradients(tmp_path):
    path = tmp_path / 'model.wgt'
    path.write_text(SOURCE)
    model = load(path)
    x, h = torch.randn(2,6,2, requires_grad=True), torch.randn(2,4, requires_grad=True)
    full = call(model,x,h)
    a = call(model,x[:,:2],h)
    b = call(model,x[:,2:],a['new'])
    chunked = torch.cat([a['states'],b['states']],1)
    torch.testing.assert_close(chunked,full['states'])
    params = (*model.parameters(),x,h)
    expected = torch.autograd.grad(full['states'].square().sum(),params,retain_graph=True)
    actual = torch.autograd.grad(chunked.square().sum(),params)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b)
    assert full['checked_mask'].shape == (2,7)


def test_roundtrip_export_save_and_decompile_keep_symbolic_contract(tmp_path):
    path = tmp_path / 'model.wgt'
    path.write_text(SOURCE)
    built = compile_file(path,tmp_path/'models',no_train=True)
    assert built.manifest['format_version'] == 5
    restored = load_artifact(built.path,warn_untrained=False)
    saved = restored.save(tmp_path/'saved')
    assert saved.manifest['format_version'] == 5
    target = tmp_path/'standalone.py'
    export_python(built.path,target)
    spec = importlib.util.spec_from_file_location('symbolic_export',target)
    exported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported)
    exported_model = exported.load_model()
    decompiled = tmp_path/'recovered.wgt'
    decompile(str(built.path),decompiled,references=True)
    recovered = load(decompiled)
    for size in (2,4,6):
        x,h = torch.randn(2,size,2),torch.randn(2,4)
        expected = call(restored.module,x,h)
        for model in (saved.module, exported_model, recovered):
            for key, value in call(model,x,h).items():
                torch.testing.assert_close(value,expected[key])
    with pytest.raises(ValueError, match='constraint'):
        call(exported_model,torch.randn(2,3,2),torch.zeros(2,4))


@pytest.mark.parametrize('old,new',[
    ('min=1 max=8','min=9 max=8'),
    ('min=1 max=8','min=1 max=4097'),
    ('require T % 2 == 0','require T % 0 == 0'),
    ('require T % 2 == 0','require T / 2 == 0'),
    ('require T % 2 == 0','require T > 99'),
    ('input prefix: bool[P]','input prefix: bool[P+1]'),
    ('output states: f32[time=T,4]','output states: f32[time=T+1,4]'),
    ('input mask: bool[P+T]','input mask: bool[P-T]'),
    ('require P + T <= 8','require P * 2147483647 <= 8'),
])
def test_invalid_contracts(old,new):
    with pytest.raises(WgtError):
        parse(SOURCE.replace(old,new))


def test_runtime_rejects_before_any_layer_executes(tmp_path):
    path = tmp_path/'model.wgt'
    path.write_text(SOURCE)
    model = load(path)
    executed = []
    handle = model.memory.register_forward_pre_hook(lambda *_: executed.append(True))
    with pytest.raises(ValueError,match='constraint'):
        call(model,torch.randn(2,3,2),torch.zeros(2,4))
    with pytest.raises(ValueError,match='range'):
        call(model,torch.randn(2,10,2),torch.zeros(2,4))
    with pytest.raises(ValueError,match='port mask'):
        model(torch.randn(2,4,2), torch.zeros(2,4), torch.ones(2,1,dtype=torch.bool), torch.ones(2,4,dtype=torch.bool))
    assert not executed
    handle.remove()


def test_wrong_shape_away_from_representative_is_rejected():
    source = '''wgt 0.3
model bad
dim T min=1 max=4
input x: f32[T,4]
output y: f32[1,4] from head
layer head: linear(in=4,out=4)
init seed=1
'''
    with pytest.raises(WgtError,match='mismatch'):
        parse(source)
    with pytest.raises(WgtError):
        parse(source.replace('f32[T,4]','f32[4,T]').replace('f32[1,4]','f32[4,4]').replace('linear(in=4,out=4)','linear(out=4)'))


def test_manifest_tampering_and_version_downgrade_fail(tmp_path):
    path = tmp_path/'model.wgt'
    path.write_text(SOURCE)
    built = compile_file(path,tmp_path/'models',no_train=True)
    manifest = built.path/'manifest.json'
    original = manifest.read_text()
    for mutate in (
        lambda m: m.update(format_version=4),
        lambda m: m['interface']['outputs'][0].update(shape_expr=[1,4]),
        lambda m: m['interface'].update(constraints=[['==','T',999]]),
        lambda m: m['interface']['inputs'][0].update(shape_expr=['unknown',2]),
    ):
        record = json.loads(original)
        mutate(record)
        manifest.write_text(json.dumps(record))
        with pytest.raises(WgtError):
            load_artifact(built.path,warn_untrained=False)


def test_variable_attention_and_unbatched_boundary(tmp_path):
    source = '''wgt 0.3
model attention
dim T min=1 max=6
input x: f32[T,4]
output y: f32[T,4]
layer attn: self_attention(dim=4,heads=2,causal=1)
init seed=4
'''
    path = tmp_path/'attention.wgt'
    path.write_text(source)
    model = load(path)
    x = torch.randn(6,4,requires_grad=True)
    torch.testing.assert_close(model(x)['y'],model(x.unsqueeze(0))['y'][0])
    torch.testing.assert_close(model(x[:3])['y'],model(x)['y'][:3])
    model(x)['y'].square().sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in model.parameters())


def test_integer_constraints_equal_constant_configuration():
    source = SOURCE.replace('require T % 2 == 0','const HEADS = 6\nconst KV_HEADS = 2\nrequire HEADS % KV_HEADS == 0\nrequire T % 2 == 0')
    assert parse(source).input.shape == (2,2)
    with pytest.raises(WgtError,match='no legal assignment'):
        parse(source.replace('const HEADS = 6','const HEADS = 5'))


@pytest.mark.parametrize('operation',['transpose(features,time)','mean(axis=time)','sum(axis=time)'])
def test_axis_sugar_cannot_bypass_fixed_plan_restriction(operation):
    source = f'''wgt 0.3
model fixed_plan
dim T min=1 max=4
input x: f32[time=T,features=4]
output y: f32[4] from changed
layer head: linear(in=4,out=4)
layer changed: {operation}
init seed=1
'''
    with pytest.raises(WgtError,match='does not yet support bounded dimensions'):
        parse(source)
