import json

import pytest
import torch

from weightlang import WgtError, load, parse, compile_file, load_artifact
from test_primitives import variants

CODE = '''wgt 0.3
model typed
wtype ternary = [-1, 0, 1]
wtype custom = [-2, -0.25, 0.5, 3]
input x: f32[3]
output y: f32[3]
layer projection: linear(in=3,out=3,bias=0)
weight projection.weight: ternary axis=0 start=0 stop=1
weight projection.weight: custom axis=0 start=1 stop=2
init seed=7
'''


def test_mixture_forward_gradient_optimizer_and_all_roundtrips(tmp_path):
    for model in variants(tmp_path, CODE):
        master = model.get_parameter('projection.weight')
        with torch.no_grad():
            master.copy_(torch.tensor([[-0.5,0.2,0.8],[-1.5,0.125,2.5],[0.1,0.2,0.3]]))
        expected = torch.tensor([[-1.,0.,1.],[-2.,-0.25,3.],[0.1,0.2,0.3]])
        x = torch.eye(3, requires_grad=True)
        actual = model(x)['y']
        torch.testing.assert_close(actual,expected.T)
        actual.sum().backward()
        torch.testing.assert_close(master.grad,torch.ones_like(master))
        torch.testing.assert_close(x.grad,expected.sum(0).expand(3,3))
        before = master.detach().clone()
        torch.optim.SGD(model.parameters(),lr=0.1).step()
        torch.testing.assert_close(master,before-0.1)
        assert list(model.state_dict()) == ['projection.weight']
        torch.testing.assert_close(model.state_dict()['projection.weight'],master)


def test_saved_updated_master_and_manifest_validation(tmp_path):
    source = tmp_path/'model.wgt'; source.write_text(CODE)
    built = compile_file(source,tmp_path/'built',no_train=True)
    artifact = load_artifact(built.path,warn_untrained=False)
    with torch.no_grad():
        artifact.module.projection.weight.fill_(0.8)
    restored = artifact.save(tmp_path/'saved')
    torch.testing.assert_close(restored.module(torch.eye(3))['y'],artifact.module(torch.eye(3))['y'])
    manifest_path = restored.path/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['interface']['weight_assignments'][0]['stop'] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WgtError, match='slice'):
        load_artifact(restored.path,warn_untrained=False)


@pytest.mark.parametrize('declaration',[
    'wtype bad = [0, 0]',
    'wtype bad = [1e100]',
    'wtype bad = [1, 1.000000001]',
    'weight projection.weight: missing',
    'weight missing.weight: ternary',
    'weight projection.weight: ternary',
    'weight projection.weight: ternary axis=1 start=0 stop=1',
    'weight projection.weight: ternary axis=0 start=2 stop=4',
    'weight projection.weight: ternary axis=0',
])
def test_invalid_declarations(declaration):
    with pytest.raises(WgtError):
        parse(CODE+declaration+'\n',check_training=False)


def test_shared_alias_domain_and_gradients(tmp_path):
    code = '''wgt 0.3
model shared
wtype binary = [-1, 1]
input x: f32[2]
output y: f32[2]
layer a: linear(in=2,out=2,bias=0)
layer b: linear(in=2,out=2,bias=0)
parameter shared = a.weight
share b.weight = shared
weight a.weight: binary
init seed=7
'''
    for model in variants(tmp_path,code):
        assert model.a.weight is model.b.weight
        with torch.no_grad():
            model.a.weight.fill_(0.2)
        output = model(torch.ones(2))['y']
        torch.testing.assert_close(output,torch.full((2,),4.))
        output.sum().backward()
        torch.testing.assert_close(model.a.weight.grad,torch.full((2,2),4.))


def test_parent_direct_parameter_read_and_frozen_master():
    from weightlang.weight_types import attach_weight_types
    class DirectRead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(2,2,bias=False)
        def forward(self,x):
            return torch.nn.functional.linear(x,self.child.weight)
    model = DirectRead()
    with torch.no_grad():
        model.child.weight.fill_(0.2)
    model.child.weight.requires_grad_(False)
    attach_weight_types(model,{'binary':[-1,1]},[{'target':'child.weight','type':'binary'}])
    x = torch.ones(2,requires_grad=True)
    y = model(x)
    torch.testing.assert_close(y,torch.full((2,),2.))
    y.sum().backward()
    torch.testing.assert_close(x.grad,torch.full((2,),2.))
    assert model.child.weight.grad is None
    torch.testing.assert_close(model.child.weight,torch.full((2,2),0.2))


def test_tied_embedding_noncontiguous_and_cuda(tmp_path):
    code = '''wgt 0.3
model tokens
wtype binary = [-1, 1]
input x: i64[2]
output y: f32[2,3]
layer embed: embedding(vocab=3,dim=2)
layer head: linear(in=2,out=3,bias=0,tie_embedding=1)
weight head.weight: binary
init seed=7
'''
    for model in variants(tmp_path,code):
        devices = ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            model.to(device)
            assert model.embed.weight is model.head.weight
            with torch.no_grad():
                model.head.weight.fill_(0.2)
            x = torch.tensor([[0,1],[1,2]],device=device).T
            result = model(x)['y']
            torch.testing.assert_close(result,torch.full((2,2,3),2.,device=device))
            result.sum().backward()
            assert model.head.weight.grad is not None
            model.zero_grad()


def test_attention_projection_matches_explicit_quantized_model(tmp_path):
    code = '''wgt 0.3
model attention
wtype custom = [-0.5, 0, 0.25]
input x: f32[2,4]
output y: f32[2,4]
layer attn: self_attention(dim=4,heads=2)
weight attn.qkv.weight: custom
weight attn.proj.weight: custom
init seed=7
'''
    reference = load_source(tmp_path,code.replace('weight attn.qkv.weight: custom\n','').replace('weight attn.proj.weight: custom\n',''))
    for model in variants(tmp_path,code):
        reference.load_state_dict(model.state_dict())
        with torch.no_grad():
            for name in ['attn.qkv.weight','attn.proj.weight']:
                parameter = reference.get_parameter(name)
                book = torch.tensor([-0.5,0.,0.25])
                parameter.copy_(book[(parameter[...,None]-book).abs().argmin(-1)])
        x = torch.randn(2,2,4)
        torch.testing.assert_close(model(x)['y'],reference(x)['y'])


def load_source(tmp_path,code):
    path = tmp_path/'reference.wgt'; path.write_text(code)
    return load(path)


def test_save_rejects_runtime_domain_edits(tmp_path):
    path = tmp_path/'model.wgt'; path.write_text(CODE)
    built = compile_file(path,tmp_path/'built',no_train=True)
    artifact = load_artifact(built.path,warn_untrained=False)
    artifact.module.projection._weight_rules['weight'][0]['values'] = [-10,10]
    with pytest.raises(WgtError,match='structure'):
        artifact.save(tmp_path/'changed')


def test_constant_expressions_and_explicit_plus():
    program = parse(CODE.replace('wtype ternary = [-1, 0, 1]',
        'const S = 2\nwtype ternary = [-S, 0, +S / 2]'), check_training=False)
    assert program.weight_types['ternary'] == [-2.,0.,1.]


def test_old_format_cannot_silently_drop_domains(tmp_path):
    path = tmp_path/'model.wgt'; path.write_text(CODE)
    built = compile_file(path,tmp_path/'built',no_train=True)
    manifest_path = built.path/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['format_version'] = 6
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WgtError,match='format_version 7'):
        load_artifact(built.path,warn_untrained=False)
