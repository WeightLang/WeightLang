import importlib.util

import pytest
import torch

from weightlang import WgtError, compile_file, export_python, load, load_artifact, parse
from weightlang.decompile import decompile


def source(ports, shape, layers, dtype='f32', dims=''):
    return f'wgt 0.3\nmodel atoms\n{dims}' + '\n'.join(f'input {n}: {t}[{s}]' for n,t,s in ports) + f'\noutput y: {dtype}[{shape}] from result\n{layers}\ninit seed=7\n'


def variants(tmp_path, code):
    path = tmp_path/'model.wgt'
    path.write_text(code)
    built = compile_file(path,tmp_path/'built',no_train=True)
    target = tmp_path/'export.py'
    export_python(built.path,target)
    spec = importlib.util.spec_from_file_location('primitive_export',target)
    exported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported)
    recovered = tmp_path/'recovered.wgt'
    decompile(str(built.path),recovered,references=True)
    return [load(path),load_artifact(built.path,warn_untrained=False).module,exported.load_model(),load(recovered)]


def test_cross_attention_composition_noncontiguous_gradient_and_all_mask(tmp_path):
    code = source([('q','f32','T,4'),('k','f32','P,4'),('v','f32','P,3'),('mask','bool','T,P')], 'T,3', '''layer kt: transpose_axes(axis1=0,axis2=1) from k
layer scores: matmul from q,kt
layer scaled: scale(factor=0.5)
layer probabilities: masked_softmax(axis=-1) from scaled,mask
layer result: matmul from probabilities,v''',dims='dim T min=1 max=3\ndim P min=0 max=5\n')
    for model in variants(tmp_path,code):
        for p in (0,5):
            q = torch.randn(2,4,3).transpose(1,2).requires_grad_()
            k = torch.randn(2,4,p).transpose(1,2).requires_grad_()
            v = torch.randn(2,p,3,requires_grad=True)
            mask = torch.ones(2,3,p,dtype=torch.bool)
            mask[:,0] = False
            out = model(q,k,v,mask)['y']
            logits = (q @ k.transpose(-1,-2)) * 0.5
            safe = logits.masked_fill(~mask,-torch.inf)
            safe = torch.where(mask.any(-1,keepdim=True),safe,torch.zeros_like(safe))
            expected = torch.softmax(safe,-1).masked_fill(~mask,0) @ v
            torch.testing.assert_close(out,expected)
            assert torch.isfinite(out).all() and torch.count_nonzero(out[:,0]) == 0
            ga = torch.autograd.grad(out.square().sum(),(q,k,v),retain_graph=True)
            gb = torch.autograd.grad(expected.square().sum(),(q,k,v))
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b)
                assert torch.isfinite(a).all()


def test_broadcast_cat_slice_stack_and_scalar_reduction(tmp_path):
    code = source([('x','f32','1,3'),('target','bool','2,3')], '', '''layer expanded: broadcast from x,target
layer doubled: cat(axis=0) from expanded,expanded
layer selected: slice(axis=0,start=0,stop=4,step=2)
layer stacked: stack(axis=-1) from selected,expanded
layer summed: reduce_sum(axis=-1)
layer averaged: reduce_mean(axis=0)
layer result: reduce_max(axis=0)''')
    for model in variants(tmp_path,code):
        x = torch.tensor([[[1.,2.,3.]],[[2.,4.,6.]]],requires_grad=True)
        target = torch.zeros(2,2,3,dtype=torch.bool)
        out = model(x,target)['y']
        expected = 2*x[:,0].amax(-1)
        torch.testing.assert_close(out,expected)
        ga = torch.autograd.grad(out.sum(),x,retain_graph=True)[0]
        gb = torch.autograd.grad(expected.sum(),x)[0]
        torch.testing.assert_close(ga,gb)


def test_gather_scatter_duplicate_indices_and_gradient(tmp_path):
    code = source([('base','f32','4'),('indices','i64','3'),('values','f32','3')], '3', '''layer updated: scatter_add(axis=0) from base,indices,values
layer result: gather(axis=0) from updated,indices''')
    for model in variants(tmp_path,code):
        base = torch.randn(2,4,requires_grad=True)
        values = torch.randn(2,3,requires_grad=True)
        index = torch.tensor([[1,1,3],[0,2,2]])
        out = model(base,index,values)['y']
        expected = base.scatter_add(1,index,values).gather(1,index)
        torch.testing.assert_close(out,expected)
        ga = torch.autograd.grad(out.square().sum(),(base,values),retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(),(base,values))
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b)
        for invalid in (-1,4):
            with pytest.raises(ValueError,match='bounds'):
                model(base,torch.full_like(index,invalid),values)


@pytest.mark.parametrize('op,fn',[('eq',torch.eq),('lt',torch.lt)])
def test_bool_comparison(tmp_path,op,fn):
    code = source([('x','i64','3'),('z','i64','3')],'3',f'layer result: {op} from x,z','bool')
    for model in variants(tmp_path,code):
        x,z = torch.tensor([1,2,3]),torch.tensor([2,2,1])
        torch.testing.assert_close(model(x,z)['y'],fn(x,z))


@pytest.mark.parametrize('layer',[
    'matmul from x,z','broadcast from x,z','slice(axis=2,start=0,stop=1) from x',
    'slice(axis=0,start=-1,stop=1) from x','slice(axis=0,start=0,stop=3,step=0) from x',
    'reduce_sum(axis=0,keepdim=2) from x','gather(axis=0) from x,z',
    'eq from x,z','stack(axis=0) from x,z',
])
def test_invalid_primitives_fail_statically(layer):
    with pytest.raises(WgtError):
        parse(source([('x','f32','2,3'),('z','f32','2,4')],'2,3','layer result: '+layer))


@pytest.mark.parametrize('groups,dilation,causal,stride',[(1,1,0,1),(2,2,0,2),(4,2,1,1)])
def test_grouped_conv1d_forward_gradient_and_causal_prefix(tmp_path,groups,dilation,causal,stride):
    padding = 0 if causal else 2
    length = (9+(dilation*2 if causal else 2*padding)-dilation*2-1)//stride+1
    code = source([('x','f32','4,9')],f'4,{length}',f'layer result: conv1d(in=4,out=4,kernel=3,groups={groups},dilation={dilation},causal={causal},padding={padding},stride={stride})')
    for model in variants(tmp_path,code):
        x = torch.randn(2,4,9,requires_grad=True)
        conv = model.result
        out = model(x)['y']
        padded = torch.nn.functional.pad(x,(dilation*2,0)) if causal else x
        expected = torch.nn.functional.conv1d(padded,conv.weight,conv.bias,stride=stride,padding=padding,dilation=dilation,groups=groups)
        torch.testing.assert_close(out,expected)
        params = (x,conv.weight,conv.bias)
        ga = torch.autograd.grad(out.square().sum(),params,retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(),params)
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b)
        if causal:
            altered = x.detach().clone()
            altered[:,:,5:] += 10
            torch.testing.assert_close(model(altered)['y'][:,:,:5],out[:,:,:5])
