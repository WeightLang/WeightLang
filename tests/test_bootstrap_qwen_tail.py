"""Internal edit gradients, executable task grading, and split isolation."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from weightlang.tensor_program import parse_tensor_program

ROOT=Path(__file__).resolve().parents[1]
EXAMPLE=ROOT/'examples/bootstrap/qwen'


def curriculum():
    spec=importlib.util.spec_from_file_location('bootstrap_curriculum',EXAMPLE/'curriculum.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_curriculum_executes_and_has_disjoint_families():
    mod=curriculum();data=mod.dataset()
    assert {s:len(rows) for s,rows in data.items()}=={'train':54,'validation':6,'test':8,'general':6}
    sets=[{r['family'] for r in data[s]} for s in ('train','validation','test')]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    for rows in data.values():
        for row in rows:
            assert mod.grade(row,row['answer'])['task_correct'],row
    row=next(r for r in data['test'] if r['family']=='sum_completion')
    wrong=row['answer'].replace('sum(axis=time)','mean(axis=time)')
    result=mod.grade(row,wrong)
    assert result['parses'] and result['runs'] and not result['task_correct']
    assert not mod.grade(row,'layer h: linear(999999)')['task_correct']
    with pytest.raises(ValueError):
        mod.execute('wgt 0.3\ninit python="evil.py"\n')


def test_internal_edits_identity_gradient_and_causality():
    torch.set_num_threads(2);torch.manual_seed(42)
    classes=parse_tensor_program((EXAMPLE/'model.wgt').read_text()).classes({})
    c=dict(hidden_size=32,intermediate_size=64,num_attention_heads=4,num_key_value_heads=2,
           head_dim=8,linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,
           linear_value_head_dim=8,linear_conv_kernel_dim=4,rms_norm_eps=1e-6,
           layer_types=['linear_attention','full_attention'],
           rope_parameters={'partial_rotary_factor':0.5,'rope_theta':10000000.0})
    tail=classes['EditedTail'](c,start=0,rank=8,steps=2,parent_rank=4)
    for layer in tail.layers:
        layer.requires_grad_(False)
    tail.norm.requires_grad_(False)
    frozen={n:p.clone() for n,p in tail.named_parameters() if not p.requires_grad}
    x=torch.randn(1,9,32)
    torch.testing.assert_close(tail(x),tail(x,edited=False),rtol=0,atol=0)
    opt=torch.optim.AdamW([p for p in tail.parameters() if p.requires_grad],lr=0.01)
    for _ in range(2):
        opt.zero_grad();tail(x).square().mean().backward();opt.step()
    for i in (0,1):
        for name,p in tail.edits[i].named_parameters():
            assert p.grad is not None and torch.count_nonzero(p.grad)>0,(i,name)
    assert all(torch.equal(p,frozen[n]) for n,p in tail.named_parameters() if n in frozen)
    assert tail.edits[0].up.weight is not tail.edits[1].up.weight
    assert not torch.equal(tail(x),tail(x,edited=False))
    changed=x.clone();changed[:,5:]=torch.randn_like(changed[:,5:])
    torch.testing.assert_close(tail(x)[:,:5],tail(changed)[:,:5],rtol=0,atol=0)


def test_streamed_prefix_batching_and_lossless_block_cache(monkeypatch):
    pytest.importorskip('transformers',exc_type=ImportError)
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from train import StreamedModel
    classes=parse_tensor_program((EXAMPLE/'model.wgt').read_text()).classes({})
    c=dict(hidden_size=32,intermediate_size=64,num_attention_heads=4,num_key_value_heads=2,
           head_dim=8,vocab_size=64,hidden_act='silu',linear_num_key_heads=2,
           linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
           linear_conv_kernel_dim=4,rms_norm_eps=1e-6,layer_types=['linear_attention','full_attention'],
           rope_parameters={'rope_type':'default','partial_rotary_factor':0.5,'rope_theta':10000000.0})
    model=classes['QwenText'](c).eval()
    stream=object.__new__(StreamedModel)
    stream.c=c;stream.classes=classes;stream.device='cpu';stream.prefix=''
    stream.cache_bytes=1000000;stream.cached_bytes=0;stream.block_cache={}
    state=model.state_dict();reads=[]
    def tensor(key):
        reads.append(key)
        return state[key].clone()
    stream.tensor=tensor
    sequences=[torch.tensor([1,2,3]),torch.tensor([4,5,6,7,8,9,10])]
    singles=stream.features(sequences,stop=2,batch_size=1)
    count=len(reads)
    batched=stream.features(sequences,stop=2,batch_size=8)
    assert len(reads)==count+1  # embedding only; both decoder blocks are reused without quantization
    for a,b in zip(singles,batched):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    assert [len(h) for h in batched]==[3,7]
    stream.block_cache.clear();stream.cached_bytes=0;stream.cache_bytes=0
    uncached=stream.features(sequences,stop=2,batch_size=1)
    for a,b in zip(singles,uncached):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
