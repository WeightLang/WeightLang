import importlib.util

import pytest
import torch

from weightlang import compile_file, export_python, load_artifact, parse, run_sequence, WgtError
from weightlang.devices import placement_plan

SOURCE = '''wgt 0.3
model decoder
input tokens: i64[time=4]
output logits: f32[time=4, 13]
layer embed: embedding(vocab=13, dim=8)
layer attn: rope_attention(dim=8, heads=2, kv_heads=1, head_dim=4, qk_norm=1, max_len=32)
layer experts: sparse_moe(dim=8, hidden=12, experts=3, top_k=2)
layer head: linear(in=8, out=13, bias=0, tie_embedding=1)
init seed=5
'''


def test_artifact_export_cache_and_aliases(tmp_path):
    path=tmp_path/'model.wgt'; path.write_text(SOURCE)
    built=compile_file(path,tmp_path/'base',no_train=True)
    model=load_artifact(built.path,warn_untrained=False).module
    assert model.embed.weight is model.head.weight
    x=torch.tensor([[1,2,3,4],[4,3,2,1]])
    full=run_sequence(model,x)
    first=run_sequence(model,x[:,:2],use_cache=True)
    last=run_sequence(model,x[:,2:],past_key_values=first.past_key_values,use_cache=True)
    torch.testing.assert_close(full.output[:,2:],last.output,atol=1e-6,rtol=1e-5)
    model.train()
    output=run_sequence(model,x)
    (output.output.square().mean()+output.aux_loss).backward()
    assert all(p.grad is not None for p in model.parameters())
    model.eval()
    exported=tmp_path/'standalone.py'
    export_python(built.path,exported)
    spec=importlib.util.spec_from_file_location('standalone',exported)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    restored=module.load_model()
    assert restored.embed.weight is restored.head.weight
    torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
    for cache in [(), ((torch.zeros(2,1,2,4),torch.zeros(2,1,2,4)),)]:
        with pytest.raises(ValueError,match='cache'):
            run_sequence(model,x,past_key_values=cache)
    with pytest.raises(ValueError,match='attention_mask'):
        run_sequence(model,x,attention_mask=torch.full((2,4),0.5))


@pytest.mark.parametrize('old,new',[
    ('top_k=2','top_k=4'),('tie_embedding=1','tie_embedding=2'),
    ('bias=0','bias=1'),('max_len=32','max_len=0'),
    ('qk_norm=1','qk_norm=9'),('experts=3','experts=0'),
])
def test_invalid_contracts(old,new):
    with pytest.raises(WgtError):
        parse(SOURCE.replace(old,new))


def test_explicit_initial_state_preserves_frozen_weights(tmp_path):
    from weightlang import compile_program, parse_file
    source=tmp_path/'snapshot.wgt';source.write_text(SOURCE+'\nfreeze embed\n')
    program=parse_file(source,check_training=False)
    baseline=compile_program(program,tmp_path/'base',no_train=True)
    state={key:value.clone() for key,value in baseline.module.state_dict().items()}
    state['embed.weight'].add_(1);state['head.weight'].copy_(state['embed.weight'])
    result=compile_program(program,tmp_path/'snapshot',no_train=True,initial_state=state)
    restored=load_artifact(result.path,warn_untrained=False)
    assert not restored.module.embed.weight.requires_grad
    assert restored.module.embed.weight is restored.module.head.weight
    assert torch.equal(restored.module.embed.weight,state['embed.weight'])
    assert all(v['origin']=='edited' for v in restored.manifest['weights']['tensors'].values())
    with pytest.raises(WgtError,match='no_train'):
        compile_program(program,tmp_path/'bad',initial_state=state)
    state['head.weight'].add_(1)
    with pytest.raises(WgtError,match='tied'):
        compile_program(program,tmp_path/'bad',no_train=True,initial_state=state)
    assert not (tmp_path/'bad').exists()
