import torch

from test_primitives import variants

CODE = '''wgt 0.3
model state_ports
dim T min=1 max=5
dim P min=0 max=5
require P + T <= 6
input x: f32[T,4]
input mask: bool[P+T]
input old_key: f32[P,1,4]
input old_value: f32[P,1,4]
output y: f32[T,4] from experts.output
output key: f32[P+T,1,4] from attention.key
output value: f32[P+T,1,4] from attention.value
output aux: f32[] from experts.aux
layer attention: rope_attention(dim=4,heads=1,kv_heads=1,head_dim=4,ports=1,max_len=6,flash=0) from x,mask,old_key,old_value
layer experts: sparse_moe(dim=4,hidden=6,experts=2,top_k=2,ports=1)
init seed=7
'''


def test_cache_and_aux_are_tensor_ports_with_chunk_gradients(tmp_path):
    for model in variants(tmp_path,CODE):
        model.train()
        x = torch.randn(2,5,4,requires_grad=True)
        empty = torch.empty(2,0,1,4)
        full = model(x,torch.ones(2,5,dtype=torch.bool),empty,empty)
        a = model(x[:,:2],torch.ones(2,2,dtype=torch.bool),empty,empty)
        b = model(x[:,2:],torch.ones(2,5,dtype=torch.bool),a['key'],a['value'])
        actual = torch.cat([a['y'],b['y']],1)
        torch.testing.assert_close(actual,full['y'])
        torch.testing.assert_close(b['key'],full['key'])
        torch.testing.assert_close(b['value'],full['value'])
        assert full['aux'].shape == (2,) and torch.isfinite(full['aux']).all()
        parameters = (x,*model.parameters())
        ga = torch.autograd.grad(actual.square().sum(),parameters,retain_graph=True)
        gb = torch.autograd.grad(full['y'].square().sum(),parameters,retain_graph=True)
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        aux_grad = torch.autograd.grad(full['aux'].sum(),model.experts.gate.weight)[0]
        assert torch.isfinite(aux_grad).all()
        assert not any('cache' in k or 'old_' in k for k in model.state_dict())
