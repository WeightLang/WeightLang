import torch

from test_primitives import variants

CODE = '''wgt 0.3
model multi_state
dim T min=0 max=6
input x: f32[T,2]
input hidden: f32[3]
input cell: f32[3]
output y: f32[T,3] from memory.output
output h: f32[3] from memory.state0
output c: f32[3] from memory.state1
block pair():
  layer total: add(input,arg1)
  layer product: mul(input,arg1)
  return output=total product=product
block step():
  layer projected: linear(in=2,out=3)
  layer new_cell: add(projected,state1)
  layer paired: pair() from new_cell,state0
  layer new_hidden: tanh from paired.output
  return output=new_hidden state0=new_hidden state1=new_cell
layer memory: scan_state(step()) from x,hidden,cell
init seed=7
'''


def test_multiple_states_nested_record_block_chunks_and_empty(tmp_path):
    for model in variants(tmp_path,CODE):
        x,h,c = torch.randn(2,6,2,requires_grad=True),torch.randn(2,3,requires_grad=True),torch.randn(2,3,requires_grad=True)
        full = model(x,h,c)
        empty = model(x[:,:0],h,c)
        torch.testing.assert_close(empty['h'],h)
        torch.testing.assert_close(empty['c'],c)
        assert empty['y'].shape == (2,0,3)
        a = model(x[:,:2],empty['h'],empty['c'])
        b = model(x[:,2:],a['h'],a['c'])
        actual = torch.cat([a['y'],b['y']],1)
        torch.testing.assert_close(actual,full['y'])
        torch.testing.assert_close(b['h'],full['h'])
        torch.testing.assert_close(b['c'],full['c'])
        parameters = (x,h,c,*model.parameters())
        ga = torch.autograd.grad(actual.square().sum()+b['c'].square().sum(),parameters,retain_graph=True)
        gb = torch.autograd.grad(full['y'].square().sum()+full['c'].square().sum(),parameters)
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b)


def test_invalid_state_returns_fail_before_execution():
    import pytest
    from weightlang import WgtError, parse
    for code in (
        CODE.replace('state1=new_cell', ''),
        CODE.replace('state1=new_cell', 'state1=missing'),
        CODE.replace('state1=new_cell', 'state1=input'),
        CODE.replace('product=product', 'product=missing'),
    ):
        with pytest.raises(WgtError):
            parse(code)


def test_scalar_emission_with_index_and_bool_states(tmp_path):
    code = '''wgt 0.3
model typed_states
dim T min=0 max=3
input x: f32[T,2]
input counts: i64[2]
input enabled: bool[]
output y: f32[T] from memory.output
output indices: i64[2] from memory.state0
output gate: bool[] from memory.state1
block cell():
  # A scalar per time step, with unchanged index and boolean state tensors.
  layer output: reduce_sum(axis=0)
  return output=output state0=state0 state1=state1
layer memory: scan_state(cell()) from x,counts,enabled
init seed=3
'''
    for model in variants(tmp_path,code):
        for length in (0,3):
            x = torch.randn(2,length,2,requires_grad=True)
            counts = torch.tensor([[1,2],[3,4]])
            enabled = torch.tensor([True,False])
            result = model(x,counts,enabled)
            torch.testing.assert_close(result['y'],x.sum(-1))
            assert result['indices'] is counts and result['gate'] is enabled
            result['y'].sum().backward()
            torch.testing.assert_close(x.grad,torch.ones_like(x))
