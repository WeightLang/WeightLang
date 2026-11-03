"""Control flow must skip work, preserve ordering and expose useful gradients."""
import pytest
import torch

from weightlang import WgtError, parse
from test_primitives import variants

ROUTE = '''wgt 0.3
model routed
input choice: i64[]
input x: f32[3]
output y: f32[3] from selected
block expert():
  layer value: linear(3) -> tanh
layer selected: route(expert(),expert(),identity) from choice,x
init seed=7
'''

PONDER = '''wgt 0.3
model pondered
input x: f32[2]
output y: f32[2] from reasoning.output
output states: f32[4,2] from reasoning.states
output probabilities: f32[4,1] from reasoning.probabilities
output weights: f32[4,1] from reasoning.halt_weights
output steps: f32[] from reasoning.expected_steps
block cell():
  layer next: linear(2) -> tanh from input
  layer stop: linear(1) -> sigmoid from next
  layer result: identity from next
layer reasoning: ponder(4,cell(),until=stop)
init seed=7
'''


def test_route_real_sparse_execution_order_and_gradients(tmp_path):
    for m in variants(tmp_path, ROUTE):
        x = torch.randn(5,3,requires_grad=True)
        choices = torch.tensor([1,0,1,0,1])
        seen = [[],[]]
        hooks = [m.selected._modules[f'branch{i}'].register_forward_pre_hook(
            lambda mod,args,i=i: seen[i].append(args[0].shape[0])) for i in range(2)]
        unused = m.selected.branch2.register_forward_pre_hook(lambda *args: (_ for _ in ()).throw(AssertionError('unused branch ran')))
        actual = m(choices,x)['y']
        assert seen == [[2],[3]]
        for hook in [*hooks,unused]: hook.remove()
        expected = torch.stack([m.selected._modules[f'branch{int(c)}'](v[None])[0] for c,v in zip(choices,x)])
        torch.testing.assert_close(actual,expected)
        params = (x,*m.selected.branch0.parameters(),*m.selected.branch1.parameters())
        ga = torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(),params)
        for a,b in zip(ga,gb): torch.testing.assert_close(a,b)
        torch.testing.assert_close(m(torch.tensor(2),x[0])['y'],x[0])
        assert m(torch.empty(0,dtype=torch.int64),x[:0])['y'].shape == (0,3)
        with pytest.raises(ValueError,match='range'): m(torch.tensor([3]*5),x)


def test_boolean_route_and_nested_multiport_block(tmp_path):
    code = '''wgt 0.3
model conditional
input choose: bool[]
input x: f32[2]
output y: f32[2] from outer.output
block pair():
  layer routed: route(identity,scale(factor=2)) from input,arg1
  return output=routed
layer outer: pair() from choose,x
init seed=0
'''
    for m in variants(tmp_path,code):
        x = torch.tensor([[1.,2.],[3.,4.]])
        torch.testing.assert_close(m(torch.tensor([False,True]),x)['y'],torch.tensor([[1.,2.],[6.,8.]]))


def test_ponder_distribution_manual_gradient_and_training(tmp_path):
    for m in variants(tmp_path,PONDER):
        m.train()
        x = torch.randn(3,2,requires_grad=True)
        out = m(x)
        states,ps,masses = [],[],[]
        value = x
        remaining = torch.ones(3,1)
        for step in range(4):
            value = m.reasoning.next(value)
            p = m.reasoning.stop(value)
            states.append(value); ps.append(p)
            masses.append(remaining if step == 3 else remaining*p)
            remaining = remaining*(1-p)
        expected = torch.stack(masses,1)
        torch.testing.assert_close(out['states'],torch.stack(states,1))
        torch.testing.assert_close(out['probabilities'],torch.stack(ps,1))
        torch.testing.assert_close(out['weights'],expected)
        torch.testing.assert_close(out['weights'].sum(1),torch.ones(3,1))
        torch.testing.assert_close(out['steps'],(expected*torch.arange(1,5).view(1,4,1)).sum(1).squeeze(-1))
        costs = out['states'].square().mean(-1,keepdim=True)
        loss = (costs*out['weights']).sum(1).mean() + .1*out['steps'].mean()
        before = m.reasoning.stop.step0.weight.detach().clone()
        optimizer = torch.optim.SGD(m.parameters(),lr=.1)
        optimizer.zero_grad(); loss.backward()
        assert m.reasoning.stop.step0.weight.grad.abs().sum() > 0
        assert torch.isfinite(x.grad).all()
        optimizer.step()
        assert not torch.equal(before,m.reasoning.stop.step0.weight)
        # Pondering exposes a deterministic full trace in both modes.
        m.eval()
        torch.testing.assert_close(m(x)['weights'],m(x)['weights'])
        assert m(x[0])['states'].shape == (4,2)


def test_compact_recur_actually_reduces_batch_and_preserves_gradient(tmp_path, monkeypatch):
    source = PONDER[:PONDER.index('output y:')] + '''output y: f32[2] from reasoning
block cell():
  layer next: linear(2) from input
  layer stop: linear(1) -> sigmoid from next
  layer result: identity from next
layer reasoning: recur(5,cell(),until=stop,compact=1)
init seed=0
'''
    for m in variants(tmp_path,source):
        with torch.no_grad():
            m.reasoning.next.weight.copy_(torch.eye(2));m.reasoning.next.bias.fill_(1)
            m.reasoning.stop.step0.weight.zero_();m.reasoning.stop.step0.bias.zero_()
        def draw(p):
            result = torch.full_like(p,.9);result[0]=.1
            return result
        monkeypatch.setattr(torch,'rand_like',draw)
        sizes=[]
        hook=m.reasoning.next.register_forward_pre_hook(lambda mod,args:sizes.append(len(args[0])))
        x=torch.zeros(4,2,requires_grad=True)
        y=m(x)['y']
        hook.remove()
        assert sizes==[4,3,2,1]
        torch.testing.assert_close(y,torch.arange(1.,5.)[:,None].expand(4,2))
        y.sum().backward()
        torch.testing.assert_close(m.reasoning.next.bias.grad,torch.full((2,),10.))
        torch.testing.assert_close(x.grad,torch.ones_like(x))
        torch.testing.assert_close(m(torch.zeros(2))['y'],torch.ones(2))


@pytest.mark.parametrize('code', [
    ROUTE.replace('i64[]','f32[]'),
    ROUTE.replace('i64[]','bool[]'),
    ROUTE.replace('linear(3)','linear(4)'),
    ROUTE.replace('route(expert(),expert(),identity)','route(identity)'),
    PONDER.replace(',until=stop',''),
    PONDER.replace('until=stop','until=missing'),
    PONDER.replace('linear(1)','linear(2)'),
    PONDER.replace('ponder(4','ponder(0'),
    PONDER.replace('until=stop','until=stop,compact=1'),
])
def test_invalid_control_contracts(code):
    with pytest.raises(WgtError): parse(code)


def test_ponder_can_compose_as_primary_tensor_without_record_outputs(tmp_path):
    code = '''wgt 0.3
model primary
input x: f32[2]
output y: f32[2]
block cell():
  layer next: scale(factor=0.5) from input
  layer stop: linear(1) -> sigmoid from next
  layer result: identity from next
block body():
  layer thought: ponder(3,cell(),until=stop)
  layer result: scale(factor=2)
layer result: residual(body())
init seed=0
'''
    for m in variants(tmp_path,code):
        x = torch.randn(3,2,requires_grad=True)
        y = m(x)
        torch.testing.assert_close(y,1.25*x)
        y.sum().backward()
        torch.testing.assert_close(x.grad,torch.full_like(x,1.25))


def test_ponder_single_step_has_unit_mass(tmp_path):
    code = PONDER.replace('ponder(4','ponder(1').replace('f32[4,2]','f32[1,2]').replace('f32[4,1]','f32[1,1]')
    for m in variants(tmp_path,code):
        out = m(torch.randn(2,2))
        torch.testing.assert_close(out['weights'],torch.ones(2,1,1))
        torch.testing.assert_close(out['steps'],torch.ones(2))
