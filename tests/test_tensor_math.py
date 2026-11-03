"""Native graph replacements for math used by the legacy tensor frontend."""
from pathlib import Path

import pytest
import torch

from weightlang import WgtError, load, parse
from weightlang.primitives import arguments
from test_primitives import source, variants


@pytest.mark.parametrize('op,fn', [
    ('square', torch.square), ('rsqrt', torch.rsqrt), ('exp', torch.exp),
    ('log', torch.log), ('sin', torch.sin), ('cos', torch.cos),
    ('softplus', torch.nn.functional.softplus),
    ('shift(value=1e-6)', lambda x: x + 1e-6),
    ('cumsum(axis=-1)', lambda x: x.cumsum(-1)),
])
def test_math_forward_gradients_and_artifact_paths(tmp_path, op, fn):
    code = source([('x', 'f32', '2,3')], '2,3', f'layer result: {op}')
    for model in variants(tmp_path, code):
        x = (torch.rand(2,3,2) + 0.5).transpose(1,2).requires_grad_()
        actual = model(x)['y']
        expected = fn(x)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        gb = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(ga, gb)


@pytest.mark.parametrize('block,reduction', [('l2norm', torch.sum), ('rms', torch.mean)])
def test_native_normalization_zero_noncontiguous_and_symbolic(tmp_path, block, reduction):
    code = source([('x', 'f32', 'T,4')], 'T,4',
                  f'layer result: tensor_math.{block}(eps=1e-6)',
                  dims='use std/tensor_math\ndim T min=1 max=7\n')
    for model in variants(tmp_path, code):
        for length in (1, 7):
            for batch in (False, True):
                shape = (2,4,length) if batch else (4,length)
                x = torch.randn(shape).transpose(-1,-2).requires_grad_()
                actual = model(x)['y']
                expected = x * torch.rsqrt(reduction(x*x, -1, keepdim=True) + 1e-6)
                torch.testing.assert_close(actual, expected)
                ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
                gb = torch.autograd.grad(expected.square().sum(), x)[0]
                torch.testing.assert_close(ga, gb)
                zero = torch.zeros_like(x, requires_grad=True)
                out = model(zero)['y']
                assert torch.count_nonzero(out) == 0
                assert torch.isfinite(torch.autograd.grad(out.sum(), zero)[0]).all()


def test_cumulative_decay_full_and_split_gradient():
    path = Path(__file__).resolve().parents[1] / 'examples/tensor_math.wgt'
    model = load(path)
    x = torch.randn(2,7,4, requires_grad=True)
    whole = model(x)['decay']
    first, second = model(x[:,:3])['decay'], model(x[:,3:])['decay']
    split = torch.cat((first, second * first[:,-1:]), dim=1)
    torch.testing.assert_close(whole, split)
    a = torch.autograd.grad(whole.sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(split.sum(), x)[0]
    torch.testing.assert_close(a, b)


def test_native_l2norm_matches_existing_qwen_helper(tmp_path):
    from weightlang.tensor_program import parse_tensor_program

    root = Path(__file__).resolve().parents[1]
    legacy = parse_tensor_program((root/'examples/bootstrap/qwen/model.wgt').read_text()).classes({})['l2norm']
    path = tmp_path/'l2.wgt'
    path.write_text(source([('x','f32','2,7,4')], '2,7,4',
                           'layer result: tensor_math.l2norm(eps=1e-6)',
                           dims='use std/tensor_math\n'))
    model = load(path)
    x = torch.randn(3,2,4,7).transpose(-1,-2).requires_grad_()
    actual, expected = model(x)['y'], legacy(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(a, b)


def test_cumsum_empty_axis_and_softplus_large_values(tmp_path):
    path = tmp_path/'math.wgt'
    path.write_text(source([('x','f32','0,3')], '0,3', 'layer result: cumsum(axis=0)'))
    x = torch.empty(2,0,3, requires_grad=True)
    y = load(path)(x)['y']
    assert y.shape == x.shape
    assert torch.autograd.grad(y.sum(), x)[0].shape == x.shape
    path.write_text(source([('x','f32','5')], '5', 'layer result: softplus'))
    x = torch.tensor([-100., 0., 19., 21., 100.], requires_grad=True)
    actual, expected = load(path)(x)['y'], torch.nn.functional.softplus(x)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    a = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('op', ['square','rsqrt','exp','log','sin','cos','softplus','shift(value=2)'])
def test_scalar_empty_and_rejected_integer_inputs(tmp_path, op):
    for shape in ('', '0,3'):
        code = source([('x','f32',shape)], shape, f'layer result: {op}')
        path = tmp_path/'math.wgt'
        path.write_text(code)
        x = torch.ones(()) if not shape else torch.empty(0,3)
        assert load(path)(x)['y'].shape == x.shape
    with pytest.raises(WgtError, match='f32'):
        parse(source([('x','i64','3')], '3', f'layer result: {op}', dtype='i64'))


def test_math_domains_and_argument_rejection(tmp_path):
    for value in (True, float('nan'), float('inf'), '1'):
        with pytest.raises(ValueError, match='finite number'):
            arguments('shift', {'value': value})
    for args in ({'axis': 0.5}, {'axis': True}, {'axis': 0, 'extra': 1}):
        with pytest.raises(ValueError):
            arguments('cumsum', args)
    with pytest.raises(WgtError, match='axis'):
        parse(source([('x','f32','')], '', 'layer result: cumsum(axis=0)'))
    for op in ('log', 'rsqrt'):
        path = tmp_path/f'{op}.wgt'
        path.write_text(source([('x','f32','2')], '2', f'layer result: {op}'))
        actual = load(path)(torch.tensor([-1., 0.]))['y']
        expected = getattr(torch, op)(torch.tensor([-1., 0.]))
        torch.testing.assert_close(actual, expected, equal_nan=True)
