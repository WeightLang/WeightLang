"""Shape contracts, numerical semantics and autograd for named axes."""

import json

import pytest
import torch

from weightlang.axes import AxisTransform, plan


def test_serializable_plan_and_size_inference():
    sizes = {"head": 4}
    result = plan("time (head width) -> head time width", (3, 20), sizes)
    assert result == {
        "expanded_shape": [3, 4, 5],
        "permutation": [1, 0, 2],
        "output_shape": [4, 3, 5],
        "reduction_axes": [],
        "input_axes": ["time", None],
        "output_axes": ["head", "time", "width"],
    }
    assert json.loads(json.dumps(result)) == result
    assert sizes == {"head": 4}
    assert plan("(a b) -> (b a)", (6,), {"a": 2, "b": 3})["output_axes"] == [None]


@pytest.mark.parametrize("batch", [(), (2,), (2, 3), (0,)])
@pytest.mark.parametrize("kind", ["transpose", "split", "merge"])
def test_rearrange_values_and_gradients(batch, kind):
    x = torch.arange(torch.tensor(batch + (3, 8)).prod().item(), dtype=torch.float64)
    x = x.reshape(batch + (3, 8)).requires_grad_()
    if kind == "transpose":
        module = AxisTransform("time feature -> feature time", {}, (3, 8))
        expected = x.transpose(-1, -2)
    elif kind == "split":
        module = AxisTransform("time (head width) -> head time width", {"head": 4}, (3, 8))
        expected = x.reshape(batch + (3, 4, 2)).transpose(-3, -2)
    else:
        module = AxisTransform("time feature -> (time feature)", {}, (3, 8))
        expected = x.flatten(-2)
    actual = module(x)
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(actual.numel(), dtype=x.dtype).reshape(actual.shape)
    actual_grad = torch.autograd.grad((actual * weights).sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3), (0,)])
@pytest.mark.parametrize("op", ["mean", "sum"])
def test_reduce_permuted_grouped_axes_values_and_gradients(batch, op):
    x = torch.randn(batch + (3, 8, 5), dtype=torch.float64, requires_grad=True)
    module = AxisTransform("time (head width) feature -> (feature head)", {"head": 4}, (3, 8, 5), op)
    assert module.plan["permutation"] == [3, 1, 0, 2]
    assert module.plan["reduction_axes"] == [2, 3]
    expanded = x.reshape(batch + (3, 4, 2, 5))
    expected = getattr(expanded, op)(dim=(-4, -2)).transpose(-1, -2).flatten(-2)
    actual = module(x)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weights).sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("op", ["mean", "sum"])
@pytest.mark.parametrize("batch", [(), (2, 3)])
def test_reduce_all_and_no_axes(op, batch):
    x = torch.randn(batch + (2, 4), requires_grad=True)
    actual = AxisTransform("a b ->", {}, (2, 4), op)(x)
    torch.testing.assert_close(actual, getattr(x, op)(dim=(-2, -1)))
    assert actual.shape == batch
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 1 / 8 if op == "mean" else 1))
    torch.testing.assert_close(AxisTransform("a b -> b a", {}, (2, 4), op)(x), x.transpose(-1, -2))


def test_noncontiguous_and_scalar_samples():
    x = torch.arange(24).reshape(8, 3).T
    assert not x.is_contiguous()
    actual = AxisTransform("t (h w) -> (w t) h", {"h": 4}, (3, 8))(x)
    torch.testing.assert_close(actual, x.reshape(3, 4, 2).permute(2, 0, 1).reshape(6, 4))
    for shape in [(), (2, 3)]:
        scalar_samples = torch.randn(shape)
        torch.testing.assert_close(AxisTransform("->", {}, ())(scalar_samples), scalar_samples)


@pytest.mark.parametrize("pattern,shape,sizes,reduction", [
    ("a b", (2, 3), {}, None),
    ("a -> a -> a", (2,), {}, None),
    ("a a -> a", (2, 2), {}, "sum"),
    ("a -> a a", (2,), {}, None),
    ("a (a b) -> b", (2, 6), {"b": 3}, "sum"),
    ("a -> b", (2,), {}, "sum"),
    ("a b -> a", (2, 3), {}, None),
    ("(a b) -> a b", (6,), {}, None),
    ("(a b) -> a b", (7,), {"a": 2}, None),
    ("(a b) -> a b", (6,), {"a": 2, "b": 4}, None),
    ("a -> a", (2,), {"a": 3}, None),
    ("a -> a", (2,), {"unknown": 2}, None),
    ("a -> a", (2, 3), {}, None),
    ("a -> a", (0,), {}, None),
    ("a -> a", (-2,), {}, None),
    ("a -> a", (2.0,), {}, None),
    ("a -> a", (True,), {}, None),
    ("a -> a", (2,), {"a": 0}, None),
    ("a -> a", (2,), {"a": -2}, None),
    ("a -> a", (2,), {"a": 2.0}, None),
    ("a -> a", (2,), {"a": True}, None),
    ("a -> a", (2,), {}, "max"),
])
def test_invalid_plans(pattern, shape, sizes, reduction):
    with pytest.raises(ValueError):
        plan(pattern, shape, sizes, reduction)


@pytest.mark.parametrize("side", ["()", "((a))", "(a", "a)", "a,b", "...", "2", "a-b", "(a b)c", "a (b())"])
def test_malformed_patterns(side):
    with pytest.raises(ValueError):
        plan(f"{side} -> a", (6,), {}, "sum")
    with pytest.raises(ValueError):
        plan(f"a -> {side}", (6,), {})


@pytest.mark.parametrize("shape", [(6,), (3, 2), (4, 2, 4)])
def test_runtime_rejects_incorrect_sample_tail(shape):
    with pytest.raises(ValueError, match="sample tail"):
        AxisTransform("a b -> b a", {}, (2, 3))(torch.zeros(shape))
