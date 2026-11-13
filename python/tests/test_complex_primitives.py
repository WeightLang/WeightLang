"""Native/PyTorch differential coverage for LightNest's new selection kernels."""

import pytest
import torch
from weightlang import torch_api as wgt
from test_numeric_primitives import native, source


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "body,values,gradient",
    [
        (
            "return gelu(x);",
            [-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0],
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0],
        ),
        (
            "let v=x as [2,4];return (v*topk_mask(v,[2])) as [8];",
            [4.0, 4.0, 2.0, 1.0, -5.0, -4.0, -4.0, -8.0],
            [1.0] * 8,
        ),
        (
            "return masked_softmax(x as [2,3],[1,0,1,0,1,1] as [2,3],1) as [6];",
            [0.2, 1000.0, -0.3, 1000.0, 2.0, 1.0],
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        ),
        (
            "return masked_softmax(x as [2,3],[1,0,1,0,1,1] as [2,3],0) as [6];",
            [0.2, 1000.0, -0.3, 1000.0, 2.0, 1.0],
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        ),
    ],
)
def test_complex_kernel_forward_and_vjp(dynamic, body, values, gradient):
    if dynamic:
        body = "if positive([1]) {" + body + "} return x;"
    text = source(body, width=len(values), output=len(values))
    expected = native(text, values, gradient)
    model = wgt.loads(text, device="cpu")
    value = torch.tensor(values, requires_grad=True)
    output = model(value)
    output.backward(torch.tensor(gradient))
    torch.testing.assert_close(
        output.detach(), torch.tensor(expected["output"]), atol=1e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        value.grad, torch.tensor(expected["input_gradient"]), atol=1e-6, rtol=2e-6
    )
    if "gelu" in body:
        oracle = torch.tensor(values, requires_grad=True)
        want = torch.nn.functional.gelu(oracle)
        want.backward(torch.tensor(gradient))
        torch.testing.assert_close(output.detach(), want.detach(), atol=1e-6, rtol=2e-6)
        torch.testing.assert_close(value.grad, oracle.grad, atol=1e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "body,values",
    [
        ("return topk_mask(x,[-1]);", [1.0, 2.0]),
        ("return topk_mask(x,[3]);", [1.0, 2.0]),
        ("return topk_mask(x,[0.5]);", [1.0, 2.0]),
        ("return masked_softmax(x,[0,0],0);", [1.0, 2.0]),
        ("return masked_softmax(x,[1,0.5],0);", [1.0, 2.0]),
    ],
)
def test_invalid_selection_and_masks_reject_in_both_backends(body, values):
    text = source(body, width=len(values), output=len(values))
    native(text, values, error="topk_mask" if "topk_mask" in body else "masked_softmax")
    with pytest.raises((ValueError, RuntimeError)):
        wgt.loads(text, device="cpu")(torch.tensor(values))


@pytest.mark.parametrize("dynamic", [False, True])
def test_shaped_large_parameter_read_and_sparse_embedding_gradient(dynamic):
    body = "return (embedding(table as [257,128],x) as [128]);"
    if dynamic:
        body = "if positive([1]) {" + body + "} return [0];"
    # Keep the dynamic fallback's return shape equal without a 128-entry literal.
    if dynamic:
        body = body.replace(
            "return [0];", "return embedding(table as [257,128],x) as [128];"
        )
    text = (
        "wgt 0.5\nmodel Large\ninput x:f32[1]\noutput y:f32[128]\nparameter table:f32[257,128]=ones()\nfn forward(x){"
        + body
        + "}"
    )
    model = wgt.loads(text, device="cpu")
    value = torch.tensor([256.0], requires_grad=True)
    output = model(value)
    output.sum().backward()
    torch.testing.assert_close(output, torch.ones(128), atol=0, rtol=0)
    gradient = next(model.parameters()).grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[:256]) == 0
    torch.testing.assert_close(gradient[256], torch.ones(128), atol=0, rtol=0)
    assert value.grad is None, "discrete embedding addresses have no Torch gradient"
