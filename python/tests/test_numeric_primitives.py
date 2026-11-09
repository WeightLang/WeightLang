"""New tensor primitives agree with Rust execution and native VJP contracts."""

from pathlib import Path
import json
import os
import subprocess

import pytest
import torch

import weightlang as wgt


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def compiler(monkeypatch):
    monkeypatch.setenv("WGT_COMPILER", os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt")))


def source(body, *, width=2, output=2):
    return f"wgt 0.5\nmodel Numeric\ninput x:f32[{width}]\noutput y:f32[{output}]\nfn forward(x){{{body}}}"


def native(text, inputs, gradient=None, *, mode="forward", error=None):
    executable = os.environ.get("WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference"))
    request = {"source": text, "input": inputs, "gradient": gradient, "mode": mode}
    process = subprocess.run([executable], input=json.dumps(request), text=True, capture_output=True, timeout=30)
    if error:
        assert process.returncode and error in process.stderr, process.stderr
        return
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize("body,inputs,gradient,output", [
    ("return div(x,[2,-4]);", [3., 8.], [2., -3.], 2),
    ("return sqrt(x);", [4., 9.], [3., -2.], 2),
    ("return scale(slice(x,0,2),slice(x,2,1));", [2., 3., 0.5], [2., -1.], 2),
    ("return div(slice(x,0,1),slice(x,1,1));", [1e-38, 1e-38], [1e-38], 1),
    ("return scale(slice(x,0,2),slice(x,2,1));", [3e38, -3e38, 1.], [2., 2.], 2),
    ("return sqrt(x);", [0., 9.], [0., 1.], 2),
])
@pytest.mark.parametrize("dynamic", [False, True])
def test_numeric_forward_and_vjp_match_native(body, inputs, gradient, output, dynamic):
    if dynamic:
        body = "if positive([1]) { " + body + " } return " + ("[0]" if output == 1 else "[0,0]") + ";"
    text = source(body, width=len(inputs), output=output)
    expected = native(text, inputs, gradient)
    model = wgt.loads(text, device="cpu")
    assert isinstance(model, torch.fx.GraphModule) != dynamic
    value = torch.tensor(inputs, requires_grad=True)
    result = model(value)
    result.backward(torch.tensor(gradient))
    torch.testing.assert_close(result, torch.tensor(expected["output"]), rtol=1e-6, atol=0.)
    torch.testing.assert_close(value.grad, torch.tensor(expected["input_gradient"]), rtol=2e-6, atol=0.)


def test_hard_comparison_extremes_equal_endpoint_and_grad_refusal():
    text = source("return ge(x,[-3e38,3e38,1,1]);", width=4, output=4)
    model = wgt.loads(text, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(model(torch.tensor([3e38, -3e38, 1., 0.])), torch.tensor([1., 0., 1., 0.]), rtol=0, atol=0)
    model.eval()  # Inference mode is an autograd decision, not the train flag.
    with pytest.raises(RuntimeError, match="ge.*automatic differentiation"):
        model(torch.ones(4))
    native(text, [1., 1., 1., 1.], error="ge does not support automatic differentiation")


def test_ge_only_rejects_executed_ad_paths_and_typed_step_uses_no_ad():
    text = source("if positive(sum(x)) { return ge(x,[0,0]); } return x;")
    model = wgt.loads(text, device="cpu")
    x = torch.tensor([-1., -1.], requires_grad=True)
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(2))
    with pytest.raises(RuntimeError, match="ge.*automatic differentiation"):
        model(torch.ones(2))
    cell = wgt.loads("""wgt 0.5
model Spike
state v:f32[2]=[0,0];
fn step(x:f32[2])->f32[2] { let u=v+x; let spike=ge(u,[1,1]); v <- u-spike; return spike; }
fn reset() { v <- [0,0]; }
""", device="cpu")
    torch.testing.assert_close(cell.step(torch.tensor([0.6, 1.])), torch.tensor([0., 1.]))
    torch.testing.assert_close(cell.step(torch.tensor([0.6, 1.])), torch.tensor([1., 1.]))
    cell.reset()
    torch.testing.assert_close(cell.step(torch.tensor([0.6, 1.])), torch.tensor([0., 1.]))


def test_dynamic_scaling_keeps_model_axes_and_arbitrary_batch_axes():
    model = wgt.loads(source("return scale(x[0:4] as [2,2],x[4:5]) as [4];", width=5, output=4), device="cpu")
    x = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5).requires_grad_()
    actual = model(x)
    torch.testing.assert_close(actual, x[..., :4] * x[..., 4:])
    actual.sum().backward()
    expected = torch.cat((x[..., 4:].expand(2, 3, 4), x[..., :4].sum(-1, keepdim=True)), -1)
    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize("body,inputs", [("return div(x,[0,1]);", [1., 1.]), ("return sqrt(x);", [-1., 1.])])
def test_invalid_numeric_results_are_not_silently_published(body, inputs):
    text = source(body)
    with pytest.raises(RuntimeError, match="nonfinite"):
        wgt.loads(text, device="cpu")(torch.tensor(inputs))
    native(text, inputs, error="nonfinite")


def test_sqrt_zero_nonzero_seed_rejects_nonfinite_backward():
    text = source("return sqrt(x);", width=1, output=1)
    native(text, [0.], [1.], error="nonfinite")
    with pytest.raises(RuntimeError, match="nonfinite.*gradient"):
        wgt.loads(text, device="cpu")(torch.zeros(1, requires_grad=True)).sum().backward()


def test_new_ir_contracts_reject_general_broadcast_and_old_version():
    definition = wgt.compile_source(source("return scale(x,2);"))
    # A literal factor remains the old scale instruction for backward compatibility.
    assert any(item["op"] == "scale" for item in definition["program"]["instructions"])
    definition = wgt.compile_source(source("return scale(x,mean(x));"))
    assert any(item["op"] == "scale_by" for item in definition["program"]["instructions"])
    definition["program"]["version"] = 2
    with pytest.raises(ValueError, match="version 3"):
        wgt.from_ir(definition, device="cpu")
    for body in ("return ge(x,[1]);", "return div(x,[1]);", "return scale(x,[1,2]);"):
        with pytest.raises(wgt.CompilationError):
            wgt.compile_source(source(body))
