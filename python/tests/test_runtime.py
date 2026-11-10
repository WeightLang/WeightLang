"""Numerical and gradient contracts across the Rust -> PyTorch boundary."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch.nn import functional as F

from weightlang.runtime import build_module


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compile_plan():
    executable = os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt"))
    assert Path(executable).is_file(), "Build the native wgt compiler before SDK tests"

    def compile_source(source):
        result = subprocess.run(
            [executable, "compile", "-", "--target", "torch"],
            input=source, text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return compile_source


def model_source(width, output, body, parameters=""):
    return f"""wgt 0.5
model TestModel
input x: f32[{width}]
output y: f32[{output}]
{parameters}
fn forward(x) {{ {body} }}
"""


@pytest.mark.parametrize("body,parameters", [
    ("let y=(x as [1,2]) @ (w as [2,2]); return softmax(layer_norm(y,[1,1],[0,0],1,0.0001),1) as [2];",
     "parameter w:f32[2,2]=[0.2,0.4,-0.1,0.3]"),
    ("return conv(x as [1,2],w as [1,1,1],[0],[1],[0],[1],1) as [2];",
     "parameter w:f32[1,1,1]=[0.5]"),
])
def test_native_precision_adapters_support_compile_export_and_backward(compile_plan, body, parameters):
    model = build_module(compile_plan(model_source(2, 2, body, parameters)))
    value = torch.tensor([[0.2, -0.3]], requires_grad=True)
    expected = model(value)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (value, *model.parameters()))
    exported = torch.export.export(model, (value,))
    torch.testing.assert_close(exported.module()(value), expected)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(value)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (value, *model.parameters()))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def weight(model, name):
    return model.get_parameter(model.parameter_names[name])


def assert_value_and_gradient(actual, expected, arguments):
    torch.testing.assert_close(actual, expected)
    upstream = torch.linspace(0.1, 1.0, actual.numel()).reshape(actual.shape)
    actual_grads = torch.autograd.grad(actual, arguments, upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, arguments, upstream, retain_graph=True)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_matrix_geometry_nonlast_norm_axis_and_noncontiguous_batches(compile_plan):
    source = model_source(6, 12, """
        let matrix = x as [3,2];
        let projected = matrix @ (w as [2,4]);
        let normalized = layer_norm(projected,gamma,beta,0,0.00001);
        let probabilities = softmax(normalized,1);
        return permute(probabilities,[1,0]) as [12];
    """, """
parameter w: f32[2,4] = [0.1,0.2,-0.3,0.4,0.5,-0.6,0.7,0.8]
parameter gamma: f32[3] = [0.7,1.0,1.3]
parameter beta: f32[3] = [-0.2,0.1,0.3]
""")
    model = build_module(compile_plan(source))
    x = torch.linspace(-1, 1, 36).reshape(2, 3, 6).transpose(0, 1).requires_grad_()
    projected = x.reshape(3, 2, 3, 2) @ weight(model, "w")
    normalized = F.layer_norm(projected.movedim(-2, -1), (3,), eps=1e-5).movedim(-1, -2)
    affine = normalized * weight(model, "gamma").reshape(3, 1) + weight(model, "beta").reshape(3, 1)
    expected = torch.softmax(affine, -1).transpose(-2, -1).reshape(3, 2, 12)
    assert_value_and_gradient(model(x), expected, (x, *model.parameters()))


def test_layer_norm_can_share_values_with_input_dependent_affine_parameters(compile_plan):
    definition = compile_plan(model_source(2, 2, "return layer_norm(w,x,[0,0],0,0.0001);",
                                           "parameter w:f32[2]=[-0.5,0.7]"))
    model = build_module(definition)
    x = torch.tensor([[1.0, 2.0], [-0.5, 0.3]], requires_grad=True)
    expected = F.layer_norm(weight(model, "w"), (2,), eps=0.0001) * x
    assert_value_and_gradient(model(x), expected, (x, weight(model, "w")))


def test_static_heads_and_dynamic_leading_batch_are_independent(compile_plan):
    source = model_source(12, 8, """
        let heads = x as [2,2,3];
        let projected = batch_matmul(heads,w as [2,3,2]);
        return projected as [8];
    """, "parameter w: f32[2,3,2] = [0.1,0.2,0.3,-0.4,0.5,0.6,0.7,-0.8,0.9,1.0,1.1,1.2]")
    model = build_module(compile_plan(source))
    x = torch.linspace(-2, 2, 240).reshape(4, 5, 12).requires_grad_()
    expected = (x.reshape(4, 5, 2, 2, 3) @ weight(model, "w")).reshape(4, 5, 8)
    assert_value_and_gradient(model(x), expected, (x, weight(model, "w")))


@pytest.mark.parametrize("input_shape,kernel_shape,groups", [
    ((2, 5), (4, 1, 3), 2),
    ((1, 4, 4), (2, 1, 3, 3), 1),
    ((1, 3, 3, 3), (2, 1, 2, 2, 2), 1),
])
def test_convolution_ranks_grouping_values_and_gradients(compile_plan, input_shape, kernel_shape, groups):
    rank = len(input_shape) - 1
    output_shape = (kernel_shape[0], *(i - k + 1 for i, k in zip(input_shape[1:], kernel_shape[2:])))
    kernel_data = [round((i % 11 - 5) * 0.03, 3) for i in range(math.prod(kernel_shape))]
    bias_data = [round(i * 0.1, 2) for i in range(kernel_shape[0])]
    source = model_source(math.prod(input_shape), math.prod(output_shape), f"""
        let image = x as {list(input_shape)};
        let result = conv(image,kernel as {list(kernel_shape)},bias,
                          {[1] * rank},{[0] * rank},{[1] * rank},{groups});
        return result as [{math.prod(output_shape)}];
    """, f"""
parameter kernel: f32[{','.join(map(str, kernel_shape))}] = {kernel_data}
parameter bias: f32[{kernel_shape[0]}] = {bias_data}
""")
    model = build_module(compile_plan(source))
    x = torch.linspace(-1, 1, math.prod(input_shape) * 6).reshape(2, 3, -1).requires_grad_()
    convolution = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
    expected = convolution(x.reshape(-1, *input_shape), weight(model, "kernel"), weight(model, "bias"), groups=groups)
    expected = expected.reshape(2, 3, math.prod(output_shape))
    assert_value_and_gradient(model(x), expected, (x, *model.parameters()))
    torch.testing.assert_close(model(x[0, 0]), expected[0, 0])


@pytest.mark.parametrize("operation", ["max_pool2d", "avg_pool2d"])
def test_pooling_keeps_examples_separate(compile_plan, operation):
    source = model_source(32, 12, f"""
        let image = x as [2,4,4];
        return {operation}(image,[2,2],[1,2]) as [12];
    """)
    model = build_module(compile_plan(source))
    x = torch.linspace(-1, 1, 192).reshape(2, 3, 32).requires_grad_()
    expected = getattr(F, operation)(x.reshape(-1, 2, 4, 4), (2, 2), (1, 2)).reshape(2, 3, 12)
    assert_value_and_gradient(model(x), expected, (x,))


def test_clipping_endpoints_and_minimum_ties_keep_wgt_derivatives(compile_plan):
    clip_model = build_module(compile_plan(model_source(5, 5, "return clip(x,-1,1);")))
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    torch.testing.assert_close(clip_model(x), torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))
    clip_model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    min_model = build_module(compile_plan(model_source(3, 3, "return minimum(x,p);", "parameter p: f32[3] = [0,0,0]")))
    values = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    min_model(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(weight(min_model, "p").grad, torch.tensor([0.0, 0.5, 1.0]))


def test_parameter_read_storage_flags_names_and_sharing(compile_plan):
    source = model_source(1, 1, "return x * p + x * p + training;", """
parameter p: prob = 0.25
frozen training: f32[1] = [3]
parameter definition: f32[1] = [9]
""")
    model = build_module(compile_plan(source))
    assert len(tuple(model.parameters())) == 3
    assert weight(model, "training").requires_grad is False
    assert weight(model, "definition").item() == 9
    assert isinstance(model.definition, dict)
    assert model.training is True
    x = torch.tensor([[2.0], [4.0]], requires_grad=True)
    torch.testing.assert_close(model(x), 0.5 * x + 3)
    model(x).sum().backward()
    torch.testing.assert_close(weight(model, "p").grad, torch.tensor([2.25]))
    assert weight(model, "definition").grad is None
    assert set(model.state_dict()) == set(model.parameter_names.values())
    replica = build_module(compile_plan(source))
    replica.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(replica(x), model(x))


def test_constant_model_repeats_per_example_and_buffers_follow_dtype(compile_plan):
    model = build_module(compile_plan(model_source(3, 2, "return [1,2];"))).double()
    x = torch.zeros(2, 5, 3, dtype=torch.float64)
    torch.testing.assert_close(model(x), torch.tensor([1.0, 2.0], dtype=torch.float64).expand(2, 5, 2))
    assert list(model.parameters()) == []
    assert all(buffer.dtype == torch.float64 for buffer in model.buffers())
    torch.testing.assert_close(model(torch.zeros(3, dtype=torch.float64)), torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_mean_sum_and_detach_reduce_model_axes_only(compile_plan):
    source = model_source(6, 1, """
        let matrix = x as [2,3];
        let energy = mean(matrix * matrix);
        return energy + detach(sum(matrix));
    """)
    model = build_module(compile_plan(source))
    x = torch.linspace(-1, 1, 36).reshape(2, 3, 6).requires_grad_()
    expected = x.square().mean(-1, keepdim=True) + x.sum(-1, keepdim=True).detach()
    assert_value_and_gradient(model(x), expected, (x,))


def test_input_contract_and_unsupported_ir_fail_clearly(compile_plan):
    definition = compile_plan(model_source(2, 2, "return x;"))
    model = build_module(definition)
    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.zeros(2, 3))
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(2, dtype=torch.int64))
    with pytest.raises(TypeError, match="torch.Tensor"):
        model([1.0, 2.0])
    malformed = deepcopy(definition)
    malformed["program"]["registers"][0]["shape"] = [3]
    with pytest.raises(ValueError, match="shape/length mismatch"):
        build_module(malformed)
    unsupported = deepcopy(definition)
    unsupported["program"]["instructions"].insert(0, {"op": "write"})
    with pytest.raises(ValueError, match="fields for write"):
        build_module(unsupported)
    malformed = deepcopy(definition)
    malformed["program"]["version"] = 999
    with pytest.raises(ValueError, match="IR version"):
        build_module(malformed)


def test_precompiled_parameter_shapes_and_learn_domains_are_checked(compile_plan):
    definition = compile_plan(model_source(1, 1, "return x * p;", "parameter p: f32[1] = [2]"))
    malformed = deepcopy(definition)
    malformed["program"]["parameters"][0]["initializer"] = {"op": "full", "shape": [2], "value": 1}
    with pytest.raises(ValueError, match="shape mismatch"):
        build_module(malformed)
    malformed = deepcopy(definition)
    malformed["program"]["parameters"][0]["learn"] = {"kind": "bool"}
    with pytest.raises(ValueError, match="outside its domain"):
        build_module(malformed)


def test_imported_model_composes_with_standard_fx_tracing(compile_plan):
    definition = compile_plan(model_source(2, 2, "return clip(x,-1,1);"))
    model = torch.nn.Sequential(build_module(definition), torch.nn.Linear(2, 1))
    graph = torch.fx.symbolic_trace(model)
    x = torch.tensor([[0.1, 2.0], [-0.5, 0.7]], requires_grad=True)
    assert_value_and_gradient(graph(x), model(x), (x, *model.parameters()))


def test_probability_validation_preserves_double_precision(compile_plan):
    source = model_source(2, 2, "return x;").replace("output y: f32[2]", "output y: probabilities[2]")
    model = build_module(compile_plan(source)).double()
    valid = torch.tensor([0.5, 0.50000999], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model(valid), valid, rtol=0, atol=0)
    model(valid).sum().backward()
    torch.testing.assert_close(valid.grad, torch.ones_like(valid))
    invalid = torch.tensor([0.5, 0.50001001], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="sum to 1"):
        model(invalid)


@pytest.mark.parametrize("name", ["state_dict", "graph"])
def test_torch_reserved_parameter_names_remain_registered(compile_plan, name):
    source = model_source(1, 1, f"return x * {name};", f"parameter {name}: f32[1] = [3]")
    model = build_module(compile_plan(source))
    assert model.parameter_names[name] != name
    assert list(model.state_dict()) == [model.parameter_names[name]]
    assert isinstance(model.graph, torch.fx.Graph)
    torch.testing.assert_close(model(torch.tensor([2.0])), torch.tensor([6.0]))
