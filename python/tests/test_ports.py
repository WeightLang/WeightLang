"""Logical ports, using real Rust packed IR and a native gradient oracle."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import torch
from weightlang import torch_api as wgt
from weightlang.ports import ForwardModule

ROOT = Path(__file__).resolve().parents[2]
SOURCE = """wgt 0.5
model Ports
parameter gain: f32[2] = [0.5, 0.75]
parameter unused: f32[1] = [7]
fn forward(left: f32[1,2], right: f32[2]) -> (f32[1,2], f32[1,2], f32[2]) {
    let y = (((left as [2]) + right) * gain) as [1,2];
    return (y, y, right);
}
"""
# Standalone native vector oracle also lets adapter tests run independently of
# the CLI metadata rollout. WGT is compiled by Rust, never parsed in Python.
PACKED_SOURCE = """wgt 0.5
model PackedPorts
input x: f32[4]
output y: f32[6]
parameter gain: f32[2] = [0.5, 0.75]
parameter unused: f32[1] = [7]
fn forward(x) {
    # Matrix projections form an independent packed-vector oracle for tuple
    # lowering: merge the two inputs, duplicate y, then append the raw right.
    let merged = (x as [1,4]) @ ([1,0,0,1,1,0,0,1] as [4,2]);
    let y = (merged as [2]) * gain;
    let repeated = (y as [1,2]) @ ([1,0,1,0,0,0,0,1,0,1,0,0] as [2,6]);
    let direct = (x as [1,4]) @ ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1] as [4,6]);
    return (repeated + direct) as [6];
}
"""
PORTS = {"inputs": [{"name": "left", "shape": [1, 2], "offset": 0, "width": 2},
                    {"name": "right", "shape": [2], "offset": 2, "width": 2}],
         "outputs": [{"shape": [1, 2], "offset": 0, "width": 2},
                     {"shape": [1, 2], "offset": 2, "width": 2},
                     {"shape": [2], "offset": 4, "width": 2}],
         "input": 4, "output": 6, "tuple": True}


@pytest.fixture
def definition():
    result = wgt.compile_source(PACKED_SOURCE)
    result["forward_interface"] = deepcopy(PORTS)
    return result


def values(dtype=torch.float32):
    return (torch.tensor([[.2, -.3]], dtype=dtype, requires_grad=True),
            torch.tensor([.4, .7], dtype=dtype, requires_grad=True))


def test_native_packed_oracle_and_repeated_output_gradient(definition):
    model = wgt.from_ir(definition, device="cpu")
    left, right = values()
    outputs = model(left, right)
    # The third field is unused by the loss; the first two repeat the same y.
    seed = torch.tensor([1., 2., 3., 4., 0., 0.])
    (torch.cat([v.flatten() for v in outputs]) * seed).sum().backward()
    executable = os.environ.get("WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference"))
    result = subprocess.run([executable], input=json.dumps({
        "source": PACKED_SOURCE, "mode": "forward",
        "input": torch.cat([left.detach().flatten(), right.detach()]).tolist(),
        "gradient": seed.tolist(),
    }), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    expected = json.loads(result.stdout)
    torch.testing.assert_close(torch.cat([v.flatten() for v in outputs]), torch.tensor(expected["output"]))
    torch.testing.assert_close(torch.cat([left.grad.flatten(), right.grad]), torch.tensor(expected["input_gradient"]))
    for name, attr in model.parameter_names.items():
        parameter = model.get_parameter(attr)
        gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        torch.testing.assert_close(gradient.flatten(), torch.tensor(expected["gradients"][name]))
    # Contributions from both repeated outputs reach the same gain object.
    gain = model.get_parameter(model.parameter_names["gain"])
    torch.testing.assert_close(gain.grad, torch.tensor([2.4, 2.4]))


def test_finite_difference_host_loss(definition):
    model = wgt.from_ir(definition, device="cpu", dtype=torch.float64)
    left, right = values(torch.float64)
    def loss(a, b):
        y, repeated, unused = model(a, b)
        return y.square().sum() + repeated.sum()
    loss(left, right).backward()
    eps = 1e-5
    for index, value in enumerate((left, right)):
        for element in range(value.numel()):
            plus = [left.detach().clone(), right.detach().clone()]
            minus = [v.clone() for v in plus]
            plus[index].view(-1)[element] += eps
            minus[index].view(-1)[element] -= eps
            finite_difference = (loss(*plus) - loss(*minus)) / (2 * eps)
            torch.testing.assert_close(value.grad.flatten()[element], finite_difference)


def test_kwargs_identity_optimizer_and_hidden_chain(definition):
    model = wgt.from_ir(definition, device="cpu")
    assert isinstance(model, ForwardModule)
    assert model.definition is model.base.definition
    parameters = tuple(model.parameters())
    assert len(parameters) == 2
    assert len({id(p) for p in model.parameters()}) == 2
    assert all(key.startswith("base.") for key in model.state_dict())
    assert all(model.get_parameter(attr) is model.base.get_parameter(attr.removeprefix("base."))
               for attr in model.parameter_names.values())
    left, right = values()
    torch.testing.assert_close(model(right=right, left=left), model(left, right))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    handle = model.bind_optimizer(optimizer)
    hidden = right
    losses = []
    for _ in range(3):
        y, hidden_row, _ = model(left, hidden)
        hidden = hidden_row.reshape(2)  # Remains connected across host steps.
        losses.append(y.square().sum())
    sum(losses).backward()
    reference_left = left.detach().clone().requires_grad_()
    reference_right = right.detach().clone().requires_grad_()
    reference_gain = model.get_parameter(model.parameter_names["gain"]).detach().clone().requires_grad_()
    reference_hidden = reference_right
    reference_losses = []
    for _ in range(3):
        reference_hidden = (reference_left.flatten() + reference_hidden) * reference_gain
        reference_losses.append(reference_hidden.square().sum())
    sum(reference_losses).backward()
    torch.testing.assert_close(left.grad, reference_left.grad)
    torch.testing.assert_close(right.grad, reference_right.grad)
    torch.testing.assert_close(model.get_parameter(model.parameter_names["gain"]).grad, reference_gain.grad)
    optimizer.step()
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    handle.remove()
    assert not hasattr(model, "loss")  # Host loss is explicit.


def test_single_tuple_field_and_unused_output_seed(definition):
    definition["forward_interface"]["outputs"] = [{"shape": [2, 3], "offset": 0, "width": 6}]
    model = wgt.from_ir(definition, device="cpu")
    result = model(*values())
    assert isinstance(result, tuple) and len(result) == 1 and result[0].shape == (2, 3)


def test_saved_wrapper_roundtrip_without_initializers(definition, tmp_path, monkeypatch):
    model = wgt.from_ir(definition, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        model.get_parameter(model.parameter_names["gain"]).fill_(1.25)
    path = wgt.save(model, tmp_path / "ports.pt")
    monkeypatch.setenv("WGT_COMPILER", "/missing/compiler")
    with patch("weightlang.initialization.recipe_value", side_effect=AssertionError("initializer ran")):
        restored = wgt.load(path, device="cpu")
    assert isinstance(restored, ForwardModule)
    assert restored.parameter_names == model.parameter_names
    assert tuple(restored.state_dict()) == tuple(model.state_dict())
    assert next(restored.parameters()).dtype == torch.float64
    torch.testing.assert_close(restored(*values(torch.float64)), model(*values(torch.float64)))
    # Compiled JSON loading follows the same wrapping branch.
    json_path = tmp_path / "ports.json"
    json_path.write_text(json.dumps(definition))
    assert isinstance(wgt.load(json_path, device="cpu"), ForwardModule)


def test_input_contracts(definition):
    model = wgt.from_ir(definition, device="cpu")
    left, right = values()
    for call in (lambda: model(left), lambda: model(left, right=right),
                 lambda: model(left=left, wrong=right)):
        with pytest.raises(TypeError):
            call()
    for pair, message in (((left.flatten(), right), "logical shape"),
                          ((left.double(), right), "dtype/device"),
                          ((left, right.to("meta")), "dtype/device")):
        with pytest.raises(ValueError, match=message):
            model(*pair)
    with pytest.raises(TypeError, match="Tensor"):
        model([[.2, -.3]], right)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(tuple=False), lambda p: p.update(input=5),
    lambda p: p["inputs"][1].update(name="left"),
    lambda p: p["inputs"][0].update(shape=[2, 2]),
    lambda p: p["outputs"][1].update(offset=0),
    lambda p: p["outputs"][0].update(width=True),
])
def test_invalid_port_metadata(definition, mutation):
    mutation(definition["forward_interface"])
    with pytest.raises(ValueError, match="forward_interface"):
        wgt.from_ir(definition, device="cpu")


def test_dynamic_ports_share_parameter_schema(definition):
    from weightlang.dynamic import _schema
    assert set(_schema(definition)) == {"gain", "unused"}


def test_real_tuple_source_export_and_native(definition):
    compiled = wgt.compile_source(SOURCE)
    assert compiled["forward_interface"] == PORTS
    model = wgt.from_ir(compiled, device="cpu")
    torch.testing.assert_close(model(*values()), wgt.from_ir(definition, device="cpu")(*values()))


def test_save_packed_base_normalizes_port_layout(definition, tmp_path):
    model = wgt.from_ir(definition, device="cpu")
    path = wgt.save(model.base, tmp_path / "packed.pt")
    restored = wgt.load(path, device="cpu")
    assert isinstance(restored, ForwardModule)
    torch.testing.assert_close(restored(*values()), model(*values()))


def test_ports_source_loss_rejected(definition):
    model = wgt.from_ir(definition, device="cpu")
    model.base.definition["program"]["loss"] = {}
    with pytest.raises(ValueError, match="host loss"):
        ForwardModule(model.base)


def test_dynamic_port_runtime_with_rust_packed_ir(tmp_path):
    # Isolate Python graph ownership from the concurrent tuple/template CLI
    # rollout. Rust compiles ordinary packed source; this fixture supplies the
    # logical port metadata and stable template identity explicitly.
    from weightlang.dynamic import DynamicModule
    def compile_shape(shapes):
        rows = shapes["x"][0]
        width, total = rows * 2, rows * 2 + 2
        def identity_map(height, columns, input_start, output_start, count):
            data = [int(0 <= i - input_start < count and k - output_start == i - input_start)
                    for i in range(height) for k in range(columns)]
            return f"({data} as [{height},{columns}])"
        source = f"""wgt 0.5
model PackedHiddenFixture
input packed: f32[{total}]
output result: f32[{total}]
parameter weight: f32[2,2] = [0.2,-0.1,0.3,0.4]
fn forward(packed) {{
    let x = ((packed as [1,{total}]) @ {identity_map(total,width,0,0,width)}) as [{rows},2];
    let h = (packed as [1,{total}]) @ {identity_map(total,2,width,0,2)};
    let y = x @ (weight as [2,2]);
    let next_h = h + (scale([1,1],sum(y)) as [1,2]);
    let packed_y = (y as [1,{width}]) @ {identity_map(width,total,0,0,width)};
    let packed_h = next_h @ {identity_map(2,total,0,width,2)};
    return (packed_y + packed_h) as [{total}];
}}
"""
        definition = wgt.compile_source(source)
        definition["template_fingerprint"] = "a" * 64
        definition["forward_interface"] = {
            "inputs": [{"name": "x", "shape": [rows, 2], "offset": 0, "width": width},
                       {"name": "hidden", "shape": [1, 2], "offset": width, "width": 2}],
            "outputs": [{"shape": [rows, 2], "offset": 0, "width": width},
                        {"shape": [1, 2], "offset": width, "width": 2}],
            "input": total, "output": total, "tuple": True}
        return definition
    model = DynamicModule(compile_shape, input_shapes={"hidden": (1, 2), "x": (2, 2)}, device="cpu")
    parameter = next(model.parameters())
    reference_weight = parameter.detach().clone().requires_grad_()
    x1 = torch.arange(4, dtype=torch.float32).reshape(2, 2).requires_grad_()
    x2 = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_()
    hidden = torch.tensor([[.1, .2]], requires_grad=True)
    rx1, rx2, rh = [v.detach().clone().requires_grad_() for v in (x1, x2, hidden)]
    _, h1 = model(x1, hidden)
    with patch("weightlang.initialization.recipe_value", side_effect=AssertionError("initializer ran")):
        y2, h2 = model(hidden=h1, x=x2)
    ry1, ry2 = rx1 @ reference_weight, rx2 @ reference_weight
    rh2 = rh + ry1.sum() + ry2.sum()
    torch.testing.assert_close(h2, rh2)
    (h2.square().sum() + y2.square().sum()).backward()
    (rh2.square().sum() + ry2.square().sum()).backward()
    for actual, expected in zip((x1, x2, hidden, parameter), (rx1, rx2, rh, reference_weight)):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert next(model.parameters()) is parameter
    assert len(model.state_dict()) == 1
    for candidate, _ in model._cache.values():
        assert next(candidate.parameters()) is parameter

    # Dynamic checkpoints omit constants; the base static export keeps them.
    state = model.state_dict()
    model.load_state_dict(state, assign=True)
    assert not model._cache
    model(torch.ones(3, 2), torch.ones(1, 2))
    path = wgt.save(model.base, tmp_path / "dynamic-base.pt")
    restored = wgt.load(path, device="cpu")
    torch.testing.assert_close(restored(torch.ones(2, 2), torch.ones(1, 2)),
                               model.base(torch.ones(2, 2), torch.ones(1, 2)))
