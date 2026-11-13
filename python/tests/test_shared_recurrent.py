"""Two shared GRU steps: SDK/native VJPs and independent nn.GRUCell parity."""

import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

import weightlang as wgt

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "examples/native_training/shared_recurrent/model.wgt"
# Row-major PyTorch (reset, update, new) layout; no zeros, including b_hn.
BINDINGS = {
    "weight_ih": [.2, -.1, .3, .4, -.2, .5, .1, -.3, .6, .2, -.4, .3],
    "weight_hh": [-.3, .2, .1, -.4, .5, .3, -.2, .6, .4, -.5, .2, .1],
    "bias_ih": [.11, -.07, .03, .13, -.17, .19],
    "bias_hh": [-.05, .09, -.12, .08, .23, -.21],
}
INPUTS = [
    [.7, -.4, -.2, .6, .3, -.5],
    [-.6, .8, .5, -.3, -.4, .2],
]
TARGET = [.15, -.25]
SEED = [.7, -.4]


@pytest.fixture(scope="module")
def definition():
    # Compile the real file so std imports go through the shared frontend.
    # Callers select prebuilt tools through WGT_COMPILER / WGT_REFERENCE.
    return wgt.compile(MODEL)


def native(definition, **request):
    executable = os.environ.get(
        "WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference")
    )
    process = subprocess.run(
        [executable],
        input=json.dumps({"source": definition["source"], "bindings": BINDINGS, **request}),
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def pair(definition):
    model = wgt.from_ir(definition, device="cpu")
    cell = torch.nn.GRUCell(2, 2, device="cpu")
    assert set(model.parameter_names) == set(BINDINGS)
    assert sum(p.numel() for p in model.parameters()) == 36
    with torch.no_grad():
        for name, values in BINDINGS.items():
            parameter = model.get_parameter(model.parameter_names[name])
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
            reference = getattr(cell, name)
            reference.copy_(torch.tensor(values).reshape(reference.shape))
    return model, cell


def unroll(cell, packed):
    # One nn.GRUCell instance is called twice: autograd sums both uses' VJPs.
    hidden1 = cell(packed[:2], packed[4:6])
    return cell(packed[2:4], hidden1)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("inputs", INPUTS)
@pytest.mark.parametrize("objective", ["seed", "loss"])
def test_outputs_and_input_and_shared_parameter_gradients(definition, inputs, objective):
    model, cell = pair(definition)
    x = torch.tensor(inputs, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    output, reference_output = model(x), unroll(cell, reference_x)
    request = {"gradient": SEED} if objective == "seed" else {"target": TARGET}
    expected = native(definition, mode="forward", input=inputs, **request)
    close(output, reference_output)
    close(output, torch.tensor(expected["output"]))
    if objective == "seed":
        output.backward(torch.tensor(SEED))
        reference_output.backward(torch.tensor(SEED))
    else:
        loss = model.loss(output, torch.tensor(TARGET))
        reference_loss = torch.nn.functional.mse_loss(reference_output, torch.tensor(TARGET))
        close(loss.reshape(()), reference_loss)
        close(loss.reshape(()), torch.tensor(expected["loss"]))
        loss.backward()
        reference_loss.backward()
    close(x.grad, reference_x.grad)
    close(x.grad, torch.tensor(expected["input_gradient"]))
    # Check every slot (including both biases), not just aggregate gradient norms.
    for name, values in BINDINGS.items():
        parameter = model.get_parameter(model.parameter_names[name])
        close(parameter.grad.flatten(), getattr(cell, name).grad.flatten())
        close(parameter.grad.flatten(), torch.tensor(expected["gradients"][name]))
        torch.testing.assert_close(parameter.detach().flatten(), torch.tensor(values), atol=0, rtol=0)


def test_sgd_step_and_checkpoint_roundtrip(definition, tmp_path, monkeypatch):
    model, cell = pair(definition)
    x, target = torch.tensor(INPUTS[0]), torch.tensor(TARGET)
    rate = .05
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    wgt.bind_optimizer(model, optimizer)
    reference_optimizer = torch.optim.SGD(cell.parameters(), lr=rate)
    before = model.loss(model(x), target)
    reference_loss = torch.nn.functional.mse_loss(unroll(cell, x), target)
    before.backward()
    reference_loss.backward()
    optimizer.step()
    reference_optimizer.step()
    expected = native(
        definition, mode="train", inputs=[INPUTS[0]], targets=[TARGET], rate=rate, steps=1
    )
    close(before.detach().reshape(()), torch.tensor(expected["loss"]))
    for name in BINDINGS:
        parameter = model.get_parameter(model.parameter_names[name])
        close(parameter.flatten(), getattr(cell, name).flatten())
        close(parameter.flatten(), torch.tensor(expected["parameters"][name]))
    trained = model(x).detach()
    close(trained, unroll(cell, x).detach())
    assert model.loss(trained, target).item() < before.item()

    path = wgt.save(model, tmp_path / "shared_recurrent.pt")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "missing-compiler"))
    restored = wgt.load(path, device="cpu")
    assert restored.parameter_names == model.parameter_names
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    torch.testing.assert_close(restored(x), trained, atol=0, rtol=0)
    # The saved model includes trained parameters; hidden remains a host input.
    alternate = torch.tensor(INPUTS[1])
    torch.testing.assert_close(restored(alternate), model(alternate), atol=0, rtol=0)
