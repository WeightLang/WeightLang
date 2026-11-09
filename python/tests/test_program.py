"""Execution contracts around native control flow, storage, and learn reads."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

import weightlang as wgt
from weightlang.program import ProgramExecutor, ProgramModule, SUPPORTED_OPS, project_value


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def compiler(monkeypatch):
    executable = os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt"))
    assert Path(executable).is_file(), "Build the wgt compiler before SDK acceptance"
    monkeypatch.setenv("WGT_COMPILER", executable)


def source(body, declarations=""):
    return f"wgt 0.5\nmodel Execution\ninput x:f32[1]\noutput y:f32[1]\n{declarations}\nfn forward(x) {{ {body} }}"


def test_executor_instruction_coverage_matches_native_catalog():
    catalog = json.loads((ROOT / "language/tensor_contracts.json").read_text())
    assert SUPPORTED_OPS == {item["name"] for item in catalog["operators"]}


def test_loaded_program_and_executor_snapshot_caller_metadata():
    definition = wgt.compile_source(source("if positive(x) { return x*gain; } return -x;", "parameter gain:f32[1]=[2]"))
    model = wgt.from_ir(definition, device="cpu")
    assert isinstance(model, ProgramModule)
    executor = ProgramExecutor(definition["program"])
    definition["program"]["instructions"].clear()
    definition["program"]["parameters"][0]["initializer"] = {"op": "full", "shape": [1], "value": 999}
    x = torch.tensor([3.0], requires_grad=True)
    torch.testing.assert_close(model(x), torch.tensor([6.0]))
    result = executor.run(x, {"gain": torch.tensor([4.0])}, differentiable=True)
    torch.testing.assert_close(result.output, torch.tensor([12.0]))
    assert model.definition["program"]["instructions"]


@pytest.mark.parametrize("location", ["program", "register", "parameter"])
def test_unknown_native_ir_fields_are_rejected(location):
    definition = wgt.compile_source(source("return x*gain;", "parameter gain:f32[1]=[2]"))
    program = definition["program"]
    item = program if location == "program" else program[location + "s"][0]
    item["unsupported_future_field"] = True
    with pytest.raises(ValueError, match="fields"):
        wgt.from_ir(definition, device="cpu")


@pytest.mark.parametrize("with_parameters", [False, True])
def test_loss_cannot_capture_parameters_or_nest_another_loss(with_parameters):
    text = source("return x*gain;", "parameter gain:f32[1]=[2]")
    text += "\nfn loss(prediction,target) { return square(prediction-target).mean; }"
    definition = wgt.compile_source(text)
    loss = definition["program"]["loss"]
    if with_parameters:
        loss["parameters"] = deepcopy(definition["program"]["parameters"])
    else:
        loss["loss"] = deepcopy(loss)
    with pytest.raises(ValueError, match="parameter-free.*nested loss"):
        wgt.from_ir(definition, device="cpu")


@pytest.mark.parametrize("dynamic", [False, True])
def test_empty_batches_are_rejected_for_both_execution_paths(dynamic):
    body = "if positive(x) { return x; } return -x;" if dynamic else "return x;"
    model = wgt.loads(source(body), device="cpu")
    with pytest.raises(ValueError, match="nonempty batch"):
        model(torch.empty(0, 1))


def test_integer_read_does_not_round_an_already_exact_large_integer_up():
    model = wgt.loads(source("return x*count;", "learn count:int in 0..=16777216=16777215;"), device="cpu")
    value = model(torch.ones(1))
    torch.testing.assert_close(value, torch.tensor([16777215.0]), rtol=0, atol=0)
    value.backward()
    torch.testing.assert_close(model.get_parameter(model.parameter_names["count"]).grad, torch.ones(1))


def test_positive_projection_keeps_native_f32_successor_in_float64():
    expected = torch.nextafter(torch.tensor(0.25), torch.tensor(float("inf"))).double()
    actual = project_value(torch.tensor([-1.0], dtype=torch.float64), {"kind": "positive", "low": 0.25})
    torch.testing.assert_close(actual, expected.reshape(1), rtol=0, atol=0)


def test_lazy_route_obeys_grad_mode_independently_of_training_flag():
    model = wgt.loads(source("return route([1],x,log(-x));"), device="cpu")
    model.train()
    with torch.no_grad():
        torch.testing.assert_close(model(torch.ones(1)), torch.ones(1))
    model.eval()
    with pytest.raises(RuntimeError, match="log.*nonfinite"):
        model(torch.ones(1))


def test_workspace_admission_and_use_after_free_roll_back_writes():
    model = wgt.loads(source("""
        write(state,x);
        let bank=memory_alloc(2,1);
        memory_write(bank,[1,0],x);
        let output=memory_read(bank,[1,0]);
        memory_free(bank);
        return output;
    """, "parameter state:f32[1]=[1]"), device="cpu")
    parameter = model.get_parameter(model.parameter_names["state"])
    required = model._executor.managed_bytes + 8
    with pytest.raises(RuntimeError, match="workspace memory budget"):
        model.run(torch.tensor([3.0]), memory_bytes=required - 1)
    torch.testing.assert_close(parameter, torch.ones(1))
    torch.testing.assert_close(model.run(torch.tensor([3.0]), memory_bytes=required), torch.tensor([3.0]))
    torch.testing.assert_close(parameter, torch.tensor([3.0]))
    invalid = wgt.loads(source("""
        write(state,x);
        let bank=memory_alloc(1,1);
        memory_free(bank);
        return memory_read(bank,[1]);
    """, "parameter state:f32[1]=[1]"), device="cpu")
    with pytest.raises(RuntimeError, match="memory_read.*freed"):
        invalid.run(torch.tensor([7.0]))
    torch.testing.assert_close(invalid.get_parameter(invalid.parameter_names["state"]), torch.ones(1))


def test_embedded_graph_recipes_preserve_native_initializers_exactly():
    definition = wgt.compile(ROOT / "python/examples/mlp/model.wgt")
    native = wgt.from_ir(definition, device="cpu")
    del definition["graph_parameters"]
    before = torch.random.get_rng_state()
    restored = wgt.from_ir(definition, device="cpu")
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    for name, value in native.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_backward_rejects_nonfinite_active_gradients_and_preserves_zero_paths(dynamic):
    body = "if positive(x) { return log(x); } return x;" if dynamic else "return log(x);"
    model = wgt.loads(source(body), device="cpu")
    x = torch.tensor([torch.finfo(torch.float32).smallest_normal * torch.finfo(torch.float32).eps], requires_grad=True)
    model(x).backward(torch.zeros(1))
    torch.testing.assert_close(x.grad, torch.zeros(1))
    x.grad = None
    with pytest.raises(RuntimeError, match="nonfinite WeightLang gradient"):
        model(x).backward(torch.ones(1))


@pytest.mark.parametrize("dynamic", [False, True])
def test_diagnostic_anchor_preserves_frozen_storage_and_module_lifecycle(dynamic, tmp_path):
    body = "if positive(x) { return x*gain; } return -x;" if dynamic else "return x*gain;"
    model = wgt.loads(source(body, "frozen gain:f32[1]=[2]"), device="cpu").to(torch.float64)
    assert model._wgt_gradient_anchor.is_leaf
    assert model._wgt_gradient_anchor.requires_grad
    assert "_wgt_gradient_anchor" not in model.state_dict()
    clone = deepcopy(model)
    x = torch.ones(1, dtype=torch.float64, requires_grad=True)
    clone(x).backward()
    torch.testing.assert_close(x.grad, torch.tensor([2.0], dtype=torch.float64))
    assert clone.get_parameter(clone.parameter_names["gain"]).requires_grad is False
    assert clone.get_parameter(clone.parameter_names["gain"]).grad is None
    assert clone._wgt_gradient_anchor.grad is None
    with torch.no_grad():
        assert not clone(x).requires_grad
    wgt.summary(model, input_data=torch.ones(1, dtype=torch.float64))
    path = wgt.save(model, tmp_path / "frozen.pt")
    restored = wgt.load(path, device="cpu")
    assert restored._wgt_gradient_anchor.is_leaf
    assert "_wgt_gradient_anchor" not in restored.state_dict()
    restored(torch.ones(1, dtype=torch.float64)).backward()
    assert restored.get_parameter(restored.parameter_names["gain"]).grad is None
    assert restored._wgt_gradient_anchor.grad is None


def test_constant_only_program_retains_diagnostics_without_trainable_parameters():
    model = wgt.loads(source("return [2];"), device="cpu")
    assert list(model.parameters()) == []
    model(torch.ones(1)).backward()
    assert model._wgt_gradient_anchor.grad is None
    for name, buffer in model.named_buffers():
        if name != "_wgt_gradient_anchor":
            assert not buffer.requires_grad and buffer.grad is None
    invalid = wgt.loads(source("return log([1.401298464324817e-45]);"), device="cpu")
    with pytest.raises(RuntimeError, match="nonfinite WeightLang gradient"):
        invalid(torch.ones(1)).backward()


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("returns_parameter", [False, True])
def test_inference_aliases_do_not_expose_input_or_parameter_gradients(context, returns_parameter):
    body = "return gain;" if returns_parameter else "return x;"
    declarations = "parameter gain:f32[1]=[2]" if returns_parameter else ""
    model = wgt.loads(source(body, declarations), device="cpu")
    x = torch.ones(1, requires_grad=True)
    original = model.get_parameter(model.parameter_names["gain"]) if returns_parameter else x
    with context():
        result = model(x)
    torch.testing.assert_close(result, original)
    assert not result.requires_grad and result.grad_fn is None
    assert original.requires_grad and original.grad is None
    model(x).sum().backward()
    torch.testing.assert_close(original.grad, torch.ones_like(original))
