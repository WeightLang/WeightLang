"""Typed model acceptance through the real Rust compiler and Python SDK."""

from copy import deepcopy
import os
from pathlib import Path
import shutil

import pytest
import torch

from weightlang import torch_api as wgt
from weightlang.operations import OperationsModule


ROOT = Path(__file__).resolve().parents[2]
SOURCE = """wgt 0.5
model Cell
# Two independent state coordinates persist across successful operations.
state activity: f32[2] = [0,0];
parameter gain: f32[2] = [1,1]
frozen fixed: f32[2] = [2,2]
fn forward(left: f32[2], right: f32[2]) -> f32[2] {
    return activity + left * gain + right;
}
fn observe() -> f32[2] { return activity; }
fn advance(input: f32[2]) -> f32[2] {
    activity <- activity + input;
    return activity;
}
# Success publishes the two state writes and the parameter write together.
fn twice(input: f32[2]) -> f32[2] {
    activity <- input;
    activity <- activity + input;
    gain <- activity;
    return gain;
}
fn update(input: f32[2]) { gain <- input; }
fn reset() { activity <- [0,0]; }
# A failed result must discard every earlier candidate write.
fn bad(input: f32[2]) -> f32[2] {
    activity <- input;
    gain <- input;
    return log([0,0]);
}
fn stalled(input: f32[2]) -> f32[2] {
    activity <- input;
    while positive([1]) { activity <- activity + input; }
    return activity;
}
fn state_dict(input: f32[2]) -> f32[2] { return input; }
"""


@pytest.fixture
def compiler(monkeypatch):
    executable = os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt"))
    assert Path(executable).is_file(), "Build the native wgt compiler before SDK acceptance tests"
    monkeypatch.setenv("WGT_COMPILER", executable)
    return executable


@pytest.fixture
def definition(compiler):
    return wgt.compile_source(SOURCE)


@pytest.fixture
def model(definition):
    return wgt.from_ir(definition, device="cpu")


def assert_storage(model, expected):
    assert model.state_dict().keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_typed_forward_multiple_inputs_and_void_updates(model):
    assert isinstance(model, OperationsModule)
    assert not isinstance(model, torch.fx.GraphModule)
    assert model.operation_schemas["forward"]["inputs"] == [
        {"name": "left", "width": 2}, {"name": "right", "width": 2},
    ]
    assert model.update(torch.tensor([2.0, 3.0])) is None
    torch.testing.assert_close(model(torch.tensor([4.0, 5.0]), torch.tensor([1.0, -1.0])),
                               torch.tensor([9.0, 14.0]))
    assert all(not value.requires_grad for value in model.parameters())
    assert model.state_names["activity"] in dict(model.named_buffers())
    assert model.parameter_names["gain"] in dict(model.named_parameters())


def test_state_transactions_read_after_write_reset_and_isolation(model, definition):
    other = wgt.from_ir(definition, device="cpu")
    first = model.advance(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(other.observe(), torch.zeros(2))
    torch.testing.assert_close(model.twice(torch.tensor([3.0, 4.0])), torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(model.observe(), torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(getattr(model, model.parameter_names["gain"]), torch.tensor([6.0, 8.0]))
    # Returned values remain snapshots after later calls mutate shared storage.
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))
    assert model.reset() is None
    torch.testing.assert_close(model.observe(), torch.zeros(2))
    torch.testing.assert_close(getattr(model, model.parameter_names["gain"]), torch.tensor([6.0, 8.0]))


def test_nonfinite_result_and_exhausted_work_roll_back_all_storage(model):
    model.twice(torch.tensor([1.0, 2.0]))
    expected = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="log.*nonfinite"):
        model.bad(torch.tensor([7.0, 8.0]))
    assert_storage(model, expected)
    with pytest.raises(ValueError, match="budget"):
        model.call("stalled", torch.ones(2), max_work=50)
    assert_storage(model, expected)
    torch.testing.assert_close(model.advance(torch.ones(2)), torch.tensor([3.0, 5.0]))


def test_memory_budget_failure_keeps_parameters_and_states(model):
    expected = deepcopy(model.state_dict())
    model.memory_bytes = 0
    with pytest.raises(ValueError, match="memory budget"):
        model.twice(torch.ones(2))
    assert_storage(model, expected)


@pytest.mark.parametrize("value", [torch.ones(1), torch.ones(1, 2), torch.ones(2, dtype=torch.int64),
                                  torch.ones(2, dtype=torch.float64), torch.tensor([float("nan"), 1.0])])
def test_invalid_input_is_rejected_before_writes(model, value):
    expected = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="shape|dtype/device|finite"):
        model.advance(value)
    assert_storage(model, expected)


def test_no_cross_state_autograd_and_grad_input_rejection(model):
    expected = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="autograd"):
        model.advance(torch.ones(2, requires_grad=True))
    assert_storage(model, expected)
    assert not model(torch.ones(2), torch.zeros(2)).requires_grad
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="autograd"):
        model.observe()
    assert_storage(model, expected)


def test_reserved_method_names_are_available_through_call(model):
    assert isinstance(model.state_dict(), dict)
    value = torch.tensor([2.0, 5.0])
    torch.testing.assert_close(model.call("state_dict", value), value)
    with pytest.raises(ValueError, match="unknown model operation"):
        model.call("not_defined")
    with pytest.raises(ValueError, match="expects 2 input vectors"):
        model(torch.ones(2))


def test_checkpoint_restores_state_and_assigned_tensor_precision(model, compiler, tmp_path, monkeypatch):
    model.double()
    model.twice(torch.tensor([2.0, 4.0], dtype=torch.float64))
    path = wgt.save(model, tmp_path / "cell.pt")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "absent"))
    restored = wgt.load(path, device="cpu")
    assert_storage(restored, model.state_dict())
    torch.testing.assert_close(restored.advance(torch.ones(2, dtype=torch.float64)),
                               torch.tensor([5.0, 9.0], dtype=torch.float64))
    torch.testing.assert_close(model.observe(), torch.tensor([4.0, 8.0], dtype=torch.float64))


def test_copies_are_independent_and_state_dict_contains_only_declared_storage(model):
    model.advance(torch.tensor([1.0, 3.0]))
    duplicate = deepcopy(model)
    duplicate.advance(torch.ones(2))
    torch.testing.assert_close(model.observe(), torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(duplicate.observe(), torch.tensor([2.0, 4.0]))
    assert set(model.state_dict()) == set(model.parameter_names.values()) | set(model.state_names.values())


def test_zero_storage_and_void_only_models_follow_device_and_dtype(compiler):
    plain = wgt.loads("""wgt 0.5
model Plain
fn combine(left: f32[2], right: f32[2]) -> f32[2] { return left + right; }
fn noop() {}
""", device="cpu").double()
    assert not plain.state_dict()
    torch.testing.assert_close(plain.combine(torch.ones(2, dtype=torch.float64),
                                            torch.full((2,), 2.0, dtype=torch.float64)),
                               torch.full((2,), 3.0, dtype=torch.float64))
    assert plain.noop() is None
    assert not hasattr(plain, "reset")
    with pytest.raises(ValueError, match="no forward operation"):
        plain()


def test_shared_storage_names_do_not_replace_module_methods(compiler):
    model = wgt.loads("""wgt 0.5
model Reserved
state call: f32[1] = [1]
parameter state_dict: f32[1] = [2]
fn advance(x: f32[1]) -> f32[1] { call <- call + x; return call * state_dict; }
""", device="cpu")
    torch.testing.assert_close(model.advance(torch.tensor([2.0])), torch.tensor([6.0]))
    assert callable(model.call) and callable(model.state_dict)
    assert model.state_names["call"] != "call"
    assert model.parameter_names["state_dict"] != "state_dict"


def test_untrusted_ir_effects_and_forward_writes_are_rejected(definition):
    broken = deepcopy(definition)
    advance = next(entry for entry in broken["operations"]["entries"] if entry["schema"]["name"] == "advance")
    advance["schema"]["effects"]["writes_state"] = []
    with pytest.raises(ValueError, match="effects"):
        wgt.from_ir(broken, device="cpu")
    broken = deepcopy(definition)
    broken["operations"]["entries"] = [entry for entry in broken["operations"]["entries"]
                                         if entry["schema"]["name"] != "forward"]
    next(entry for entry in broken["operations"]["entries"]
         if entry["schema"]["name"] == "advance")["schema"]["name"] = "forward"
    with pytest.raises(ValueError, match="forward must not update"):
        wgt.from_ir(broken, device="cpu")


def test_typed_learn_updates_validate_only_the_final_transaction(compiler):
    model = wgt.loads("""wgt 0.5
model DiscreteUpdates
learn gain: int in 0..=3 = 1;
state count: f32[1] = [0]
fn set(value: f32[1]) { count <- count + [1]; gain <- value; }
# The native transaction can temporarily leave the declared domain. Reads still
# use the hard integer value; only the final published value must fit the domain.
fn bounce(value: f32[1]) -> f32[1] {
    gain <- value;
    let candidate = gain;
    gain <- [1];
    return candidate;
}
fn observe() -> f32[1] { return gain; }
""", device="cpu")
    assert model.set(torch.tensor([2.0])) is None
    torch.testing.assert_close(model.observe(), torch.tensor([2.0]))
    expected = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="domain"):
        model.set(torch.tensor([2.5]))
    assert_storage(model, expected)
    torch.testing.assert_close(model.bounce(torch.tensor([4.5])), torch.tensor([5.0]))
    torch.testing.assert_close(model.observe(), torch.tensor([1.0]))


def test_ir_visualization_shows_control_and_storage_without_running(model):
    model.max_work = 1
    expected = deepcopy(model.state_dict())
    mermaid = wgt.visualize(model, entry="stalled")
    dot = wgt.visualize(model, format="dot", entry="advance")
    assert "WGT compiler IR" in mermaid
    assert "state activity" in mermaid and "Cell.stalled" in mermaid
    assert "branch" in mermaid and "jump" in mermaid and "write" in mermaid
    assert "Cell.advance" in dot and "style=dashed" in dot
    assert "Cell.bad" not in dot
    assert_storage(model, expected)
    with pytest.raises(ValueError, match="no single static FX graph"):
        wgt.trace(model)
    with pytest.raises(ValueError, match="unknown WGT entry"):
        wgt.visualize(model, entry="missing")


def test_stateful_ir_renders_as_svg_with_real_graphviz(model):
    pytest.importorskip("graphviz")
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")
    svg = wgt.visualize(model, format="svg", entry="stalled")
    assert "<svg" in svg and "Cell.stalled" in svg and "activity" in svg


def test_forward_control_flow_and_custom_loss_are_inspectable(compiler):
    model = wgt.loads("""wgt 0.5
model Branch
input x: f32[2]
output y: f32[2]
parameter gain: f32[2] = [2,3]
fn forward(x) {
    if positive(mean(x)) { return x * gain; }
    return -x;
}
fn loss(prediction, target) { return 0.5 * square(prediction - target).mean; }
""", device="cpu")
    assert "branch" in wgt.visualize(model)
    loss_graph = wgt.visualize(model, entry="loss")
    assert "Branch.loss" in loss_graph and "Branch.forward" not in loss_graph
    with pytest.raises(ValueError, match="no single static FX graph"):
        wgt.trace(model)


def test_summary_with_inputs_preserves_state_on_its_private_copy(model):
    pytest.importorskip("torchinfo")
    expected = deepcopy(model.state_dict())
    report = wgt.summary(model, input_data=[torch.ones(2), torch.zeros(2)])
    assert report.total_params == 4
    assert report.trainable_params == 0
    assert_storage(model, expected)
