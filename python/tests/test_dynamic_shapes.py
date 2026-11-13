"""Real shared compiler/native oracle acceptance; run after building both tools.

WGT_COMPILER and WGT_REFERENCE may select externally built binaries. No tests
build Rust or parse/transform WGT source in Python.
"""
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import torch
import weightlang as wgt
from weightlang import dynamic

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "examples/native_training/input_shapes/model.wgt"
SOURCE = MODEL.read_text()


def load(**kwargs):
    return wgt.loads_dynamic(SOURCE, input_shapes={"x": (2, 3)}, device="cpu", **kwargs)


def oracle(rows, value):
    executable = os.environ.get("WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference"))
    result = subprocess.run([executable], input=json.dumps({
        "source": SOURCE, "input_shapes": {"x": (rows, 3)},
        "input": value.detach().flatten().tolist(), "mode": "forward",
    }), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_native_each_shape_forward_and_gradients():
    model = wgt.load_dynamic(MODEL, input_shapes={"x": (2, 3)}, device="cpu")
    parameters = dict(model.named_parameters())
    for rows in (2, 5, 1, 5):
        model.zero_grad(set_to_none=True)
        # Noncontiguous inputs still flatten differentiably in logical order.
        x = torch.linspace(-.4, .8, rows * 3).reshape(3, rows).T.detach().requires_grad_()
        y = model(x)
        assert y.shape == (rows * 2,)
        y.sum().backward()
        expected = oracle(rows, x)
        torch.testing.assert_close(y, torch.tensor(expected["output"]), atol=2e-5, rtol=5e-5)
        torch.testing.assert_close(x.grad.flatten(), torch.tensor(expected["input_gradient"]), atol=2e-5, rtol=5e-5)
        for name, attr in model.parameter_names.items():
            torch.testing.assert_close(model.get_parameter(attr).grad.flatten(),
                                       torch.tensor(expected["gradients"][name]), atol=2e-5, rtol=5e-5)
        assert all(model.get_parameter(name) is p for name, p in parameters.items())


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
def test_optimizer_continuity_identity_and_no_initializer(optimizer_type):
    model = load()
    parameter = next(model.parameters())
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = optimizer_type(model.parameters(), lr=.03)
    expected_optimizer = optimizer_type([reference], lr=.03)
    handle = wgt.bind_optimizer(model, optimizer)
    rng = torch.random.get_rng_state().clone()
    # The schema-only path must never execute a recipe for a new graph.
    with patch("weightlang.initialization.recipe_value", side_effect=AssertionError("initializer ran")):
        for rows in (5, 2, 7, 5):
            x = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) / 10
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
            output = model(x)
            expected = (x @ reference.reshape(3, 2)).flatten()
            torch.testing.assert_close(output, expected)
            output.square().mean().backward()
            expected.square().mean().backward()
            optimizer.step()
            expected_optimizer.step()
            torch.testing.assert_close(parameter, reference)
            assert next(model.parameters()) is parameter
            assert optimizer.param_groups[0]["params"] == [parameter]
            for candidate, _ in model._cache.values():
                assert next(candidate.parameters()) is parameter
    handle.remove()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(model.state_dict()) == len(list(model.parameters())) == 1
    assert len(optimizer.state) <= 1


def test_cache_lru_failure_and_count_one():
    model = load(max_specializations=3)
    parameter = next(model.parameters())
    for rows in (3, 4, 3, 5):
        model(torch.ones(rows, 3))
    assert model.specialization_shapes == ((2, 3), (3, 3), (5, 3))
    before = parameter.detach().clone()
    shapes = model.specialization_shapes
    # Inference succeeds, but changing FEATURES changes parameter shape.
    with pytest.raises(ValueError, match="schema"):
        model(torch.ones(3, 4))
    assert model.specialization_shapes == shapes
    torch.testing.assert_close(parameter, before)
    # Actual Rust compilation failure also leaves the cache untouched.
    with pytest.raises(wgt.CompilationError):
        model(torch.ones(9))
    assert model.specialization_shapes == shapes
    single = load(max_specializations=1)
    single(torch.ones(3, 3))
    assert single.specialization_shapes == ((2, 3),)


def test_metadata_budget(monkeypatch):
    model = load()
    monkeypatch.setattr(dynamic, "_METADATA_BUDGET", model._base_bytes)
    model(torch.ones(3, 3))
    assert model.specialization_shapes == ((2, 3),)
    monkeypatch.setattr(dynamic, "_METADATA_BUDGET", 1)
    with pytest.raises(ValueError, match="budget"):
        model(torch.ones(4, 3))
    assert model.specialization_shapes == ((2, 3),)


def test_train_conversion_and_parameter_checkpoint(tmp_path):
    model = load()
    model(torch.ones(3, 3))
    model.eval()
    assert model.specialization_shapes == ((2, 3),)
    model(torch.ones(4, 3))
    assert all(not module.training for module, _ in model._cache.values())
    model.double()
    assert model.specialization_shapes == ((2, 3),)
    model(torch.ones(3, 3, dtype=torch.float64)).sum().backward()
    assert next(model.parameters()).grad.dtype == torch.float64
    with pytest.raises(ValueError, match="dtype/device"):
        model(torch.ones(3, 3))
    with pytest.raises(TypeError, match="DynamicModule"):
        wgt.save(model, tmp_path / "model.pt")
    assert not (tmp_path / "model.pt").exists()
    restored = load(dtype=torch.float64)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(torch.ones(5, 3, dtype=torch.float64)),
                               model(torch.ones(5, 3, dtype=torch.float64)))


@pytest.mark.parametrize("shapes", [{}, {"x": ()},
                                   {"x": (True, 3)}, {"x": (0, 3)}])
def test_invalid_input_shapes(shapes):
    with pytest.raises(ValueError, match="input_shapes"):
        wgt.loads_dynamic(SOURCE, input_shapes=shapes)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_cache_limit(count):
    with pytest.raises(ValueError, match="max_specializations"):
        load(max_specializations=count)


def test_dtype_device_and_input_validation():
    model = load()
    for value in (torch.ones(2, 3, dtype=torch.float64), torch.ones(2, 3, device="meta")):
        with pytest.raises(ValueError, match="dtype/device"):
            model(value)
    with pytest.raises(TypeError, match="Tensor"):
        model([[1., 2., 3.]])
    with pytest.raises(ValueError, match="input_shapes"):
        model(torch.ones(0, 3))


def test_schema_recipe_change_and_rejections():
    model = load()
    definition = wgt.compile_source(SOURCE, input_shapes={"x": (3, 3)})
    definition["program"]["parameters"][0]["initializer"] = {"op": "scalar", "value": 9}
    with patch.object(model, "_compile_shape", return_value=definition):
        with pytest.raises(ValueError, match="schema"):
            model(torch.ones(3, 3))
    assert not model._cache
    for kind in ("operations", "torch_module"):
        with pytest.raises(ValueError, match="ordinary"):
            dynamic._schema({"kind": kind})
    for instruction in ("write",):
        with pytest.raises(ValueError, match="writes"):
            dynamic._schema({"kind": "forward", "program": {"instructions": [{"op": instruction}]}})
    with pytest.raises(ValueError, match="state buffers"):
        dynamic._schema({"kind": "forward", "states": [{"name": "cache"}]})


def test_assign_checkpoint_invalidates_graphs():
    model = load()
    model(torch.ones(3, 3))
    state = {name: torch.full_like(value, .75) for name, value in model.state_dict().items()}
    model.load_state_dict(state, assign=True)
    assert not model._cache
    torch.testing.assert_close(model(torch.ones(3, 3)), torch.full((6,), 2.25))
    candidate, _ = model._cache[(3, 3)]
    assert next(candidate.parameters()) is next(model.parameters())


def test_projection_hook_runs_once_on_canonical_base():
    model = load()
    model(torch.ones(3, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    # The existing binder traverses registered children; candidates must never
    # add extra hook targets. Projection metadata belongs to the base alone.
    original = model.base.project_parameters_
    with patch.object(model.base, "project_parameters_", wraps=original) as project:
        handle = model.bind_optimizer(optimizer)
        model(torch.ones(4, 3)).sum().backward()
        optimizer.step()
        assert project.call_count == 1
        handle.remove()


@pytest.mark.parametrize("change", ["root", "import"])
def test_changed_source_fingerprint_rejected_before_build(tmp_path, change):
    helper = tmp_path / "ops.wgt"
    helper.write_text('wgt 0.5\noperators Local\nfn boost(x) { return 2*x; }\n')
    path = tmp_path / "model.wgt"
    source = SOURCE.replace("model ShapeBoundProjection",
                            'import ops from "./ops.wgt"\nmodel ShapeBoundProjection')
    source = source.replace('return x @ (weight as [FEATURES,OUTPUTS]);',
                            'return ops.boost(x @ (weight as [FEATURES,OUTPUTS]));')
    path.write_text(source)
    model = wgt.load_dynamic(path, input_shapes={"x": (2, 3)}, device="cpu")
    before = model(torch.ones(3, 3)).detach().clone()
    parameter = next(model.parameters())
    fingerprint = model.template_fingerprint
    shapes = model.specialization_shapes
    if change == "root":
        path.write_text(source.replace('return ops.boost(', 'return 3*ops.boost('))
    else:
        helper.write_text(helper.read_text().replace('2*x', '3*x'))
    with patch("weightlang.dynamic.build_module", side_effect=AssertionError("candidate built")):
        with pytest.raises(ValueError, match="reload required"):
            model(torch.ones(4, 3))
    assert model.template_fingerprint == fingerprint
    assert model.specialization_shapes == shapes
    assert next(model.parameters()) is parameter
    torch.testing.assert_close(model(torch.ones(3, 3)), before)


@pytest.mark.parametrize("fingerprint", [None, "", "bad", "x" * 64])
def test_missing_or_invalid_fingerprint_fails_closed(fingerprint):
    with pytest.raises(wgt.CompilationError, match="template_fingerprint"):
        dynamic.DynamicModule(lambda shapes: {"template_fingerprint": fingerprint},
                              input_shapes={"x": (2, 3)}, device="cpu")


def test_local_memory_is_not_a_persistent_write():
    definition = {"format": "weightlang.torch.v1", "kind": "forward",
                  "program": {"parameters": [], "instructions": [
                      {"op": op} for op in ("memory_alloc", "memory_write", "memory_free")]}}
    assert dynamic._schema(definition) == {}


def test_inline_compiler_path_pinned(monkeypatch):
    calls = []
    definition = wgt.compile_source(SOURCE, input_shapes={"x": (2, 3)})
    second = wgt.compile_source(SOURCE, input_shapes={"x": (3, 3)})
    resolved = dynamic._compiler(None)
    def compile_inline(source, **kwargs):
        calls.append(kwargs["compiler"])
        return definition if len(calls) == 1 else second
    with patch("weightlang.dynamic.compile_source", side_effect=compile_inline):
        model = load()
        monkeypatch.setenv("WGT_COMPILER", "/missing/replaced/compiler")
        model(torch.ones(3, 3))
    assert calls == [str(Path(resolved).resolve())] * 2


def test_fingerprint_gate_preserves_state_before_candidate_build():
    # Exercise the guard independently of compiler metadata rollout. The file
    # and import tests above require genuine compiler-generated fingerprints.
    definition = wgt.compile_source(SOURCE, input_shapes={"x": (2, 3)})
    definition["template_fingerprint"] = "a" * 64
    model = dynamic.DynamicModule(lambda shapes: definition,
                                  input_shapes={"x": (2, 3)}, device="cpu")
    parameter = next(model.parameters())
    original = parameter.detach().clone()
    changed = dict(definition, template_fingerprint="b" * 64)
    with patch.object(model, "_compile_shape", return_value=changed), patch(
            "weightlang.dynamic.build_module", side_effect=AssertionError("candidate built")):
        with pytest.raises(ValueError, match="reload required"):
            model(torch.ones(3, 3))
    assert model.specialization_shapes == ((2, 3),)
    assert next(model.parameters()) is parameter
    torch.testing.assert_close(parameter, original)


HIDDEN_SOURCE = """wgt 0.5
dim ROWS = 2
model DynamicExplicitHidden
# The activation row axis varies; projection parameters stay [2,2].
parameter weight: f32[2,2] = [0.2,-0.1,0.3,0.4]
# Hidden has one row so it can link calls with different activation row counts.
# Sum all token/features into a signal for both hidden channels. The
# explicit scale keeps the shared f32[1] reduction differentiable.
fn forward(x: f32[ROWS,2], hidden: f32[1,2]) -> (f32[ROWS,2], f32[1,2]) {
    let y = x @ (weight as [2,2]);
    let contribution = scale([1,1], sum(y)) as [1,2];
    let next_hidden = hidden + contribution;
    return (y, next_hidden);
}
"""


def test_dynamic_ports_linked_hidden_gradients_identity_and_optimizer():
    # Binding insertion order differs from declaration order deliberately.
    model = wgt.loads_dynamic(HIDDEN_SOURCE, input_shapes={"hidden": (1, 2), "x": (2, 2)},
                              device="cpu", max_specializations=2)
    parameter = next(model.parameters())
    reference_weight = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    reference_optimizer = torch.optim.Adam([reference_weight], lr=.02)
    handle = model.bind_optimizer(optimizer)
    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        x1 = torch.tensor([[.2, -.1], [.3, .4]], requires_grad=True)
        x2 = torch.tensor([[.1, -.2], [.5, .2], [-.3, .4]], requires_grad=True)
        hidden = torch.tensor([[.1, .2]], requires_grad=True)
        rx1, rx2, rh = [v.detach().clone().requires_grad_() for v in (x1, x2, hidden)]
        with patch("weightlang.initialization.recipe_value", side_effect=AssertionError("candidate initialized")):
            _, h1 = model(x1, hidden)
            y2, h2 = model(hidden=h1, x=x2)
        ry1 = rx1 @ reference_weight.reshape(2, 2)
        rh1 = rh + ry1.sum()
        ry2 = rx2 @ reference_weight.reshape(2, 2)
        rh2 = rh1 + ry2.sum()
        torch.testing.assert_close(y2, ry2)
        torch.testing.assert_close(h2, rh2)
        (h2.square().sum() + y2.square().sum()).backward()
        (rh2.square().sum() + ry2.square().sum()).backward()
        for value, reference in zip((x1, x2, hidden, parameter), (rx1, rx2, rh, reference_weight)):
            torch.testing.assert_close(value.grad, reference.grad)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(parameter, reference_weight)
        assert next(model.parameters()) is parameter
        assert len(list(model.parameters())) == len(model.state_dict()) == 1
        for candidate, _ in model._cache.values():
            assert next(candidate.parameters()) is parameter
    assert model.specialization_shapes == (((2, 2), (1, 2)), ((3, 2), (1, 2)))
    with pytest.raises(TypeError):
        model(x=torch.ones(2, 2))
    # A failed Rust dimension unification must preserve both cache and weights.
    shapes = model.specialization_shapes
    with pytest.raises(wgt.CompilationError):
        model(torch.ones(4, 2), torch.ones(2, 2))
    assert model.specialization_shapes == shapes
    handle.remove()
    model.eval().double()
    assert len(model.specialization_shapes) == 1
    y, hidden = model(torch.ones(3, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64))
    assert y.dtype == hidden.dtype == torch.float64


def test_input_names_are_validated_by_shared_rust_frontend():
    # Multiple bindings are legal; this particular source only declares x.
    # Do not turn Rust semantic diagnostics into blanket Python shape errors.
    with pytest.raises(wgt.CompilationError, match="input|name|unknown|forward"):
        wgt.loads_dynamic(SOURCE, input_shapes={"x": (2, 3), "y": (2, 3)}, device="cpu")
