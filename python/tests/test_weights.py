"""Weight lifecycle acceptance: immutable inputs, explicit coverage, direct binding."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

import weightlang as wgt
from weightlang import initialization


ROOT = Path(__file__).resolve().parents[2]


def schema(**parameters):
    return {"format": "weightlang.torch.v1", "kind": "forward", "program": {
        "parameters": [{"name": name, "shape": list(shape), "data": [], "initializer": {"op": "unbound"}}
                       for name, shape in parameters.items()]}}


@pytest.fixture
def compiler(monkeypatch):
    executable = os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt"))
    assert Path(executable).is_file(), "Build wgt before compiler/SDK acceptance"
    monkeypatch.setenv("WGT_COMPILER", executable)


def test_mix_only_selected_layers_and_pin_roundtrip(tmp_path):
    definition = schema(first=(2, 2), second=(2,), untouched=(1,))
    a = wgt.WeightSet({"first": torch.eye(2), "second": torch.tensor([1., 2.]), "untouched": torch.tensor([7.])})
    b = wgt.WeightSet({"first": torch.full((2, 2), 3.), "second": torch.tensor([5., 6.]), "untouched": torch.tensor([99.])})
    plan = wgt.Plan(definition, base=a)
    for name in ("first", "second"):
        plan.replace(name, .75 * a[name] + .25 * b[name], sources=(a.ref(name), b.ref(name)))
    result = plan.build()
    torch.testing.assert_close(result["first"], torch.tensor([[1.5, .75], [.75, 1.5]]))
    torch.testing.assert_close(result["second"], torch.tensor([2., 3.]))
    torch.testing.assert_close(result["untouched"], a["untouched"])
    result["first"].zero_()
    a["first"].fill_(50)
    torch.testing.assert_close(a["first"], torch.eye(2))
    path = result.save(tmp_path / "mixed")
    restored = wgt.WeightSet.load(path, revision=result.revision)
    assert restored.revision == result.revision
    torch.testing.assert_close(restored["first"], result["first"])
    with pytest.raises(FileExistsError):
        result.save(path)
    with pytest.raises(ValueError, match="pin"):
        wgt.WeightSet.load(path, revision="0" * 64)
    payload = path / "weights.safetensors"
    payload.write_bytes(payload.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        wgt.WeightSet.load(path)


def test_snapshots_roundtrip_in_both_directions_with_rust(tmp_path):
    executable = Path(os.environ.get("WGT_WEIGHT_BRIDGE", ROOT / "target/debug/examples/weights_roundtrip"))
    assert executable.is_file(), "Build cargo build -p wgt-model --example weights_roundtrip"

    def bridge(*arguments):
        result = subprocess.run([str(executable), *map(str, arguments)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    rust_path = tmp_path / "rust-original"
    revision = bridge("--create", rust_path)
    original = wgt.WeightSet.load(rust_path, revision=revision)
    torch.testing.assert_close(original["matrix"], torch.tensor([[1., 2.], [3., 4.]]))
    torch.testing.assert_close(original["bias"], torch.tensor([5., 6.]))
    migrated = (wgt.Plan(schema(matrix=(2, 2), bias=(2,)), base=original)
                .replace("matrix", original["matrix"] + 10).build())
    python_path, returned_path = tmp_path / "python-derived", tmp_path / "rust-republished"
    migrated.save(python_path)
    assert bridge(python_path, returned_path) == migrated.revision
    returned = wgt.WeightSet.load(returned_path, revision=migrated.revision)
    torch.testing.assert_close(returned["matrix"], torch.tensor([[11., 12.], [13., 14.]]))
    torch.testing.assert_close(returned["bias"], original["bias"])
    with pytest.raises(ValueError, match="pin"):
        wgt.WeightSet.load(returned_path, revision=original.revision)


@pytest.mark.parametrize("dimension", [True, 1.0])
def test_manifest_dimensions_require_integers_accepted_by_both_sdks(tmp_path, dimension):
    snapshot = wgt.WeightSet({"x": torch.ones(1)})
    root = snapshot.save(tmp_path / "snapshot")
    manifest = snapshot.manifest
    manifest["tensors"][0]["shape"] = [dimension]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integer"):
        wgt.WeightSet.load(root)


def test_migration_renames_expands_and_initializes_only_explicit_new_slots():
    old = wgt.WeightSet({"old_matrix": torch.tensor([[1., 2.], [3., 4.]]), "old_bias": torch.tensor([5., 6.])})
    definition = schema(matrix=(3, 2), bias=(2,), gain=(1,))
    calls = []
    plan = wgt.Plan(definition)
    plan.copy("bias", old.ref("old_bias"))
    plan.set("matrix", torch.cat((old["old_matrix"], torch.zeros(1, 2)), dim=0), sources=[old.ref("old_matrix")])
    plan.set("gain", lambda shape: calls.append(shape) or torch.full(shape, 2.))
    assert plan.preview()["complete"]
    assert calls == []
    result = plan.build()
    assert calls == [(1,)]
    torch.testing.assert_close(result["matrix"], torch.tensor([[1., 2.], [3., 4.], [0., 0.]]))
    torch.testing.assert_close(result["bias"], old["old_bias"])
    with pytest.raises(ValueError, match="exactly cover"):
        wgt.Plan(definition, base=old)


def test_missing_conflicts_and_failed_batch_preserve_plan_and_sources():
    definition = schema(a=(1,), b=(1,))
    plan = wgt.Plan(definition)
    calls = []
    plan.set("a", lambda shape: calls.append(shape) or torch.ones(shape))
    with pytest.raises(ValueError, match="unassigned"):
        plan.build()
    assert calls == []
    with pytest.raises(ValueError, match="already assigned"):
        plan.set("a", torch.ones(1))
    with pytest.raises(KeyError):
        plan.set_many({"b": torch.ones(1), "typo": torch.ones(1)})
    assert plan.preview()["missing"] == ["b"]
    plan.replace("a", torch.tensor([3.]))
    with pytest.raises(ValueError, match="already replaced"):
        plan.replace("a", torch.tensor([4.]))
    with pytest.raises(ValueError, match="F32"):
        plan.set("b", torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="nonfinite"):
        plan.set("b", torch.tensor([float("nan")]))
    assert plan.preview()["missing"] == ["b"]


def test_snapshot_preserves_learn_shadows_without_rounding_or_projection():
    definition = schema(switch=(1,), count=(1,), probability=(1,))
    specs = definition["program"]["parameters"]
    specs[0]["learn"] = {"kind": "bool"}
    specs[1]["learn"] = {"kind": "int", "low": 0, "high": 8}
    specs[2].update(probability=True, frozen=True)
    plan = wgt.Plan(definition).set_many({"switch": torch.tensor([.3]), "count": torch.tensor([2.5]),
                                        "probability": torch.tensor([.75])})
    result = plan.build()
    assert result["count"].item() == 2.5
    torch.testing.assert_close(result["switch"], torch.tensor([.3]))
    with pytest.raises(ValueError, match="domain"):
        wgt.Plan(definition).set("switch", torch.tensor([2.]))
    with pytest.raises(ValueError, match="prob"):
        wgt.Plan(definition).set("probability", torch.tensor([2.]))


@pytest.mark.parametrize("body", ["return x * parameters;", "if positive(x) { return x * parameters; } return x;"])
def test_direct_restore_and_old_pt_restore_never_run_initializers(compiler, monkeypatch, tmp_path, body):
    definition = wgt.compile_source(f"""wgt 0.5
model Restored
input x: f32[1]
output y: f32[1]
parameter parameters: f32[1]
fn forward(x) {{ {body} }}
""")
    def forbidden(*args, **kwargs):
        raise AssertionError("restore executed an initializer")
    monkeypatch.setattr(initialization, "recipe_value", forbidden)
    unbound = deepcopy(definition)
    weights = wgt.WeightSet({"parameters": torch.tensor([7.])})
    model = wgt.restore(unbound, weights, device="cpu")
    torch.testing.assert_close(model(torch.tensor([3.])), torch.tensor([21.]))
    # Legacy SDK bundles had materialized literals, three fields, no name map,
    # and registered attribute names different from this reserved logical name.
    legacy = deepcopy(unbound)
    legacy["program"]["parameters"][0].pop("initializer")
    legacy["program"]["parameters"][0]["data"] = [2.]
    path = tmp_path / "old.pt"
    torch.save({"format": "weightlang.model.v1", "definition": legacy, "state_dict": model.state_dict()}, path)
    from weightlang import runtime
    monkeypatch.setattr(runtime, "_tensor", forbidden)
    restored = wgt.load(path, device="cpu")
    torch.testing.assert_close(restored(torch.tensor([3.])), torch.tensor([21.]))


def test_typed_weights_exclude_state_and_bind_frozen_before_instance_exists(compiler, tmp_path, monkeypatch):
    definition = wgt.compile_source("""wgt 0.5
model StateAndWeights
frozen gain: f32[1] = [2]
state count: f32[1] = [5]
fn forward(x: f32[1]) -> f32[1] { return x * gain + count; }
fn advance(x: f32[1]) { count <- count + x; }
""")
    model = wgt.restore(definition, wgt.WeightSet({"gain": torch.tensor([9.])}), device="cpu")
    model.advance(torch.tensor([3.]))
    snapshot = wgt.WeightSet.from_model(model)
    assert snapshot.names == ("gain",)
    fresh = wgt.restore(definition, snapshot, device="cpu")
    torch.testing.assert_close(fresh(torch.tensor([1.])), torch.tensor([14.]))
    with pytest.raises(ValueError, match="candidate|frozen|updatable"):
        fresh.update_parameters({"gain": torch.tensor([4.])})
    path = wgt.save(model, tmp_path / "with-state.pt")
    monkeypatch.setattr(initialization, "recipe_value", lambda *a, **k: (_ for _ in ()).throw(AssertionError("checkpoint initialization")))
    restored = wgt.load(path, device="cpu")
    torch.testing.assert_close(restored(torch.tensor([1.])), torch.tensor([17.]))


def test_declared_recipes_are_explicit_and_preview_does_not_execute():
    definition = schema(gain=(2,))
    spec = definition["program"]["parameters"][0]
    spec["initializer"] = {"op": "full", "shape": [2], "value": 3.}
    plan = wgt.Plan(definition)
    with pytest.raises(ValueError, match="unassigned"):
        plan.build()
    plan.initialize(["gain"])
    assert plan.preview()["assignments"][0]["deferred"]
    torch.testing.assert_close(plan.build()["gain"], torch.full((2,), 3.))
