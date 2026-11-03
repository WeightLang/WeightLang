import pytest
import torch

from weightlang import WgtError, load_artifact, parse_file
from weightlang.artifact import sha256_file
from weightlang.compiler import compile_file, compile_program
from weightlang.cli import main

BASE = """wgt 0.3
model base
input x: f32[2]
output y: f32[10]
layer trunk: linear(in=2, out=4)
layer activation: relu
layer head: linear(in=4, out=10)
init seed=42
"""


@pytest.fixture
def baseline(tmp_path):
    source = tmp_path / "base.wgt"
    source.write_text(BASE)
    return compile_file(source, tmp_path / "weights")


def source(tmp_path, baseline, body, name="child"):
    path = tmp_path / f"{name}.wgt"
    path.write_text(
        f'wgt 0.3\nmodel {name} extends "weights/base" sha256="{sha256_file(baseline.path / "manifest.json")}":\n'
        + body
    )
    return path


EXPAND = """  output y: f32[12]
  layer head: linear(in=4, out=12)
  weights head:
    default: init(seed=123)
    weight[0:10, :] = base.head.weight
    bias[0:10] = base.head.bias
"""


def test_expand_head_and_reload(tmp_path, baseline):
    path = source(tmp_path, baseline, EXPAND)
    program = parse_file(path)
    assert program.output.shape == (12,)
    result = compile_program(program, tmp_path / "weights")
    assert torch.equal(result.module.trunk.weight, baseline.module.trunk.weight)
    assert torch.equal(result.module.head.weight[:10], baseline.module.head.weight)
    assert torch.equal(result.module.head.bias[:10], baseline.module.head.bias)
    loaded = load_artifact(result.path, warn_untrained=False)
    x = torch.randn(3, 2)
    assert torch.equal(loaded.predict(x)[:, :10], baseline.module(x))
    assert result.manifest["weights"]["tensors"]["head.weight"]["origin"] == "edited"
    assert result.manifest["weights"]["tensors"]["trunk.weight"]["upstream_origin"] == "random-init"
    report = result.manifest["derivation"]
    assert report["stage"] == "before_training"
    assert report["structure"]["replaced"] == ["head"]
    assert report["tensors"]["trunk.weight"]["changed_elements"] == 0
    assert (result.path / "source.wgt").read_text() == path.read_text()
    again = compile_file(path, tmp_path / "again")
    assert result.manifest["weights"]["sha256"] == again.manifest["weights"]["sha256"]


def test_declarations_are_order_independent(tmp_path, baseline):
    a = EXPAND + "  weights trunk:\n    weight[0, 0] = base.trunk.weight[0, 0] + 0.5\n"
    b = """  weights trunk:
    weight[0, 0] = base.trunk.weight[0, 0] + 0.5
  weights head:
    bias[0:10] = base.head.bias
    weight[0:10, :] = base.head.weight
    default: init(seed=123)
  layer head: linear(in=4, out=12)
  output y: f32[12]
"""
    x = compile_file(source(tmp_path, baseline, a, "first"), tmp_path / "weights")
    y = compile_file(source(tmp_path, baseline, b, "second"), tmp_path / "weights")
    for name, tensor in x.module.state_dict().items():
        assert torch.equal(tensor, y.module.state_dict()[name])


def test_single_edit_and_immutable_base(tmp_path, baseline):
    before = (baseline.path / "weights.safetensors").read_bytes()
    body = "  weights head:\n    weight[3, 1] = 0.5\n"
    result = compile_file(source(tmp_path, baseline, body), tmp_path / "weights")
    for name, tensor in result.module.state_dict().items():
        expected = baseline.module.state_dict()[name].clone()
        if name == "head.weight":
            expected[3, 1] = 0.5
        assert torch.equal(tensor, expected)
    assert (baseline.path / "weights.safetensors").read_bytes() == before
    assert result.manifest["training"] is None


@pytest.mark.parametrize(
    "body, message",
    [
        ("  layer head: linear(in=4, out=10)\n", "no complete source"),
        (EXPAND.replace("default: init(seed=123)", "default: base.head"), "shape/type mismatch"),
        (EXPAND.replace("weight[0:10, :]", "weight[0:9, :]"), "shape mismatch"),
        ("  weights head:\n    weight[10, 0] = 0\n", "out of bounds"),
        ("  weights head:\n    weight[-1, 0] = 0\n", "non-negative"),
        ("  weights head:\n    weight[:, :] = 0\n    weight[0, 0] = 1\n", "overlapping"),
        ("  weights head:\n    weight[0, 0] = base.head.weight[0, 0] / 0\n", "invalid scalar"),
        ("  weights head:\n    weight[0, 0] = 1e999\n", "finite float32"),
        ("  weights missing:\n    weight = 0\n", "no parameter layer"),
        ("  remove layer missing\n", "unknown baseline"),
        ("  layer surprise: relu\n", "new layers need"),
        ("  remove layer head\n  layer head: relu\n", "conflicting"),
        ("  layer head: linear(in=4, out=12)\n", "shape/dtype mismatch"),
        ("  weights head:\n", "empty weights"),
    ],
)
def test_invalid_derivations(tmp_path, baseline, body, message):
    path = source(tmp_path, baseline, body)
    with pytest.raises(WgtError, match=message) as exc:
        parse_file(path)
    assert exc.value.path == str(path)
    assert exc.value.line is not None
    assert not (tmp_path / "weights/child").exists()


def test_insert_remove_and_explicit_migration(tmp_path, baseline):
    body = """  remove layer activation
  layer smoothing after trunk: tanh
  layer head: linear(in=4, out=10)
  weights head:
    default: base.head
"""
    result = compile_file(source(tmp_path, baseline, body), tmp_path / "weights")
    assert list(result.module._modules) == ["trunk", "smoothing", "head"]
    assert torch.equal(result.module.head.weight, baseline.module.head.weight)
    assert result.manifest["derivation"]["structure"]["removed"] == ["activation"]


def test_complete_assignments_without_initialization(tmp_path, baseline):
    body = """  layer head: linear(in=4, out=10)
  weights head:
    weight[0:5, :] = 0
    weight[5:10, :] = base.head.weight[5:10, :]
    bias = base.head.bias
"""
    result = compile_file(source(tmp_path, baseline, body), tmp_path / "weights")
    assert torch.count_nonzero(result.module.head.weight[:5]) == 0


def test_pin_checked_again_at_build_and_force_preserves_output(tmp_path, baseline):
    path = source(tmp_path, baseline, "  weights head:\n    bias[0] = 1\n")
    program = parse_file(path)
    result = compile_program(program, tmp_path / "weights")
    original = (result.path / "weights.safetensors").read_bytes()
    manifest = baseline.path / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(WgtError, match="sha256 mismatch"):
        compile_program(program, tmp_path / "weights", force=True)
    assert (result.path / "weights.safetensors").read_bytes() == original


def test_cli_check_and_build(tmp_path, baseline, capsys):
    path = source(tmp_path, baseline, EXPAND)
    assert main(["check", str(path)]) == 0
    assert main(["legacy-build", str(path), "--no-store", "-o", str(tmp_path / "weights")]) == 0
    assert "built" in capsys.readouterr().out


def test_derived_version_can_be_a_baseline(tmp_path, baseline):
    first = compile_file(source(tmp_path, baseline, "  weights head:\n    bias[0] = 1\n"), tmp_path / "weights")
    path = tmp_path / "grandchild.wgt"
    path.write_text(
        f'wgt 0.3\nmodel grandchild extends "weights/child" sha256="{sha256_file(first.path / "manifest.json")}":\n'
        "  weights head:\n    bias[1] = base.head.bias[0] + 1\n"
    )
    result = compile_file(path, tmp_path / "weights")
    assert result.module.head.bias[0] == 1
    assert result.module.head.bias[1] == 2


def test_nested_layer_replacement(tmp_path):
    path = tmp_path / "base.wgt"
    path.write_text("""wgt 0.3
model base
input x: f32[2]
output y: f32[2]
block body(dim):
  layer first: linear(in=dim, out=dim)
  layer act: relu
layer encoder: body(dim=2)
init seed=12
""")
    baseline = compile_file(path, tmp_path / "weights")
    child = source(
        tmp_path,
        baseline,
        """  layer encoder.first: linear(in=2, out=2)
  weights encoder.first:
    default: base.encoder.first
    bias[0] = 0.5
""",
    )
    result = compile_file(child, tmp_path / "weights")
    assert torch.equal(result.module.encoder.first.weight, baseline.module.encoder.first.weight)
    assert result.module.encoder.first.bias[0] == 0.5


def test_rng_isolation_and_declared_initialization(tmp_path, baseline):
    path = source(tmp_path, baseline, EXPAND)
    before = torch.random.get_rng_state()
    result = compile_file(path, tmp_path / "weights")
    assert torch.equal(before, torch.random.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected = torch.nn.Linear(4, 12)
    assert torch.equal(result.module.head.weight[10:], expected.weight[10:])
    assert torch.equal(result.module.head.bias[10:], expected.bias[10:])


def test_training_and_freezing_after_derivation(tmp_path, baseline):
    body = """  weights head:
    bias[0] = 0.5
  freeze all
  unfreeze head.bias
  data samples:
    [0, 0] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  train data=samples loss=mse optimizer=sgd lr=0.1 epochs=2
"""
    result = compile_file(source(tmp_path, baseline, body), tmp_path / "weights")
    assert torch.equal(result.module.trunk.weight, baseline.module.trunk.weight)
    assert torch.equal(result.module.head.weight, baseline.module.head.weight)
    assert result.module.head.bias[0] != 0.5
    assert result.manifest["weights"]["tensors"]["head.bias"]["origin"] == "trained"


def test_nonfinite_result_and_version_rejected(tmp_path, baseline):
    path = source(tmp_path, baseline, "  weights head:\n    weight = base.head.weight * 1e308\n")
    with pytest.raises(WgtError, match="non-finite"):
        compile_file(path, tmp_path / "weights")
    path.write_text(path.read_text().replace("wgt 0.3", "wgt 0.2"))
    with pytest.raises(WgtError, match="requires 'wgt 0.3'"):
        parse_file(path)


def test_force_cannot_replace_baseline(tmp_path, baseline):
    path = source(tmp_path, baseline, "  weights head:\n    bias[0] = 1\n", "same")
    path.write_text(path.read_text().replace("model same extends", "model base extends"))
    before = (baseline.path / "weights.safetensors").read_bytes()
    with pytest.raises(WgtError, match="immutable baseline"):
        compile_file(path, tmp_path / "weights", force=True)
    assert (baseline.path / "weights.safetensors").read_bytes() == before


def test_initialized_new_layer_and_dropped_parameters(tmp_path, baseline):
    body = """  remove layer head
  output y: f32[3]
  layer classifier after activation: linear(in=4, out=3)
  weights classifier:
    default: init(seed=10)
"""
    result = compile_file(source(tmp_path, baseline, body), tmp_path / "weights")
    assert result.manifest["derivation"]["dropped_tensors"] == ["head.bias", "head.weight"]
    assert result.manifest["weights"]["tensors"]["classifier.weight"]["origin"] == "random-init"


def test_same_model_order_produces_identical_weight_bytes(tmp_path, baseline):
    first = "  weights head:\n    bias[0] = 0.5\n    bias[1] = base.head.bias[0] + 1\n"
    second = "  weights head:\n    bias[1] = base.head.bias[0] + 1\n    bias[0] = 0.5\n"
    path = source(tmp_path, baseline, first)
    a = compile_file(path, tmp_path / "a")
    source(tmp_path, baseline, second)
    b = compile_file(path, tmp_path / "b")
    assert (a.path / "weights.safetensors").read_bytes() == (b.path / "weights.safetensors").read_bytes()
    assert b.module.head.bias[1] == baseline.module.head.bias[0] + 1


def test_demo_builds(tmp_path):
    import runpy

    from pathlib import Path

    demo = Path(__file__).resolve().parents[1] / "examples/derivation/demo.py"
    result = runpy.run_path(str(demo))["main"](tmp_path / "demo")
    assert result.manifest["interface"]["output"]["shape"] == [12]


def test_version_store_accepts_derived_model(tmp_path, baseline, capsys):
    from weightlang.repository import WeightRepository

    path = source(tmp_path, baseline, EXPAND)
    store = tmp_path / "store"
    assert main(["legacy-build", str(path), "--store", str(store)]) == 0
    assert main(["legacy-build", str(path), "--store", str(store)]) == 0
    repo = WeightRepository(store)
    restored = tmp_path / "restored"
    repo.store.checkout(repo.resolve("HEAD"), restored)
    loaded = load_artifact(restored, warn_untrained=False)
    assert loaded.manifest["derivation"]["baseline"]["manifest_sha256"] == sha256_file(baseline.path / "manifest.json")


def test_preview_does_not_train_or_publish(tmp_path, baseline, capsys, monkeypatch):
    import json

    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not train")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    body = """  weights head:
    bias[0] = 0.5
  data samples:
    [0, 0] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  train data=samples loss=mse optimizer=sgd lr=0.1 epochs=2
"""
    path = source(tmp_path, baseline, body)
    store = tmp_path / "uncreated-store"
    assert main(["legacy-build", str(path), "--dry-run", "--store", str(store)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "weightlang.derivation"
    assert report["execution"] == "planned"
    assert report["training_requested"]
    assert report["tensors"]["head.bias"]["changed_elements"] == 1
    assert not store.exists()
    assert not (tmp_path / "weights/child").exists()
