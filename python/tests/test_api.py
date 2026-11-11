"""Acceptance through the real Rust compiler and the public Python API."""

import json
import os
from pathlib import Path

import pytest
import torch

import weightlang as wgt

ROOT = Path(__file__).resolve().parents[2]
SCALE = """wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
"""


@pytest.fixture
def compiler(monkeypatch):
    executable = os.environ.get("WGT_COMPILER", str(ROOT / "target/debug/wgt"))
    assert Path(executable).is_file(), "Build cargo build -p weightlang-wgt --bin wgt before SDK acceptance tests"
    monkeypatch.setenv("WGT_COMPILER", executable)
    return executable


def test_source_import_gradients_optimizer_and_batch(compiler):
    model = wgt.loads(SCALE, device="cpu")
    assert isinstance(model, torch.fx.GraphModule)
    x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    torch.testing.assert_close(model(x), 2 * x)
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
    parameter = next(model.parameters())
    torch.testing.assert_close(parameter.grad, torch.tensor([6.0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    torch.testing.assert_close(model(x), 1.4 * x)
    composed = torch.nn.Sequential(model, torch.nn.Sigmoid())
    torch.testing.assert_close(composed(x), torch.sigmoid(1.4 * x))


def test_compiled_json_loads_without_compiler(compiler, tmp_path, monkeypatch):
    definition = wgt.compile_source(SCALE)
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "absent-compiler"))
    model = wgt.load(path, device="cpu")
    torch.testing.assert_close(model(torch.tensor([3.0])), torch.tensor([6.0]))


def test_saved_model_restores_trained_weights_and_dtype_without_compiler(compiler, tmp_path, monkeypatch):
    model = wgt.loads(SCALE, device="cpu", dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    torch.nn.functional.mse_loss(model(x), 3 * x).backward()
    optimizer.step()
    path = wgt.save(model, tmp_path / "weights/model.pt")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "absent-compiler"))
    restored = wgt.load(path, device="cpu")
    assert next(restored.parameters()).dtype == torch.float64
    torch.testing.assert_close(restored(x), model(x))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        wgt.save(model, path)
    assert path.read_bytes() == before
    with torch.no_grad():
        next(model.parameters()).fill_(7)
    wgt.save(model, path, overwrite=True)
    torch.testing.assert_close(wgt.load(path, device="cpu")(x), 7 * x)


def test_mlp_trains_with_standard_torch_and_preserves_host_rng(compiler):
    rng = torch.random.get_rng_state().clone()
    model = wgt.load(ROOT / "python/examples/mlp/model.wgt", device="cpu")
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    x = torch.linspace(-1, 1, 32).unsqueeze(-1)
    target = 2 * x + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    initial = torch.nn.functional.mse_loss(model(x), target).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial / 50


def test_pytorch_export_and_compile_capture_model(compiler, tmp_path):
    model = wgt.loads(SCALE, device="cpu")
    x = torch.tensor([[2.0], [3.0]])
    exported = torch.export.export(model, (x,))
    torch.testing.assert_close(exported.module()(x), model(x))
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), model(x))
    path = wgt.save(compiled, tmp_path / "compiled.pt")
    torch.testing.assert_close(wgt.load(path, device="cpu")(x), model(x))


def test_compiler_uses_source_project_version_pin(compiler, tmp_path):
    source = tmp_path / "model.wgt"
    source.write_text(SCALE, encoding="utf-8")
    (tmp_path / ".wgt-version").write_text("99.0.0\n", encoding="utf-8")
    with pytest.raises(wgt.CompilationError, match="99.0.0"):
        wgt.compile(source)


def test_diagnostics_and_host_rejection(compiler, tmp_path, monkeypatch):
    with pytest.raises(wgt.CompilationError, match="definition|entrypoint"):
        wgt.loads(SCALE + "fn main() { return 0; }\n")
    with pytest.raises(wgt.CompilationError):
        wgt.loads(SCALE.replace("return x * gain", "return missing(x)"))
    with pytest.raises(FileNotFoundError):
        wgt.load(tmp_path / "model.wgt")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="WGT_COMPILER"):
        wgt.loads(SCALE)


def test_saved_bundle_type_and_version_checked(tmp_path):
    plain = tmp_path / "plain.pt"
    torch.save({"gain": torch.tensor([2.0])}, plain)
    with pytest.raises(ValueError, match="state_dict"):
        wgt.load(plain)
    with pytest.raises(ValueError):
        wgt.from_ir({"format": "weightlang.torch.v999"})


def test_auto_device_falls_back_and_honors_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert wgt.select_device() == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert wgt.select_device() == torch.device("cuda")
    assert wgt.select_device("cpu") == torch.device("cpu")


def test_transformer_forward_and_gradients_match_pytorch(compiler):
    model = wgt.load(ROOT / "crates/wgt-model/tests/fixtures/transformer_encoder/model.wgt", device="cpu")
    values = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(17))
    actual_input = values.clone().requires_grad_()
    reference_input = values.clone().requires_grad_()
    named = dict(model.named_parameters())
    weights = {
        name: named[attribute].detach().clone().requires_grad_()
        for name, attribute in model.parameter_names.items()
    }
    tokens = reference_input.reshape(2, 3, 2, 4)
    heads = [
        (tokens @ weights[name]).reshape(2, 3, 2, 2, 2).transpose(-3, -2)
        for name in ("query_weight", "key_weight", "value_weight")
    ]
    query, key, value = heads
    probabilities = torch.softmax((query @ key.transpose(-1, -2)) / (2 ** 0.5), dim=-1)
    merged = (probabilities @ value).transpose(-3, -2).reshape(2, 3, 2, 4)
    residual = tokens + merged @ weights["attention_output"]
    normalized = torch.nn.functional.layer_norm(
        residual, (4,), weights["norm1_gamma"], weights["norm1_beta"], 1e-5,
    )
    residual = normalized + torch.relu(normalized @ weights["feed_forward_in"]) @ weights["feed_forward_out"]
    expected = torch.nn.functional.layer_norm(
        residual, (4,), weights["norm2_gamma"], weights["norm2_beta"], 1e-5,
    ).reshape(2, 3, 8)
    actual = model(actual_input)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    coefficients = torch.linspace(0.1, 0.8, 8)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=3e-6, rtol=2e-4)
    for name, attribute in model.parameter_names.items():
        torch.testing.assert_close(named[attribute].grad, weights[name].grad, atol=3e-6, rtol=2e-4)


def test_named_dimensions_signature_loss_and_roundtrip(compiler, tmp_path):
    source_file = ROOT / "examples/native_training/signature_dimensions/model.wgt"
    model = wgt.load(source_file, dimensions={"N": 3}, device="cpu")
    x = torch.ones(3, requires_grad=True)
    torch.testing.assert_close(model(x), x)
    loss = model.loss(model(x), torch.zeros(3))
    loss.backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 1 / 3))
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    expected = torch.full_like(x, 1 - 0.1 * 4 / 3)
    torch.testing.assert_close(model(x), expected)
    path = tmp_path / "model.pt"
    wgt.save(model, path)
    torch.testing.assert_close(wgt.load(path, device="cpu")(x), expected)
    with pytest.raises(ValueError, match="only apply to WGT"):
        wgt.load(path, dimensions={"N": 2})
    text = source_file.read_text().replace("dim N = 2", "dim N")
    inline = wgt.loads(text, dimensions={"N": 4}, device="cpu")
    torch.testing.assert_close(inline(torch.ones(4)), torch.ones(4))
    for bindings in ({"TYPO": 2}, {"N": 0}, {"N": True}):
        with pytest.raises((ValueError, wgt.CompilationError)):
            wgt.loads(text, dimensions=bindings, device="cpu")


def test_dimensioned_file_import_keeps_library_dimensions_independent(compiler, tmp_path):
    source = (ROOT / "examples/native_training/signature_dimensions/model.wgt").read_text()
    (tmp_path / "ops.wgt").write_text('wgt 0.5\ndim N = 7\noperators Local\nfn widen_value(x){return N * x;}\n')
    (tmp_path / "model.wgt").write_text(source.replace("model Projection", 'import ops from "./ops.wgt"\nmodel Projection').replace("return linear(x, weight, bias);", "return ops.widen_value(linear(x, weight, bias));"))
    model = wgt.load(tmp_path / "model.wgt", dimensions={"N": 3}, device="cpu")
    torch.testing.assert_close(model(torch.ones(3)), torch.full((3,), 7.))


def test_matrix_function_contracts_compile_execute_and_differentiate(compiler):
    path = ROOT / "examples/native_training/matrix_signatures/model.wgt"
    model = wgt.load(path, device="cpu")
    x = torch.tensor([1., 2., 3., 4., 5., 6.], requires_grad=True)
    output = model(x)
    torch.testing.assert_close(output, torch.tensor([1.5, 1.5, 3.75, 3.75]))
    loss = model.loss(output, torch.zeros(4))
    torch.testing.assert_close(loss, torch.tensor(4.078125))
    loss.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.1875] * 3 + [0.46875] * 3))
    source = path.read_text().replace("project(x, weight as [K,N])", "project(x.T, weight as [K,N])")
    with pytest.raises(Exception, match="shape mismatch"):
        wgt.loads(source, device="cpu")
