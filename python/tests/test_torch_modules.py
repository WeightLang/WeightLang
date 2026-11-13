"""Dynamic PyTorch target against independent torch.nn behavior and durable I/O."""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from weightlang import compile_source
from weightlang.modules import classes_from_ir, load_module, save_module

SOURCE = """wgt 0.5
target torch
host weightlang.modules
extern nn, torch
model Recurrent extends Module {
    init(self, width=3) {
        super().__init__();
        self.cell = nn.GRU(width, width, batch_first=True);
        self.alias = self.cell;
    }
    fn forward(self, x, hidden=None) {
        return self.cell(x, hidden);
    }
}
"""


@pytest.fixture
def dynamic_class(monkeypatch):
    compiler = Path(__file__).resolve().parents[2] / "target/debug/wgt"
    monkeypatch.setenv("WGT_COMPILER", os.environ.get("WGT_COMPILER", str(compiler)))
    definition = compile_source(SOURCE)
    return classes_from_ir(definition, {"nn": nn, "torch": torch}, trusted=True)[
        "Recurrent"
    ]


def test_dynamic_length_streaming_and_gradients_match_pytorch(dynamic_class):
    torch.manual_seed(82)
    model = dynamic_class()
    reference = nn.GRU(3, 3, batch_first=True)
    reference.load_state_dict(model.cell.state_dict())
    for length in (1, 2, 7):
        x = torch.randn(2, length, 3, requires_grad=True)
        expected = reference(x)
        actual = model(x)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        state = model(x[:, :1])[1]
        if length > 1:
            torch.testing.assert_close(model(x[:, 1:], state)[1], actual[1])
        model.zero_grad()
        reference.zero_grad()
        actual[0].square().mean().backward()
        expected[0].square().mean().backward()
        for left, right in zip(model.cell.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_saved_binding_preserves_aliases_freezing_rng_and_source(
    dynamic_class, tmp_path, monkeypatch
):
    model = dynamic_class().eval()
    model.cell.bias_ih_l0.requires_grad_(False)
    x = torch.randn(2, 4, 3)
    expected = model(x)
    artifact = save_module(model, tmp_path / "model", kwargs={"width": 3})
    before = torch.get_rng_state().clone()
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "absent"))
    loaded = load_module(artifact.path, trusted=True).module
    assert torch.equal(torch.get_rng_state(), before)
    assert loaded.cell.weight_ih_l0 is loaded.alias.weight_ih_l0
    assert not loaded.cell.bias_ih_l0.requires_grad
    assert not loaded.training
    for left, right in zip(expected, loaded(x)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_module(model, artifact.path, kwargs={"width": 3})
    with pytest.raises(ValueError, match="trusted"):
        load_module(artifact.path)
    (artifact.path / "source.wgt").write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_module(artifact.path, trusted=True)


def test_dependency_bindings_keep_same_named_models_from_distinct_hosts(
    dynamic_class, monkeypatch, tmp_path
):
    import weightlang.compiler as host

    left = classes_from_ir(
        compile_source(
            SOURCE.replace("host weightlang.modules", "host weightlang.api")
        ),
        {"nn": nn, "torch": torch},
        trusted=True,
    )["Recurrent"]
    monkeypatch.setattr(host, "Left", left, raising=False)
    monkeypatch.setattr(host, "Right", dynamic_class, raising=False)
    source = """wgt 0.5
target torch
host weightlang.compiler
extern Left, Right
model Composite extends Module {
    init(self) {
        super().__init__();
        self.left = Left();
        self.right = Right();
    }
    fn forward(self, x) {
        return self.left(x)[0] + self.right(x)[0];
    }
}
"""
    cls = classes_from_ir(compile_source(source), vars(host), trusted=True)["Composite"]
    model = cls()
    x = torch.randn(1, 2, 3)
    expected = model(x)
    artifact = save_module(model, tmp_path / "composite", kwargs={})
    restored = load_module(artifact.path, trusted=True).module
    assert restored.left._wgt_definition["torch_module"]["host"] == "weightlang.api"
    assert (
        restored.right._wgt_definition["torch_module"]["host"] == "weightlang.modules"
    )
    torch.testing.assert_close(restored(x), expected, atol=0, rtol=0)


def test_nonpersistent_float_buffers_restore_converted_precision(
    dynamic_class, tmp_path
):
    source = SOURCE.replace(
        "self.alias = self.cell;",
        "self.alias = self.cell;\nself.register_buffer('gain', torch.ones(width), persistent=False);",
    )
    cls = classes_from_ir(
        compile_source(source), {"nn": nn, "torch": torch}, trusted=True
    )["Recurrent"]
    model = cls().double()
    artifact = save_module(model, tmp_path / "double", kwargs={})
    restored = load_module(artifact.path, trusted=True).module
    assert restored.gain.dtype == torch.float64
    assert restored.cell.weight_ih_l0.dtype == torch.float64
    torch.testing.assert_close(restored.gain, model.gain, atol=0, rtol=0)
