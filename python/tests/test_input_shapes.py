"""Logical shape binding uses the shared Rust frontend, then concrete execution."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
from weightlang import torch_api as wgt

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "examples/native_training/input_shapes/model.wgt"
SOURCE = MODEL.read_text()


@pytest.mark.parametrize("rows,features,outputs", [(1, 2, 1), (2, 3, 2), (7, 4, 3)])
def test_shape_bound_source_native_forward_and_gradients(rows, features, outputs):
    shapes = {"x": (rows, features)}
    dimensions = {"OUTPUTS": outputs}
    model = wgt.load(MODEL, input_shapes=shapes, dimensions=dimensions, device="cpu")
    x = torch.linspace(-0.4, 0.8, rows * features, requires_grad=True)
    y = model(x)
    y.sum().backward()
    oracle = os.environ.get("WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference"))
    result = subprocess.run([oracle], input=json.dumps({
        "source": SOURCE, "input_shapes": shapes, "dimensions": dimensions,
        "input": x.detach().tolist(), "mode": "forward",
    }), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    expected = json.loads(result.stdout)
    torch.testing.assert_close(y, torch.tensor(expected["output"]), atol=2e-5, rtol=5e-5)
    torch.testing.assert_close(x.grad, torch.tensor(expected["input_gradient"]), atol=2e-5, rtol=5e-5)
    for name, key in model.parameter_names.items():
        parameter = model.get_parameter(key)
        torch.testing.assert_close(parameter.grad.flatten(), torch.tensor(expected["gradients"][name]), atol=2e-5, rtol=5e-5)
    # The SDK's inline entrypoint uses precisely the same shape binding.
    inline = wgt.loads(SOURCE, input_shapes=shapes, dimensions=dimensions, device="cpu")
    torch.testing.assert_close(inline(x.detach()), y.detach())
    with pytest.raises((RuntimeError, ValueError)):
        model(torch.ones(rows * features + 1))


def test_binding_conflicts_and_saved_shape_override_rejected(tmp_path):
    with pytest.raises(wgt.CompilationError, match="conflict"):
        wgt.loads(SOURCE, input_shapes={"x": (7, 3)}, dimensions={"ROWS": 2})
    with pytest.raises(wgt.CompilationError, match="rank mismatch"):
        wgt.loads(SOURCE, input_shapes={"x": (21,)})
    with pytest.raises(wgt.CompilationError, match="forward input"):
        wgt.loads(SOURCE, input_shapes={"typo": (7, 3)})
    for shape in [(0, 3), (True, 3), (1.5, 3), (), (1,) * 9]:
        with pytest.raises(ValueError, match="input_shapes"):
            wgt.loads(SOURCE, input_shapes={"x": shape})
    model = wgt.load(MODEL, input_shapes={"x": (7, 3)}, device="cpu")
    path = tmp_path / "model.pt"
    wgt.save(model, path)
    restored = wgt.load(path, device="cpu")
    torch.testing.assert_close(restored(torch.ones(21)), model(torch.ones(21)))
    with pytest.raises(ValueError, match="input_shapes only apply"):
        wgt.load(path, input_shapes={"x": (2, 3)})


def test_file_root_snapshot_keeps_import_scope(tmp_path):
    (tmp_path / "ops.wgt").write_text('wgt 0.5\ndim ROWS = 5\noperators Local\nfn boost(x) { return ROWS*x; }\n')
    path = tmp_path / "model.wgt"
    path.write_text(SOURCE.replace("model ShapeBoundProjection", 'import ops from "./ops.wgt"\nmodel ShapeBoundProjection').replace('return x @ (weight as [FEATURES,OUTPUTS]);', 'return ops.boost(x @ (weight as [FEATURES,OUTPUTS]));'))
    model = wgt.load(path, input_shapes={"x": (7, 3)}, device="cpu")
    torch.testing.assert_close(model(torch.ones(21)), torch.full((14,), 3.75))


def test_path_compiler_is_not_shadowed_by_an_old_checkout_binary(tmp_path, monkeypatch):
    import shutil
    compiler = os.environ.get("WGT_COMPILER") or shutil.which("wgt")
    assert compiler
    (tmp_path / "wgt").symlink_to(Path(compiler).resolve())
    monkeypatch.delenv("WGT_COMPILER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    model = wgt.loads(SOURCE, input_shapes={"x": (7, 3)}, device="cpu")
    torch.testing.assert_close(model(torch.ones(21)), torch.full((14,), .75))
