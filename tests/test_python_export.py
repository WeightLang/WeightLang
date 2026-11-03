"""Exported code must run without WeightLang and preserve values and gradients."""

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import safetensors
import torch

from weightlang import export_python, parse_file, compile_file, load_artifact, WgtError
from weightlang.artifact import build_module
from weightlang.cli import main
from weightlang import syntax


def load(path):
    spec = importlib.util.spec_from_file_location("exported_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "example",
    [
        "cnn.wgt",
        "transformer.wgt",
        "std_mlp.wgt",
        "gru.wgt",
        "recurrent.wgt",
        "swiglu.wgt",
        "gated_transformer.wgt",
        "dimensions.wgt",
    ],
)
def test_examples_preserve_values_and_gradients(tmp_path, example):
    source = Path("examples") / example
    if not source.exists():
        pytest.skip("optional architecture example is not present in this checkout")
    program = parse_file(source)
    reference = build_module([layer.to_dict() for layer in program.layers]).eval()
    target = tmp_path / "model.py"
    result = export_python(source, target)
    assert result["mode"] == "architecture-only"
    assert not target.with_suffix(".safetensors").exists()
    exported = load(target).Model().eval()
    exported.load_state_dict(reference.state_dict(), strict=True)
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in exported.parameters())
    x = (
        torch.randint(0, 2, (2, *program.input.shape))
        if program.input.dtype == "i64"
        else torch.randn(2, *program.input.shape)
    )
    expected, actual = reference(x), exported(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    for a, b in zip(reference.parameters(), exported.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    tree = ast.parse(target.read_text())
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {"torch", "pathlib", "safetensors"}


def write_model(tmp_path, body, shape="4", output="4", extra=""):
    source = tmp_path / "source.wgt"
    source.write_text(
        f"wgt 0.3\nmodel demo\ninput x: f32[{shape}]\noutput y: f32[{output}]\n{extra}{body}\ninit seed=42\n"
    )
    return source


@pytest.mark.skipif(not hasattr(syntax, "GRAPH_OPS"), reason="requires graph language extension")
@pytest.mark.parametrize("op", ["add", "mul", "concat", "mix", "where"])
def test_merges(tmp_path, op):
    refs = "a,b,c" if op in {"mix", "where"} else "a,b"
    source = write_model(
        tmp_path,
        f"layer a: linear(in=4, out=4)\nlayer b: relu from input\nlayer c: sigmoid from input\nlayer result: {op}({refs})",
        output="8" if op == "concat" else "4",
    )
    program = parse_file(source)
    reference = build_module([layer.to_dict() for layer in program.layers])
    target = tmp_path / "model.py"
    export_python(source, target)
    model = load(target).Model()
    model.load_state_dict(reference.state_dict())
    x = torch.randn(3, 4)
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


def test_artifact_standalone_weights_freezing_and_integrity(tmp_path):
    source = write_model(tmp_path, "layer body: linear(in=4,out=4)\nlayer head: linear(in=4,out=4)", extra="")
    source.write_text(
        source.read_text()
        + "freeze body\ndata samples:\n  [1,2,3,4] -> [0,0,0,0]\ntrain data=samples loss=mse optimizer=adam lr=0.01 epochs=2\n"
    )
    artifact = compile_file(source, tmp_path / "weights")
    target = tmp_path / "export" / "model.py"
    result = export_python(artifact.path, target)
    assert result["mode"] == "artifact"
    exported = load(target).load_model()
    reference = load_artifact(artifact.path, warn_untrained=False).module
    x = torch.randn(2, 4)
    torch.testing.assert_close(exported(x), reference(x), rtol=0, atol=0)
    assert [n for n, p in exported.named_parameters() if not p.requires_grad] == ["body.weight", "body.bias"]
    assert not exported.training
    # -I removes PYTHONPATH; additionally reject every WeightLang import, even
    # if the developer installed it in site-packages.
    script = """import importlib.abc, importlib.util, sys
sys.path.insert(0, sys.argv[2])  # explicit third-party dependency, no project path
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("weightlang"):
            raise ImportError("WeightLang is forbidden")
sys.meta_path.insert(0, Deny())
spec = importlib.util.spec_from_file_location("model", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
model = m.load_model()
assert tuple(model(m.torch.ones(2, 4)).shape) == (2, 4)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(target), str(Path(safetensors.__file__).parent.parent)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    original = target.read_bytes()
    with pytest.raises(WgtError, match="existing"):
        export_python(artifact.path, target)
    assert target.read_bytes() == original
    with target.with_suffix(".safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        load(target).load_model()


def test_cli_and_existing_sidecar(tmp_path, capsys):
    source = write_model(tmp_path, "layer body: linear(in=4,out=4)")
    target = tmp_path / "model.py"
    assert main(["export-python", str(source), "-o", str(target)]) == 0
    assert "architecture-only" in capsys.readouterr().out
    assert main(["export-python", str(source), "-o", str(target)]) == 1
    artifact = compile_file(source, tmp_path / "weights")
    other = tmp_path / "other.py"
    other.with_suffix(".safetensors").write_bytes(b"keep")
    with pytest.raises(WgtError, match="existing weights"):
        export_python(artifact.path, other)
    assert not other.exists()
    assert other.with_suffix(".safetensors").read_bytes() == b"keep"


@pytest.mark.skipif(not hasattr(syntax, "LOOP_OPS"), reason="requires loop language extension")
def test_shared_loop_and_all_activations(tmp_path):
    source = write_model(
        tmp_path,
        "layer loop: recur(3, unit(dim=4))\nlayer head: softmax",
        extra="""block unit(dim):
  layer a: linear(in=dim,out=dim)
  layer b: tanh -> sigmoid -> gelu -> silu -> identity -> dropout(p=0.2) -> layernorm(dim=dim)
""",
    )
    p = parse_file(source)
    ref = build_module([layer.to_dict() for layer in p.layers]).eval()
    target = tmp_path / "model.py"
    export_python(source, target)
    out = load(target).Model().eval()
    out.load_state_dict(ref.state_dict())
    x = torch.randn(2, 4)
    torch.testing.assert_close(out(x), ref(x), rtol=0, atol=0)
    ref(x).sum().backward()
    out(x).sum().backward()
    for a, b in zip(ref.parameters(), out.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("body", "shape", "output"),
    [
        (
            "layer conv: conv2d(in=1,out=2,kernel=2)\nlayer pool: maxpool2d(kernel=2,stride=2)\nlayer flat: flatten",
            "1,5,5",
            "8",
        ),
        ("layer fc: linear(in=4,out=4)\nlayer mean: meanpool", "3,4", "4"),
        ('layer fc: linear(in=4,out=4)\nlayer sum: reduce("time feature -> feature",op=sum)', "3,4", "4"),
        ('layer fc: linear(in=4,out=4)\nlayer same: reduce("time feature -> time feature",op=mean)', "3,4", "3,4"),
    ],
)
def test_pooling_and_reduction(tmp_path, body, shape, output):
    if "reduce(" in body and "reduce" not in syntax.BUILTIN_OPS:
        pytest.skip("requires axis language extension")
    source = write_model(tmp_path, body, shape, output)
    p = parse_file(source)
    ref = build_module([layer.to_dict() for layer in p.layers])
    target = tmp_path / "model.py"
    export_python(source, target)
    out = load(target).Model()
    out.load_state_dict(ref.state_dict())
    x = torch.randn(2, *p.input.shape)
    torch.testing.assert_close(out(x), ref(x), rtol=0, atol=0)


def test_failed_copy_removes_only_created_files(tmp_path, monkeypatch):
    source = write_model(tmp_path, "layer fc: linear(in=4,out=4)")
    artifact = compile_file(source, tmp_path / "weights")
    target = tmp_path / "model.py"
    from weightlang import python_export

    def fail(stream, output, length):
        output.write(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(python_export.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="disk failure"):
        export_python(artifact.path, target)
    assert not target.exists()
    assert not target.with_suffix(".safetensors").exists()
    assert load_artifact(artifact.path, warn_untrained=False).module is not None
