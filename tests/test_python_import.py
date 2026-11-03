"""Behavioral comparisons use only the small trusted model fixtures below."""

import json

import pytest
import torch

from weightlang.artifact import build_module
from weightlang.cli import main
from weightlang.decompile import decompile
from weightlang.python_import import python_architecture
from weightlang.syntax import WgtError, parse


MLP = """
import torch
from torch import nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, width=4, hidden=8):
        super().__init__()
        self.fc = nn.Linear(width, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, width)
    def forward(self, x):
        skip = x
        x = F.gelu(self.norm(self.fc(x)))
        return self.out(x) + skip
"""
CNN = """
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, (3,3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4*4*4, 2)
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.flatten(1))
"""
GATED = """
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)
        self.out = nn.Linear(8, 2)
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        value = self.value(x)
        return self.out(torch.cat([gate * value, x], dim=-1))
"""
NESTED = """
from torch import nn
class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.activation = nn.Tanh()
    def forward(self, x):
        return self.activation(self.fc(x)) + x
class Net(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.layers = nn.ModuleList([Block(width), Block(width)])
        self.out = nn.Sequential(nn.Linear(width, 2), nn.Sigmoid())
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
"""
EMBED = """
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(self.embed(x))
"""


@pytest.mark.parametrize(
    "source,shape,dtype",
    [
        (MLP, (4,), "f32"),
        (
            MLP.replace("nn.Linear(width, hidden)", "nn.Linear(width, hidden, bias=False)")
            .replace("nn.LayerNorm(hidden)", "nn.LayerNorm(hidden, eps=0.001, bias=False)")
            .replace("F.gelu(self.norm(self.fc(x)))", 'F.gelu(self.norm(self.fc(x)), approximate="tanh")'),
            (4,),
            "f32",
        ),
        (MLP, (3, 4), "f32"),
        (CNN, (3, 8, 8), "f32"),
        (GATED, (4,), "f32"),
        (NESTED, (4,), "f32"),
        (NESTED.replace("[Block(width), Block(width)]", "[Block(width) for _ in range(2)]"), (4,), "f32"),
        (
            NESTED.replace("for layer in self.layers:", "for i in range(2):").replace(
                "x = layer(x)", "x = self.layers[i](x)"
            ),
            (4,),
            "f32",
        ),
        (EMBED, (5,), "i64"),
        (EMBED.replace("nn.Embedding(10, 4)", "nn.Embedding(10, 4, padding_idx=0)"), (5,), "i64"),
        (MLP.replace("nn.LayerNorm(hidden)", "nn.LayerNorm(hidden, elementwise_affine=False)"), (4,), "f32"),
    ],
)
def test_forward_and_gradients_match(source, shape, dtype, tmp_path):
    path = tmp_path / "model.py"
    path.write_text(source)
    program = parse(python_architecture(path, class_name="Net", input_shape=shape, input_dtype=dtype))
    assert not program.imports and program.train is None
    converted = build_module([layer.to_dict() for layer in program.layers]).eval()
    namespace = {}
    exec(source, namespace)  # Only this test's trusted constants, never user files.
    original = namespace["Net"]().eval()
    modules = dict(original.named_modules())
    pairs = []
    for name, target in converted.named_children():
        if not list(target.parameters()):
            continue
        origin_path = name.split("_", 1)[1].removeprefix("Net_").replace("_", ".")
        origin = modules[origin_path]
        target.load_state_dict(origin.state_dict())
        pairs.extend(zip(origin.parameters(), target.parameters()))
    for batch in (1, 3):
        original.zero_grad()
        converted.zero_grad()
        x = torch.randint(0, 10, (batch, *shape)) if dtype == "i64" else torch.randn(batch, *shape, requires_grad=True)
        z = x.detach().clone().requires_grad_(dtype == "f32")
        expected, actual = original(x), converted(z)
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        if dtype == "f32":
            torch.testing.assert_close(z.grad, x.grad)
        for a, b in pairs:
            torch.testing.assert_close(a.grad, b.grad)


def test_constructor_options_and_cli(tmp_path, capsys):
    path, out = tmp_path / "model.py", tmp_path / "out.wgt"
    path.write_text(MLP)
    assert (
        main(
            [
                "decompile",
                str(path),
                "-o",
                str(out),
                "--class-name",
                "Net",
                "--init-args",
                json.dumps({"width": 6, "hidden": 12}),
                "--input-shape",
                "2,6",
            ]
        )
        == 0
    )
    program = parse(out.read_text())
    assert program.input.shape == (2, 6) and program.output.shape == (2, 6)
    assert main(["check", str(out), "--shapes"]) == 0
    assert "sha256=" not in out.read_text()
    assert main(["decompile", str(path), "-o", str(tmp_path / "bad.wgt"), "--init-args", "[]"]) == 1
    assert "--init-args must be a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize(
    "body,match",
    [
        ("return self.fc(self.fc(x))", "shared module"),
        ("if x.sum() > 0:\n            return self.fc(x)\n        return x", "control flow"),
        ("return self.fc(x) + 1", "unsupported forward expression"),
        ("return x.reshape(-1, 4)", "expected a literal"),
    ],
)
def test_unsupported_forward_is_explicit(tmp_path, body, match):
    path = tmp_path / "model.py"
    path.write_text(
        "from torch import nn\nclass Net(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4,4)\n"
        "    def forward(self,x):\n        " + body + "\n"
    )
    with pytest.raises(WgtError, match=match) as exc:
        decompile(str(path), tmp_path / "out.wgt")
    assert str(path) in str(exc.value)
    assert not (tmp_path / "out.wgt").exists()


@pytest.mark.parametrize(
    "constructor",
    [
        "nn.ReLU(inplace=True)",
        "nn.Softmax(dim=0)",
        "nn.Conv2d(4,4,3,groups=4)",
        "nn.BatchNorm1d(4)",
    ],
)
def test_semantic_options_are_not_silently_dropped(tmp_path, constructor):
    path = tmp_path / "model.py"
    path.write_text("from torch import nn\nmodel = nn.Sequential(" + constructor + ")")
    with pytest.raises(WgtError):
        decompile(str(path), tmp_path / "out.wgt", input_shape=(4,))
    assert not (tmp_path / "out.wgt").exists()


def test_source_is_never_executed(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "model.py"
    path.write_text(MLP + f'\nopen({str(marker)!r}, "w").write("bad")\n')
    with pytest.raises(WgtError, match="top-level"):
        decompile(str(path), tmp_path / "out.wgt")
    assert not marker.exists()


def test_ambiguity_shape_and_options(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(NESTED)
    with pytest.raises(WgtError, match="--class-name"):
        python_architecture(path)
    path.write_text(CNN)
    with pytest.raises(WgtError, match="--input-shape"):
        python_architecture(path)
    with pytest.raises(WgtError, match="unknown model class"):
        python_architecture(path, class_name="Missing")
    with pytest.raises(WgtError, match="require a .py"):
        decompile("whatever.safetensors", tmp_path / "out.wgt", class_name="Net")


@pytest.mark.parametrize(
    "source,match",
    [
        (NESTED.replace("[Block(width), Block(width)]", "[Block(width) for _ in range(1025)]"), "1024"),
        (MLP.replace("width=4, hidden=8", "width=4, nn=8").replace("hidden", "nn"), "shadowing"),
        (MLP.replace("skip = x", "torch = x"), "shadowing"),
        ("from torch import nn\nWIDTH=4\nWIDTH=8\nmodel=nn.Sequential(nn.Linear(WIDTH,2))", "reassigned"),
    ],
)
def test_unsafe_static_assumptions_fail(tmp_path, source, match):
    path = tmp_path / "model.py"
    path.write_text(source)
    with pytest.raises(WgtError, match=match):
        python_architecture(path, class_name="Net" if "class Net" in source else None)


@pytest.mark.parametrize("source,shape", [(MLP, (4,)), (CNN, (3, 8, 8)), (NESTED, (4,))])
def test_build_save_reload_matches_python(source, shape, tmp_path):
    from weightlang.artifact import load_artifact
    from weightlang.compiler import compile_file

    path, out = tmp_path / "model.py", tmp_path / "model.wgt"
    path.write_text(source)
    decompile(str(path), out, class_name="Net", input_shape=shape)
    built = compile_file(out, tmp_path / "weights", no_train=True)
    restored = load_artifact(built.path, warn_untrained=False).module.eval()
    namespace = {}
    exec(source, namespace)  # Trusted test fixture, not the converter.
    original = namespace["Net"]().eval()
    modules = dict(original.named_modules())
    for name, layer in restored.named_children():
        if list(layer.parameters()):
            origin_path = name.split("_", 1)[1].removeprefix("Net_").replace("_", ".")
            modules[origin_path].load_state_dict(layer.state_dict())
    x = torch.randn(3, *shape)
    torch.testing.assert_close(restored(x), original(x))
    assert built.manifest["training"] is None
