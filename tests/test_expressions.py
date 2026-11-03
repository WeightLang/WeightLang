import pytest
import torch

from weightlang import WgtError, parse
from weightlang.artifact import build_module


SOURCE = """wgt 0.3
const WIDTH = 2 * (1 + 1)
model expanded
input x: f32[WIDTH]
output y: f32[WIDTH]
block ff(dim, hidden=dim * 4):
  layer up: linear(in=dim, out=hidden)
  layer act: gelu
  layer down: linear(in=hidden, out=dim)
layer body: residual(ff(
  dim=WIDTH, # a multiline call
))
init seed=42
"""


def test_defaults_expressions_and_runtime():
    program = parse(SOURCE)
    assert program.layers[0].children[0].args == {"in": 4, "out": 16}
    module = build_module([layer.to_dict() for layer in program.layers])
    x = torch.randn(3, 4, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert all(p.grad is not None for p in module.parameters())
    override = parse(SOURCE.replace("dim=WIDTH,", "dim=WIDTH, hidden=8,"))
    assert override.layers[0].children[0].args["out"] == 8


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("10-2-3", 5),
        ("10 - 2 * 3", 4),
        ("20//3%4", 2),
        ("-(-4)", 4),
        ("+4", 4),
        ("8 / 2", 4.0),
        ("1e-2 * 100", 1.0),
    ],
)
def test_precedence(expression, expected):
    from weightlang.syntax import parse_library

    scope = parse_library(f"wgt 0.3\nconst RESULT = {expression}\n")
    assert scope.constants["RESULT"] == expected
    assert type(scope.constants["RESULT"]) is type(expected)


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("2 * (1 + 1)", "1 / 0", "division by zero"),
        ("2 * (1 + 1)", "1e309", "finite"),
        ("2 * (1 + 1)", "UNKNOWN", "unknown parameter"),
        ("const WIDTH = 2 * (1 + 1)", "const WIDTH = 4\nconst WIDTH = 8", "duplicate constant"),
        ("hidden=dim * 4", "hidden=later, later=4", "unknown parameter"),
        ("hidden=dim * 4", "hidden=4, required", "required parameters"),
        ("dim=WIDTH,", "dim=WIDTH, nope=8,", "takes"),
        ("dim=WIDTH,", "dim=WIDTH, hidden=2.5,", "positive integer"),
        ("f32[WIDTH]", "f32[WIDTH / 2]", "integer"),
        ("dim=WIDTH,", "dim=WIDTH],", "unmatched"),
        ("))\ninit", ")\ninit", "unclosed"),
    ],
)
def test_invalid_syntax(old, new, message):
    with pytest.raises(WgtError, match=message) as exc:
        parse(SOURCE.replace(old, new), "example.wgt")
    assert exc.value.path == "example.wgt"
    assert exc.value.line is not None


@pytest.mark.parametrize(
    "statement, message",
    [
        ("const X = 4", "const requires"),
        ("block b(dim=4):\n  layer l: linear(in=dim, out=dim)", "defaults require"),
        ("block b(dim):\n  layer l: linear(in=dim, out=dim*2)", "expressions require"),
        ("block b(\n dim\n):\n  layer l: relu", "multiline statements require"),
    ],
)
def test_version_gates(statement, message):
    from weightlang.syntax import parse_library

    with pytest.raises(WgtError, match=message):
        parse_library("wgt 0.2\n" + statement)


def test_library_lexical_scope(tmp_path):
    (tmp_path / "lib.wgt").write_text("""wgt 0.3
const WIDTH = 7
block ff(dim, hidden=WIDTH):
  layer up: linear(in=dim, out=hidden)
  layer down: linear(in=hidden, out=dim)
""")
    source = (
        SOURCE[: SOURCE.index("block ff")]
        + """use "lib.wgt" as lib
layer body: lib.ff(dim=WIDTH)
init seed=42
"""
    )
    p = parse(source, str(tmp_path / "model.wgt"))
    assert p.layers[0].children[0].args == {"in": 4, "out": 7}


def test_error_reports_multiline_start():
    with pytest.raises(WgtError, match="unknown parameter") as exc:
        parse(SOURCE.replace("dim=WIDTH,", "dim=MISSING,"))
    assert exc.value.line == 10


def test_compiled_artifact_round_trip(tmp_path):
    from weightlang import UntrainedWeightsWarning, load_artifact
    from weightlang.compiler import compile_file

    source = tmp_path / "model.wgt"
    source.write_text(SOURCE)
    result = compile_file(source, tmp_path / "weights")
    with pytest.warns(UntrainedWeightsWarning):
        loaded = load_artifact(result.path)
    x = torch.randn(2, 4)
    assert torch.equal(result.module(x), loaded.predict(x))
    assert "body.up.weight" in loaded.module.state_dict()
