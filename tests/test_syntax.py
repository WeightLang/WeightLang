import pytest

from weightlang import WgtError, parse

BASE = """wgt 0.1
model m
input  x: f32[2]
output y: f32[1]
layer a: linear(in=2, out=4)
layer r: relu
layer b: linear(in=4, out=1)
init seed=1
"""


def test_parses_examples():
    from conftest import EXAMPLES
    from weightlang import parse_file

    progs = {p.stem: parse_file(p) for p in EXAMPLES.glob("*.wgt")}
    xor = progs["xor"]
    assert xor.model == "xor_mlp"
    assert [layer.op for layer in xor.layers] == ["linear", "tanh", "linear", "sigmoid"]
    assert xor.datasets["xor"].targets == [[0.0], [1.0], [1.0], [0.0]]
    assert xor.train is not None and xor.train.epochs == 1000
    reuse = progs["xnor_reuse"]
    assert reuse.imports[0].target == "hidden" and reuse.imports[0].freeze
    assert progs["untrained"].train is None


def test_minimal_program_and_comments():
    p = parse(BASE + "# trailing comment\n")
    assert p.seed == 1 and p.input.shape == (2,)


@pytest.mark.parametrize(
    "mutation, message, line",
    [
        (lambda s: s.replace("wgt 0.1", "wgt 9.9"), "unsupported language version", 1),
        (lambda s: s.replace("wgt 0.1\n", ""), "must start with a version header", 1),
        (lambda s: s.replace("in=4, out=1", "in=5, out=1"), "expects in=5 but receives 4", 7),
        (lambda s: s.replace("output y: f32[1]", "output y: f32[3]"), "declares 3", 4),
        (lambda s: s.replace("relu", "gelu"), "'gelu' requires 'wgt 0.2'", 6),
        (lambda s: s.replace("relu", "missing_op"), "unknown layer operation or block 'missing_op'", 6),
        (lambda s: s.replace("f32[2]", "f16[2]"), "supports only f32", 3),
        (lambda s: s.replace("f32[2]", "f32[2, 2]"), "1-D", 3),
        (lambda s: s.replace("init seed=1\n", ""), "missing 'init seed", None),
        (lambda s: s + "train data=nope loss=mse optimizer=sgd lr=0.1 epochs=3\n", "unknown dataset 'nope'", 9),
        (lambda s: s + "train data=d loss=mse optimizer=sgd lr=0.1\n", "missing: epochs", 9),
        (lambda s: s + 'import a from "x" freeze\n', "freeze only makes sense", 9),
        (lambda s: s + 'import r from "x"\n', "has no parameters", 9),
        (lambda s: s + 'import a from "x" sha256=abc\n', "64 lowercase hex", 9),
        (lambda s: s + "  [1, 2] -> [3]\n", "unexpected indentation", 9),
        (lambda s: s + "data d:\n  [1, 2] -> [3, 4]\n", "2 target values per row but output declares 1", 9),
        (lambda s: s.replace("layer r: relu", "layer a: relu"), "duplicate layer name", 6),
        (lambda s: s + "frobnicate\n", "unknown statement", 9),
    ],
)
def test_errors_point_at_lines(mutation, message, line):
    with pytest.raises(WgtError) as exc:
        parse(mutation(BASE), "t.wgt")
    assert message in str(exc.value)
    assert exc.value.line == line


def test_all_layers_frozen_rejected():
    src = BASE.replace("layer b: linear(in=4, out=1)\n", "").replace("out=4", "out=1").replace("layer r: relu\n", "")
    src += 'data d:\n  [0, 0] -> [0]\nimport a from "x" freeze\ntrain data=d loss=mse optimizer=sgd lr=0.1 epochs=1\n'
    with pytest.raises(WgtError, match="nothing to update"):
        parse(src)
