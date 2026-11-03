"""Numerical evidence and independent Lean proof checks have separate scopes."""
import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from weightlang.lean import LinearFormula, FormulaGraph, FormulaNode
from weightlang.syntax import WgtError, parse


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('rows', [0, 1, 7])
def test_forward_gradients_and_row_concatenation(bias, rows):
    formula = LinearFormula(3, 2, bias)
    model = formula.module()
    x = torch.randn(rows, 3, requires_grad=True)
    actual = model(x)
    expected = x @ model.projection.weight.T
    if bias:
        expected = expected + model.projection.bias
    torch.testing.assert_close(actual, expected)
    args = (x, *model.parameters())
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * upstream).sum(), args, retain_graph=True)
    reference = torch.autograd.grad((expected * upstream).sum(), args)
    for got, want in zip(grads, reference):
        torch.testing.assert_close(got, want)
    torch.testing.assert_close(actual, torch.cat([model(x[:1]), model(x[1:])]))
    assert parse(formula.wgt_source()).layers[0].args['in'] == 3


def test_load_has_no_output_or_rng_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = torch.random.get_rng_state().clone()
    LinearFormula(3, 2).module()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('args', [(0, 2), (3, -1), (True, 2), (2.0, 3), (2, 3, 1)])
def test_bad_contracts(args):
    with pytest.raises(ValueError):
        LinearFormula(*args)


def test_runtime_rejects_bad_shape_and_dtype():
    model = LinearFormula(3, 2).module()
    for tensor in (torch.ones(4, 4), torch.ones(4, 3, dtype=torch.int64)):
        with pytest.raises(RuntimeError):
            model(tensor)


def test_missing_lean_fails():
    with pytest.raises(WgtError, match='not found'):
        LinearFormula(3, 2).verify(lean='/nonexistent/lean')


def test_checked_in_examples_match_generator():
    root = Path(__file__).resolve().parents[1] / 'examples/lean'
    formula = LinearFormula(3, 2)
    assert (root / 'model.wgt').read_text() == formula.wgt_source()
    assert (root / 'Linear.lean').read_text() == formula.lean_source()


@pytest.mark.parametrize('bias', [True, False])
def test_real_lean_and_stale_records(bias):
    lean = os.environ.get('WEIGHTLANG_LEAN', 'lean')
    if not shutil.which(lean):
        pytest.skip('set WEIGHTLANG_LEAN to pinned Lean 4.19.0 for proof checks')
    formula = LinearFormula(3, 2, bias)
    record = json.loads(json.dumps(formula.verify(lean=lean)))
    assert formula.matches_record(record)
    assert not LinearFormula(4, 2, bias).matches_record(record)
    assert not LinearFormula(3, 2, not bias).matches_record(record)
    stale_tool = {**record, 'lean_version': 'Lean (version 4.18.0, test)'}
    assert not formula.matches_record(stale_tool)
    stale_generator = {**record, 'bridge_sha256': 'changed'}
    assert not formula.matches_record(stale_generator)
    record['lean_sha256'] = 'stale'
    assert not formula.matches_record(record)


@pytest.mark.parametrize('replacement', ['exact false', 'sorry', 'exact False.elim cheat'])
def test_lean_rejects_broken_proof(monkeypatch, replacement):
    lean = os.environ.get('WEIGHTLANG_LEAN', 'lean')
    if not shutil.which(lean):
        pytest.skip('Lean not installed')
    original = LinearFormula.lean_source
    def broken(self):
        source = original(self).replace('cases r <;> rfl', replacement)
        if 'cheat' in replacement:
            source = source.replace('theorem linear_rows', 'axiom cheat : False\n\ntheorem linear_rows')
        return source
    monkeypatch.setattr(LinearFormula, 'lean_source', broken)
    with pytest.raises(WgtError, match='Lean proof failed|unexpected proof axioms'):
        LinearFormula(3, 2).verify(lean=lean)


def residual_graph():
    return FormulaGraph(3, (
        FormulaNode('up', 'linear', ('input',), 4),
        FormulaNode('square', 'mul', ('up', 'up')),
        FormulaNode('down', 'linear', ('square',), 3, False),
        FormulaNode('skip', 'add', ('down', 'input')),
    ))


@pytest.mark.parametrize('shape', [(3,), (0, 3), (1, 3), (7, 3)])
def test_graph_forward_backward_and_shared_subexpression(shape):
    graph = residual_graph()
    model = graph.module()
    x = torch.randn(shape, requires_grad=True)
    up = x @ model.n0.weight.T + model.n0.bias
    expected = (up * up) @ model.n2.weight.T + x
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    assert len(list(model.parameters())) == 3  # reusing up does not duplicate weights
    args = (x, *model.parameters())
    upstream = torch.randn_like(actual)
    got = torch.autograd.grad((actual * upstream).sum(), args, retain_graph=True)
    want = torch.autograd.grad((expected * upstream).sum(), args)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b)
    if len(shape) == 2:
        torch.testing.assert_close(actual, torch.cat([model(x[:2]), model(x[2:])]))


@pytest.mark.parametrize('nodes', [
    (),
    (FormulaNode('a', 'add', ('input', 'later')),),
    (FormulaNode('a', 'add', ('input', 'a')),),
    (FormulaNode('a', 'linear', ('input',), 2), FormulaNode('a', 'linear', ('a',), 3)),
    (FormulaNode('a', 'linear', ('input',), 2), FormulaNode('b', 'add', ('a', 'input'))),
    (FormulaNode('unused', 'linear', ('input',), 2), FormulaNode('out', 'add', ('input', 'input'))),
])
def test_invalid_graphs_fail_before_allocation(nodes):
    with pytest.raises(ValueError):
        FormulaGraph(3, nodes)


@pytest.mark.parametrize('kwargs', [
    dict(name='input', op='add', inputs=('input', 'input')),
    dict(name='x', op='softmax', inputs=('input',)),
    dict(name='x', op='linear', inputs=('input',), out_features=True),
    dict(name='x', op='add', inputs=['input', 'input']),
    dict(name='x', op='mul', inputs=('input',)),
    dict(name='x', op='mul', inputs=('input', 'input'), bias=False),
])
def test_invalid_nodes(kwargs):
    with pytest.raises(ValueError):
        FormulaNode(**kwargs)


@pytest.mark.parametrize('graph', [
    residual_graph(),
    FormulaGraph(2, (FormulaNode('out', 'add', ('input', 'input')),)),
    FormulaGraph(2, (FormulaNode('out', 'linear', ('input',), 1, False),)),
])
def test_graph_actual_lean(graph):
    lean = os.environ.get('WEIGHTLANG_LEAN', 'lean')
    if not shutil.which(lean):
        pytest.skip('Lean not installed')
    record = graph.verify(lean=lean)
    assert record['theorem'] == 'WeightLang.graph_rows'
    assert graph.matches_record(record)
    assert not LinearFormula(2, 1, False).matches_record(record)
    assert not graph.matches_record({**record, 'theorem': 'WeightLang.linear_rows'})
    other = FormulaGraph(graph.in_features, (FormulaNode('other', 'mul', ('input', 'input')),))
    assert not other.matches_record(record)


def test_graph_examples_and_parameter_free_execution():
    import runpy
    root = Path(__file__).resolve().parents[1] / 'examples/lean'
    graph = runpy.run_path(str(root / 'graph.py'))['formula']()
    assert (root / 'graph.wgt').read_text() == graph.wgt_source()
    assert (root / 'Graph.lean').read_text() == graph.lean_source()
    model = FormulaGraph(2, (FormulaNode('double', 'add', ('input', 'input')),)).module()
    assert list(model.parameters()) == []
    x = torch.randn(5, 2, requires_grad=True)
    torch.testing.assert_close(model(x), 2 * x)
    torch.testing.assert_close(torch.autograd.grad(model(x).sum(), x)[0], torch.full_like(x, 2))


def test_graph_export_load_and_artifact_roundtrip(tmp_path):
    from weightlang import load, compile_file, load_artifact
    graph = residual_graph()
    source = tmp_path / 'graph.wgt'
    source.write_text(graph.wgt_source())
    model = graph.module()
    loaded = load(source)
    artifact = compile_file(source, tmp_path / 'weights', no_train=True)
    restored = load_artifact(artifact.path, warn_untrained=False).module
    x = torch.randn(5, 3)
    for other in (loaded, restored):
        torch.testing.assert_close(other(x), model(x), rtol=0, atol=0)
        assert other.state_dict().keys() == model.state_dict().keys()
