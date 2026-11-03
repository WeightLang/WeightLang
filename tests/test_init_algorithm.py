import pytest
import torch

from weightlang import load, compile_file, WgtError
from weightlang.syntax import parse_file
from weightlang.bundle import build_bundle, install_bundle

MODEL = '''wgt 0.3
model algorithm
input x: f32[4]
output y: f32[4]
layer head: linear(4)
init file="init.wgt" seed=42
'''

# The orthogonal basis is fully described by WGT, not a built-in initializer.
DCT = '''wgt init 1
algorithm basis(rows, cols):
    let size = max(rows, cols)
    let count = min(rows, cols)
    let frequency = reshape(arange(count), [count, 1])
    let position = reshape(arange(size) + 0.5, [1, size])
    let result = cos(pi * frequency * position / size) * sqrt(2 / size)
    result[0, :] = result[0, :] / sqrt(2)
    if rows > cols:
        return transpose(result, 0, 1)
    return result

tensor "*":
    if rank == 2:
        return basis(dim(0), dim(1))
    return zeros(shape)
'''


def source(tmp_path, rules=DCT):
    path = tmp_path / 'model.wgt'
    path.write_text(MODEL)
    (tmp_path / 'init.wgt').write_text(rules)
    return path


def test_wgt_computes_algorithm_and_artifact_without_python(tmp_path, monkeypatch):
    path = source(tmp_path)
    import weightlang.plugins as plugins
    monkeypatch.setattr(plugins, 'load_callback', lambda *a: pytest.fail('WGT must not load Python'))
    module = load(path)
    positions = torch.arange(4, dtype=torch.float32)[None, :] + 0.5
    frequencies = torch.arange(4, dtype=torch.float32)[:, None]
    expected = torch.cos(torch.pi * frequencies * positions / 4) * (2 / 4) ** 0.5
    expected[0] /= 2 ** 0.5
    torch.testing.assert_close(module.head.weight, expected)
    torch.testing.assert_close(module.head.weight @ module.head.weight.T, torch.eye(4), atol=1e-6, rtol=1e-6)
    x = torch.ones(2, 4, requires_grad=True)
    module(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    built = compile_file(path, tmp_path / 'saved', no_train=True)
    assert built.manifest['init']['method'] == 'wgt-algorithm'
    (tmp_path / 'init.wgt').unlink()
    torch.testing.assert_close(load(built.path)(x), module(x))


def test_iterative_algorithm_with_for_while_and_index_assignment(tmp_path):
    path = source(tmp_path, '''wgt init 1
algorithm refine(n, iterations):
    let result = zeros([n, n])
    for row in range(n):
        for col in range(n):
            if row == col:
                result[row, col] = 2
            else:
                result[row, col] = 0.25
    let step = 0
    while step < iterations:
        # Newton-Schulz orthogonalization, computed by the user's algorithm.
        result = result / maximum(norm(result), 1)
        result = 0.5 * result @ (3 * eye(n) - transpose(result, 0, 1) @ result)
        step += 1
    return result
tensor "*.weight":
    return refine(dim(0), 4)
tensor "*.bias":
    let output = zeros(shape)
    for i in range(dim(0)):
        if i == 1:
            continue
        if i == 3:
            break
        output[i] = i + 1
    return output
''')
    model = load(path)
    expected = torch.full((4, 4), 0.25)
    expected.fill_diagonal_(2)
    for _ in range(4):
        expected = expected / max(torch.linalg.vector_norm(expected), 1)
        expected = 0.5 * expected @ (3 * torch.eye(4) - expected.T @ expected)
    torch.testing.assert_close(model.head.weight, expected)
    torch.testing.assert_close(model.head.bias, torch.tensor([1., 0., 3., 0.]))


def test_algorithm_check_does_not_evaluate(tmp_path):
    path = source(tmp_path, '''wgt init 1
tensor "*":
    return zeros([1 / 0])
''')
    parse_file(path)
    with pytest.raises(WgtError, match='init.wgt:3'):
        load(path)


@pytest.mark.parametrize('statement', [
    'import os', 'return open(1)', 'return zeros(shape).numpy()',
    'return [x for x in range(2)]', 'return __import__(1)',
    'return lambda: 1',
])
def test_unsupported_host_execution_rejected_statically(tmp_path, statement):
    path = source(tmp_path, f'wgt init 1\ntensor "*":\n    {statement}\n')
    with pytest.raises(WgtError):
        parse_file(path)


def test_budget_exhaustion_and_rng_restoration(tmp_path):
    path = source(tmp_path, '''wgt init 1
tensor "*":
    let x = randn(shape)
    while True:
        let n = 1
    return x
''')
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(WgtError, match='100000 evaluation steps'):
        load(path)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_algorithm_bundles_and_tied_weights(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    path = source(root)
    path.write_text('''wgt 0.3
model algorithm
input x: i64[2]
output y: f32[2, 5]
layer embed: embedding(vocab=5, dim=4)
layer head: linear(in=4, out=5, bias=0, tie_embedding=1)
init file="init.wgt"
''')
    archive = tmp_path / 'algorithm.wgtpkg'
    build_bundle(root, archive)
    install_bundle(archive, tmp_path / 'installed')
    module = load(tmp_path / 'installed')
    assert module.embed.weight is module.head.weight
    torch.testing.assert_close(module.head.weight.T @ module.head.weight, torch.eye(4), atol=1e-6, rtol=1e-6)


def test_seeded_custom_qr_algorithm(tmp_path):
    path = source(tmp_path, '''wgt init 1
tensor "*.weight":
    let factors = qr(randn(shape))
    return factors[0]
tensor "*.bias":
    return zeros(shape)
''')
    a, b = load(path), load(path)
    assert torch.equal(a.head.weight, b.head.weight)
    torch.testing.assert_close(a.head.weight.T @ a.head.weight, torch.eye(4), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('body', [
    'return zeros([1])', 'return full(shape, 1e999)',
    'break', 'continue', 'return missing(shape)',
])
def test_bad_algorithm_contract_or_control_flow(tmp_path, body):
    path = source(tmp_path, f'wgt init 1\ntensor "*":\n    {body}\n')
    with pytest.raises(WgtError):
        load(path)


def test_user_function_arguments_and_recursion_limit(tmp_path):
    path = source(tmp_path, 'wgt init 1\nalgorithm recurse(n):\n    return recurse(n + 1)\ntensor "*":\n    return recurse(0)\n')
    with pytest.raises(WgtError, match='64 call frames'):
        load(path)


def test_bundle_rejects_external_rules(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    path = source(root)
    outside = tmp_path / 'outside.wgt'
    outside.write_text(DCT)
    path.write_text(MODEL.replace('init.wgt', str(outside)))
    with pytest.raises(WgtError, match='inside the bundle'):
        build_bundle(root, tmp_path / 'outside.wgtpkg')


def test_linear_algebra_reductions_and_selection(tmp_path):
    path = source(tmp_path, '''wgt init 1
tensor "*.weight":
    let a = eye(dim(0)) * 2
    let inverse = solve(a, eye(dim(0)))
    let factors = svd(a)
    let restored = factors[0] @ diag(factors[1]) @ factors[2]
    return restored @ inverse
tensor "*.bias":
    let x = linspace(-2, 2, dim(0))
    let positive = where(x > 0, x, zeros(shape))
    let joined = cat([unsqueeze(positive, 0), unsqueeze(positive, 0)], 0)
    return clamp(mean(joined, 0), 0, 1)
''')
    model = load(path)
    torch.testing.assert_close(model.head.weight, torch.eye(4))
    torch.testing.assert_close(model.head.bias, torch.tensor([0., 0., 2 / 3, 1.]))


def test_one_line_rule_cannot_capture_following_indentation(tmp_path):
    path = source(tmp_path, '''wgt init 1
algorithm helper():
    return 1
tensor "*" zeros
    return zeros(shape)
''')
    with pytest.raises(WgtError, match='unexpected indented'):
        parse_file(path)
