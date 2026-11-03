"""Bundled imports expand into executable layers without package resolution."""
import pytest
import torch

from weightlang.artifact import build_module
from weightlang.syntax import WgtError, parse, parse_file

SOURCE = '''wgt 0.3
import std/nn as n
model builtin
# feature 是单样本特征轴；MLP 将 4 维映射到 8 维再回到 4 维。
input x: f32[feature=4]
output y: f32[feature=4]
# 两次投影的权重独立，参数仅初始化，尚未训练。
layer body: n.mlp(in=4, hidden=8, out=4)
init seed=42
'''


def load(path):
    program = parse_file(path)
    torch.manual_seed(program.seed)
    return build_module([layer.to_dict() for layer in program.layers])


def test_builtin_import_forward_and_gradients(tmp_path):
    path = tmp_path / 'model.wgt'
    path.write_text(SOURCE)
    imported = load(path)
    path.write_text(SOURCE.replace('import std/nn', 'use std/nn'))
    existing = load(path)
    x = torch.randn(2, 4, requires_grad=True)
    torch.testing.assert_close(imported(x), existing(x), rtol=0, atol=0)
    imported(x).sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in imported.parameters())
    assert not (tmp_path / 'wgt.sum').exists()
    assert not (tmp_path / 'weights').exists()


def test_builtin_import_default_alias_and_library(tmp_path):
    parse(SOURCE.replace(' as n', '').replace('n.mlp', 'nn.mlp'))
    library = tmp_path / 'layers.wgt'
    library.write_text('wgt 0.3\nimport std/nn\nblock body(dim):\n  layer ff: nn.mlp(in=dim, hidden=dim*2, out=dim)\n')
    path = tmp_path / 'model.wgt'
    path.write_text(SOURCE.replace('import std/nn as n', 'use "./layers.wgt" as n').replace('n.mlp(in=4, hidden=8, out=4)', 'n.body(dim=4)'))
    assert load(path)(torch.ones(2, 4)).shape == (2, 4)


@pytest.mark.parametrize('source, message', [
    (SOURCE.replace('wgt 0.3', 'wgt 0.2'), 'requires'),
    (SOURCE.replace('model builtin', 'use std/nn as n\nmodel builtin'), 'duplicate'),
    (SOURCE.replace('std/nn', 'std/does_not_exist'), 'library'),
])
def test_invalid_builtin_import(source, message):
    with pytest.raises(WgtError, match=message):
        parse(source)


def test_weight_import_named_std_still_parses():
    program = parse(SOURCE.replace('layer body:', 'layer std:') + 'import std from "./weights/base" layer=body\n')
    assert program.imports[0].target == 'std'
