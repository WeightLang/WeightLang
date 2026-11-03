import pytest
import torch

from weightlang import WgtError, load, migrate_parameters, parameter_inventory, parse
from test_primitives import variants

SOURCE = '''wgt 0.3
model named_sharing
input tokens: i64[3]
output logits: f32[3,7]
layer unused: embedding(vocab=7,dim=4) from input
layer tokens_table: embedding(vocab=7,dim=4) from input
layer head: linear(in=4,out=7,bias=0)
parameter vocabulary = tokens_table.weight
share head.weight = vocabulary
init seed=7
'''


def test_named_share_roundtrip_export_and_gradient(tmp_path):
    for model in variants(tmp_path,SOURCE):
        assert model.head.weight is model.tokens_table.weight
        assert model.head.weight is not model.unused.weight
        x = torch.tensor([[1,2,3],[2,4,5]])
        out = model(x)['logits']
        weight = model.tokens_table.weight
        expected = torch.nn.functional.linear(torch.nn.functional.embedding(x,weight),weight)
        torch.testing.assert_close(out,expected)
        ga = torch.autograd.grad(out.square().sum(),weight,retain_graph=True)[0]
        gb = torch.autograd.grad(expected.square().sum(),weight)[0]
        torch.testing.assert_close(ga,gb)
        assert parameter_inventory(model)['id:vocabulary']['paths'] == ['tokens_table.weight','head.weight']


def test_identity_survives_rename_and_migration_reports_conflicts(tmp_path):
    a,b = tmp_path/'a.wgt',tmp_path/'b.wgt'
    a.write_text(SOURCE)
    b.write_text(SOURCE.replace('tokens_table','renamed'))
    old,new = load(a),load(b)
    with torch.no_grad():
        old.tokens_table.weight.add_(1)
    before = new.renamed.weight.detach().clone()
    report = migrate_parameters(old,new)
    assert {'target':'id:vocabulary','source':'id:vocabulary'} in report['reusable']
    torch.testing.assert_close(before,new.renamed.weight)
    migrate_parameters(old,new,apply=True)
    torch.testing.assert_close(old.tokens_table.weight,new.renamed.weight)
    assert new.renamed.weight is new.head.weight
    b.write_text(SOURCE.replace('dim=4','dim=5').replace('in=4','in=5'))
    incompatible = load(b)
    report = migrate_parameters(old,incompatible)
    assert any(x['target'] == 'id:vocabulary' for x in report['conflicts'])
    with pytest.raises(WgtError,match='conflicts'):
        migrate_parameters(old,incompatible,apply=True)


@pytest.mark.parametrize('old,new',[
    ('= tokens_table.weight','= missing.weight'),
    ('= vocabulary','= missing'),
    ('out=7','out=8'),
    ('share head.weight = vocabulary','share tokens_table.weight = vocabulary'),
    ('parameter vocabulary = tokens_table.weight','parameter vocabulary = tokens_table.weight\nparameter vocabulary = unused.weight'),
])
def test_invalid_identity_rejected(old,new):
    with pytest.raises(WgtError):
        parse(SOURCE.replace(old,new))
