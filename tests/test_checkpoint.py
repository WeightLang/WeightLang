import random

import numpy as np
import pytest
import torch

from weightlang import WgtError, checkpoint_report, load, restore_checkpoint, save_checkpoint

CODE = '''wgt 0.3
model resume
input x: f32[4]
output y: f32[2]
layer hidden: linear(in=4,out=2)
parameter projection = hidden.weight
parameter bias = hidden.bias
init seed=7
'''


def step(model,optimizer):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(3,4) + random.random() + float(np.random.rand())
    loss = model(x)['y'].square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


@pytest.mark.parametrize('optimizer_class',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
def test_resume_reordered_parameters_and_renamed_layer_exact(tmp_path,optimizer_class):
    path = tmp_path/'model.wgt'
    path.write_text(CODE)
    model = load(path)
    optimizer = optimizer_class(model.parameters(),lr=0.01)
    step(model,optimizer)
    checkpoint = save_checkpoint(model,optimizer,tmp_path/'training.pt',progress={'step':1})
    expected = step(model,optimizer)
    weights = {k:v.detach().clone() for k,v in model.state_dict().items()}
    path.write_text(CODE.replace('hidden','renamed'))
    resumed = load(path)
    other = optimizer_class(list(resumed.parameters())[::-1],lr=0.8)
    assert not checkpoint_report(resumed,checkpoint)['conflicts']
    progress = restore_checkpoint(resumed,other,checkpoint)
    assert progress == {'step':1}
    actual = step(resumed,other)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for key,value in resumed.state_dict().items():
        torch.testing.assert_close(value,weights[key.replace('renamed','hidden')],rtol=0,atol=0)
    with pytest.raises(WgtError,match='exists'):
        save_checkpoint(resumed,other,checkpoint)


def test_incompatible_checkpoint_is_reported_without_mutation(tmp_path):
    path = tmp_path/'model.wgt'
    path.write_text(CODE)
    old = load(path)
    opt = torch.optim.AdamW(old.parameters())
    checkpoint = save_checkpoint(old,opt,tmp_path/'training.pt')
    path.write_text(CODE.replace('f32[2]','f32[3]').replace('out=2','out=3'))
    new = load(path)
    before = {k:v.detach().clone() for k,v in new.state_dict().items()}
    optimizer = torch.optim.AdamW(new.parameters())
    assert checkpoint_report(new,checkpoint)['conflicts']
    with pytest.raises(WgtError,match='incompatible'):
        restore_checkpoint(new,optimizer,checkpoint)
    for key,value in new.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert not optimizer.state
