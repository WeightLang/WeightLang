"""Behavioral contracts for the opt-in dynamic Torch frontend."""
from pathlib import Path
import json

import pytest
import torch

from weightlang import compile_file, load_artifact, save_artifact
from weightlang.syntax import WgtError
from weightlang.tensor_program import HEADER, parse_tensor_program, build_tensor_program

SOURCE = HEADER + '''host weightlang.tensor_program
extern

module Memory(Module):
    setup(self, width=4):
        super().__init__()
        self.memory = gru(3, width, 2, batch_first=True)
        self.head = linear(width, 2)
    fn forward(self, x, state=None):
        output, state = self.memory(x, state)
        return self.head(output), state
'''.replace('extern\n', 'extern \n')


def test_dynamic_roundtrip_and_stream(tmp_path):
    source=tmp_path/'memory.wgt'; source.write_text(SOURCE)
    parsed=parse_tensor_program(SOURCE)
    assert parsed.host=='weightlang.tensor_program'
    artifact=compile_file(source,tmp_path/'weights',no_train=True)
    model=artifact.module
    x=torch.randn(2,7,3)
    whole,state=model(x)
    first,h=model(x[:,:3]); last,h=model(x[:,3:],h)
    torch.testing.assert_close(torch.cat((first,last),1),whole)
    torch.testing.assert_close(h,state)
    with pytest.raises(WgtError,match='trusted'):
        load_artifact(artifact.path)
    loaded=load_artifact(artifact.path,trusted_tensor_source=True,warn_untrained=False)
    torch.testing.assert_close(loaded.module(x)[0],whole)
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=.01)
    old=model(x)[0].detach().clone()
    model(x)[0].square().mean().backward(); opt.step()
    assert not torch.equal(old,model(x)[0])
    updated=save_artifact(artifact,tmp_path/'updated',training={'method':'AdamW','steps':1})
    torch.testing.assert_close(updated.module(x)[0],model(x)[0])
    with pytest.raises(WgtError,match='exists'):
        artifact.save(tmp_path/'updated')
    (updated.path/'weights.safetensors').write_bytes(b'invalid')
    with pytest.raises(WgtError,match='hash'):
        load_artifact(updated.path,trusted_tensor_source=True)


def test_source_is_authoritative_and_snapshotted(tmp_path):
    source=tmp_path/'memory.wgt'; source.write_text(SOURCE)
    old=build_tensor_program(source,tmp_path/'a')
    source.write_text(SOURCE.replace('return self.head(output), state','return self.head(output) * 0, state'))
    new=build_tensor_program(source,tmp_path/'b')
    new.module.load_state_dict(old.module.state_dict())
    x=torch.ones(1,2,3)
    assert torch.count_nonzero(new.module(x)[0])==0
    restored=load_artifact(old.path,trusted_tensor_source=True,warn_untrained=False)
    torch.testing.assert_close(restored.module(x)[0],old.module(x)[0])
    assert torch.count_nonzero(restored.module(x)[0])>0


def test_freeze_hooks_and_nonfinite(tmp_path):
    source=tmp_path/'memory.wgt'; source.write_text(SOURCE)
    artifact=build_tensor_program(source,tmp_path/'base')
    artifact.module.head.weight.requires_grad_(False)
    frozen=artifact.save(tmp_path/'frozen')
    assert not frozen.module.head.weight.requires_grad
    with torch.no_grad(): frozen.module.head.weight.add_(1)
    with pytest.raises(WgtError,match='frozen'):
        frozen.save(tmp_path/'bad')
    hook=artifact.module.register_forward_hook(lambda m,a,o:o)
    with pytest.raises(WgtError,match='hooks'):
        artifact.save(tmp_path/'hooked')
    hook.remove()
    with torch.no_grad(): artifact.module.head.bias.fill_(float('nan'))
    with pytest.raises(WgtError,match='nonfinite'):
        artifact.save(tmp_path/'nan')


def test_check_does_not_execute_and_rejects_imports(tmp_path):
    source=SOURCE.replace('super().__init__()', 'raise RuntimeError("must not execute at check")')
    parse_tensor_program(source)
    with pytest.raises(WgtError,match='imports'):
        parse_tensor_program(SOURCE.replace('super().__init__()', 'import os'))
    with pytest.raises(WgtError):
        parse_tensor_program(SOURCE.replace('fn forward', 'fn forward?'))
