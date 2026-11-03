import pytest
import torch

from weightlang import compile_file, load_artifact, merge_adapter, unmerge_adapter, save_adapter, load_adapter, WgtError
from test_primitives import variants

CODE = '''wgt 0.3
model adapter_model
input x: f32[4]
output y: f32[3]
layer projection: linear(in=4,out=3)
parameter projection_weights = projection.weight
adapter task: lora target=projection rank=2 alpha=4
init seed=9
'''


def test_declared_adapter_gradients_roundtrip_merge_and_exact_unmerge(tmp_path):
    for model in variants(tmp_path,CODE):
        module = model.projection
        assert not module.weight.requires_grad and not module.bias.requires_grad
        with torch.no_grad():
            module.lora_b.normal_()
        x = torch.randn(2,4,requires_grad=True)
        output = model(x)['y']
        expected = torch.nn.functional.linear(x,module.weight,module.bias)+(x @ module.lora_a.T @ module.lora_b.T)*2
        torch.testing.assert_close(output,expected)
        output.square().sum().backward()
        assert module.weight.grad is None and module.lora_a.grad is not None and module.lora_b.grad is not None
        base = module.weight.detach().clone()
        merge_adapter(model,'task')
        torch.testing.assert_close(model(x)['y'],output)
        merge_adapter(model,'task')
        torch.testing.assert_close(model(x)['y'],output)
        with pytest.raises(ValueError,match='unmerge'):
            module.train()
        unmerge_adapter(model,'task')
        torch.testing.assert_close(module.weight,base,rtol=0,atol=0)


def test_independent_adapter_and_full_save_when_merged(tmp_path):
    path = tmp_path/'model.wgt'
    path.write_text(CODE)
    built = compile_file(path,tmp_path/'built',no_train=True)
    artifact = load_artifact(built.path,warn_untrained=False)
    model = artifact.module
    with torch.no_grad():
        model.projection.lora_b.normal_()
    adapter = save_adapter(model,'task',tmp_path/'task.pt')
    other = load_artifact(built.path,warn_untrained=False).module
    load_adapter(other,'task',adapter)
    x = torch.randn(2,4)
    torch.testing.assert_close(model(x)['y'],other(x)['y'])
    merge_adapter(model,'task')
    saved = artifact.save(tmp_path/'merged_saved')
    torch.testing.assert_close(saved.module(x)['y'],model(x)['y'])
    with torch.no_grad():
        other.projection.weight.add_(1)
    with pytest.raises(WgtError,match='digest'):
        load_adapter(other,'task',adapter)
