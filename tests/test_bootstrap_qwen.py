"""Numerical migration checks; tiny configs are fixtures, never pretrained evidence."""
from pathlib import Path
import torch
import pytest
from weightlang.tensor_program import parse_tensor_program

SOURCE = Path(__file__).resolve().parents[1] / 'examples/bootstrap/qwen/model.wgt'


def test_qwen_text_forward_gradient_and_surgery():
    hf=pytest.importorskip("transformers",exc_type=ImportError)
    if hf.__version__ != "5.7.0":
        pytest.skip("reference pinned to Transformers 5.7.0")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.set_num_threads(2)
    torch.manual_seed(12)
    c = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=64,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,
        linear_value_head_dim=8,linear_conv_kernel_dim=4,
        layer_types=['linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000000.0,
                         'partial_rotary_factor':0.5,'mrope_section':[1,1,0]})
    c._attn_implementation = 'eager'
    upstream = Qwen3_5TextModel(c).eval()
    cls = parse_tensor_program(SOURCE.read_text()).classes({})['QwenText']
    model = cls(c.to_dict(),rank=4).eval()
    result = model.load_state_dict(upstream.state_dict(),strict=False)
    assert set(result.missing_keys) == {'surgery.down.weight','surgery.up.weight'}
    assert not result.unexpected_keys
    for length in [1,7,65]:
        ids=torch.randint(0,64,(1,length))
        expected=upstream(ids,use_cache=False).last_hidden_state
        actual=model.hidden(ids)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for name,p in upstream.named_parameters():
        torch.testing.assert_close(dict(model.named_parameters())[name].grad,p.grad,atol=2e-5,rtol=2e-4)
    x=torch.randn(2,3,32)
    torch.testing.assert_close(model.surgery(x),x,rtol=0,atol=0)
    opt=torch.optim.SGD(model.surgery.parameters(),lr=0.1)
    opt.zero_grad(); model.surgery(x).square().mean().backward(); opt.step()
    assert not torch.equal(model.surgery(x),x)
    model.eval()
    ids=torch.randint(0,64,(1,9)); changed=ids.clone(); changed[:,5:]=0
    torch.testing.assert_close(model(ids)[:,:5],model(changed)[:,:5],rtol=0,atol=0)
