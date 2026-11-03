import pytest
import torch
from weightlang import load, save_adapter, load_adapter, merge_adapter, unmerge_adapter
from weightlang.adapters import LoRALinear, DoRALinear, PiSSALinear, QLoRALinear
from test_primitives import variants


def code(method):
    return f'''wgt 0.3
model adapted
input x: f32[8]
output y: f32[5] from projection
layer projection: linear(in=8,out=5)
adapter task: {method} target=projection rank=3 alpha=6
init seed=7
'''


@pytest.mark.parametrize('cls', [LoRALinear, DoRALinear, PiSSALinear, QLoRALinear])
def test_math_initialization_training_and_input_gradient(cls):
    torch.manual_seed(13)
    base = torch.nn.Linear(8, 5)
    weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
    layer = cls(base, 3, 6, 1)
    x = torch.randn(12, 8, requires_grad=True)
    initial = layer(x)
    reference_weight = layer.weight if cls is QLoRALinear else weight
    torch.testing.assert_close(initial, torch.nn.functional.linear(x, reference_weight, bias))
    if cls is QLoRALinear:
        assert 'weight' not in dict(layer.named_parameters())
        assert layer.packed.dtype == torch.uint8 and layer.packed.numel() == 20
        assert (reference_weight - weight).abs().max() < .1
    if cls is PiSSALinear:
        assert layer.lora_b.abs().sum() > 0
        torch.testing.assert_close(layer.weight + 2 * layer.lora_b @ layer.lora_a, weight)
    if cls is DoRALinear:
        with torch.no_grad():
            layer.lora_b.normal_(std=.03)
        direction = weight + 2 * layer.lora_b @ layer.lora_a
        expected = torch.nn.functional.linear(x, direction * (layer.magnitude / direction.norm(dim=1).detach())[:, None], bias)
    else:
        expected = torch.nn.functional.linear(x, layer.weight, bias) + 2 * (x @ layer.lora_a.T @ layer.lora_b.T)
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0], torch.autograd.grad(expected.sum(), x)[0])
    optimizer = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=.03)
    target = torch.randn(12, 5)
    before = torch.nn.functional.mse_loss(layer(x), target).item()
    frozen = layer.weight.detach().clone()
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < before * .9
    torch.testing.assert_close(layer.weight, frozen, rtol=0, atol=0)
    assert layer.lora_a.grad is not None and layer.lora_b.grad is not None
    if cls is DoRALinear:
        assert layer.magnitude.grad is not None


@pytest.mark.parametrize('method', ['dora', 'pissa', 'qlora'])
def test_four_paths_adapter_roundtrip_and_merge(tmp_path, method):
    models = variants(tmp_path, code(method))
    x = torch.randn(3, 8)
    for model in models[1:]:
        torch.testing.assert_close(model(x)['y'], models[0](x)['y'])
    for index, model in enumerate(models):
        layer = model.projection
        with torch.no_grad():
            layer.lora_b.add_(.1)
        expected = model(x)['y']
        saved = save_adapter(model, 'task', tmp_path / f'{index}.pt')
        with torch.no_grad():
            layer.lora_b.zero_()
        load_adapter(model, 'task', saved)
        torch.testing.assert_close(model(x)['y'], expected)
        model.eval()
        base = layer.weight.detach().clone()
        merge_adapter(model, 'task')
        torch.testing.assert_close(model(x)['y'], expected)
        merge_adapter(model, 'task')
        with pytest.raises(ValueError, match='unmerge'):
            layer.train()
        unmerge_adapter(model, 'task')
        torch.testing.assert_close(layer.weight, base, rtol=0, atol=0)
        torch.testing.assert_close(model(x)['y'], expected)


@pytest.mark.parametrize('method', ['dora', 'pissa', 'qlora'])
def test_pretrained_import_checkpoint_and_full_saved_model(tmp_path, method):
    from weightlang import compile_file, load_artifact, save_checkpoint, restore_checkpoint
    source = tmp_path / 'base.wgt'
    source.write_text('\n'.join(line for line in code(method).splitlines() if not line.startswith('adapter ')))
    base = compile_file(source, tmp_path / 'base', no_train=True)
    source.write_text(code(method) + f'import projection from "{base.path}"\n')
    built = compile_file(source, tmp_path / 'adapted', no_train=True)
    artifact = load_artifact(built.path, warn_untrained=False)
    model = artifact.module
    x = torch.randn(4, 8)
    original = load_artifact(base.path, warn_untrained=False).module(x)['y']
    if method == 'qlora':
        torch.testing.assert_close(model(x)['y'], original, rtol=.2, atol=.15)
    else:
        torch.testing.assert_close(model(x)['y'], original)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.03)
    def train_step(m, o):
        o.zero_grad()
        loss = m(x)['y'].square().mean()
        loss.backward()
        o.step()
    train_step(model, opt)
    checkpoint = save_checkpoint(model, opt, tmp_path / 'checkpoint.pt')
    train_step(model, opt)
    expected = model(x)['y'].detach()
    resumed = load_artifact(built.path, warn_untrained=False).module
    other = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.1)
    restore_checkpoint(resumed, other, checkpoint)
    train_step(resumed, other)
    torch.testing.assert_close(resumed(x)['y'], expected, rtol=0, atol=0)
    model.eval()
    merge_adapter(model, 'task')
    saved = artifact.save(tmp_path / 'saved')
    torch.testing.assert_close(saved.module(x)['y'], expected)


@pytest.mark.parametrize('shape', [(1, 1), (3, 7), (17, 19)])
def test_nf4_zero_odd_partial_blocks_and_storage(shape):
    layer = QLoRALinear(torch.nn.Linear(shape[1], shape[0], bias=False), 1, 1, 1)
    for weight in [torch.zeros(shape), torch.randn(shape)]:
        layer.encode(weight)
        decoded = layer.weight.clone()
        packed = layer.packed.clone()
        scales = layer.scales.clone()
        layer.encode(decoded)
        torch.testing.assert_close(layer.packed, packed, rtol=0, atol=0)
        torch.testing.assert_close(layer.scales, scales, rtol=0, atol=0)
        assert torch.isfinite(decoded).all()
        assert layer.packed.numel() == (weight.numel() + 1) // 2


@pytest.mark.parametrize('method', ['dora', 'pissa', 'qlora'])
def test_invalid_declarations_and_adapter_method_guard(tmp_path, method):
    from weightlang import parse, WgtError
    for invalid in [code(method).replace('rank=3', 'rank=9'), code(method).replace('alpha=6', 'alpha=0'), code(method).replace('alpha=6', 'alpha=6 freeze=0')]:
        with pytest.raises(WgtError):
            parse(invalid)
    source = tmp_path / 'model.wgt'
    source.write_text(code(method))
    model = load(source)
    saved = save_adapter(model, 'task', tmp_path / 'adapter.pt')
    record = torch.load(saved, weights_only=True)
    record['method'] = 'unknown'
    torch.save(record, saved)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(WgtError, match='method'):
        load_adapter(model, 'task', saved)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_checkpoint_rejects_same_shape_different_method(tmp_path):
    from weightlang import save_checkpoint, restore_checkpoint, WgtError
    source = tmp_path / 'model.wgt'
    source.write_text(code('lora'))
    original = load(source)
    saved = save_checkpoint(original, torch.optim.Adam(original.parameters()), tmp_path / 'checkpoint.pt')
    source.write_text(code('pissa'))
    other = load(source)
    with pytest.raises(WgtError, match='incompatible'):
        restore_checkpoint(other, torch.optim.Adam(other.parameters()), saved)


def test_quantized_base_stays_frozen_when_saving(tmp_path):
    from weightlang import compile_file, load_artifact, WgtError
    source = tmp_path / 'model.wgt'
    source.write_text(code('qlora'))
    artifact = load_artifact(compile_file(source, tmp_path / 'built', no_train=True).path, warn_untrained=False)
    artifact.module.projection.scales.mul_(2)
    with pytest.raises(WgtError, match='frozen'):
        artifact.save(tmp_path / 'modified')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('method', ['dora', 'pissa', 'qlora'])
def test_cuda_forward_backward_and_cpu_roundtrip(tmp_path, method):
    source = tmp_path / 'model.wgt'
    source.write_text(code(method))
    model = load(source, device='cuda')
    x = torch.randn(4, 8, device='cuda', requires_grad=True)
    y = model(x)['y']
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.projection.lora_b.grad is not None
    model.cpu()
    torch.testing.assert_close(model(x.detach().cpu())['y'], y.detach().cpu(), rtol=1e-4, atol=1e-5)
