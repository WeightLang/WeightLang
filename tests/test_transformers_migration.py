"""Actual Transformers base-model comparisons; no pretrained downloads."""

import importlib.metadata

import pytest
import torch

from weightlang.artifact import load_artifact
from weightlang.compiler import compile_file
from weightlang.syntax import WgtError, parse
from weightlang.transformers_import import architecture, parameter_mapping, inventory


@pytest.fixture(scope="module")
def hf():
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("optional Transformers dependency not installed")
    if version != "5.7.0":
        pytest.skip("migration reference is pinned to Transformers 5.7.0")
    return pytest.importorskip("transformers", exc_type=ImportError)


def make_reference(hf, family, window=False, bias=False):
    if family == "gpt2":
        config = hf.GPT2Config(
            vocab_size=31,
            n_positions=16,
            n_embd=16,
            n_layer=2,
            n_head=4,
            n_inner=24,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            layer_norm_epsilon=0.0002,
        )
        cls = hf.GPT2Model
    else:
        names = {
            "llama": ("LlamaConfig", "LlamaModel"),
            "mistral": ("MistralConfig", "MistralModel"),
            "qwen2": ("Qwen2Config", "Qwen2Model"),
            "qwen3": ("Qwen3Config", "Qwen3Model"),
            "gemma": ("GemmaConfig", "GemmaModel"),
            "olmo": ("OlmoConfig", "OlmoModel"),
            "olmo2": ("Olmo2Config", "Olmo2Model"),
            "olmo3": ("Olmo3Config", "Olmo3Model"),
            "starcoder2": ("Starcoder2Config", "Starcoder2Model"),
        }
        config_cls, model_cls = names[family]
        kwargs = dict(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            max_position_embeddings=32,
            rms_norm_eps=0.0002,
            attention_dropout=0.0,
            rope_parameters={"rope_type": "default", "rope_theta": 12345.0},
        )
        if family in ("llama", "qwen3", "gemma", "olmo", "olmo2", "olmo3"):
            kwargs["attention_bias"] = bias
        if family == "llama":
            kwargs["mlp_bias"] = bias
        if family in ("qwen2", "qwen3"):
            kwargs.update(use_sliding_window=window, sliding_window=3, max_window_layers=0)
        if family == "mistral":
            kwargs["sliding_window"] = 3 if window else None
        if family == "starcoder2":
            kwargs.pop("head_dim")
            kwargs.update(use_bias=bias, sliding_window=3 if window else None)
        if family == "olmo3":
            kwargs.update(sliding_window=3, num_hidden_layers=4 if window else 2,
                          layer_types=None if window else ["full_attention", "full_attention"])
        if family == "olmo":
            kwargs.update(clip_qkv=0.5, pad_token_id=0)
        config = getattr(hf, config_cls)(**kwargs)
        cls = getattr(hf, model_cls)
    config._attn_implementation = "eager"
    return config, cls(config).eval()


@pytest.mark.parametrize(
    "family,window,bias",
    [
        ("gpt2", False, False),
        ("llama", False, False),
        ("llama", False, True),
        ("mistral", False, False),
        ("mistral", True, False),
        ("qwen2", False, False),
        ("qwen2", True, False),
        ("qwen3", False, False),
        ("qwen3", True, True),
        ("gemma", False, False),
        ("gemma", False, True),
        ("olmo", False, True),
        ("olmo2", False, False),
        ("olmo2", False, True),
        ("olmo3", False, False),
        ("olmo3", True, True),
        ("starcoder2", False, False),
        ("starcoder2", True, True),
    ],
)
def test_full_base_model_values_gradients_and_reload(hf, family, window, bias, tmp_path):
    config, original = make_reference(hf, family, window, bias)
    config_dict = config.to_dict()
    if family == "olmo3" and window:
        # Exercise the adapter default schedule against the reference config.
        config_dict.pop("layer_types")
    text = architecture(family, config_dict, 7)
    source = tmp_path / "model.wgt"
    source.write_text(text)
    built = compile_file(source, tmp_path / "weights", no_train=True)
    target = load_artifact(built.path, warn_untrained=False).module.eval()
    mapping = parameter_mapping(family, config_dict)
    assert set(mapping) == set(target.state_dict())
    assert {name for name, _ in mapping.values()} == set(original.state_dict())
    # Use the persisted WGT weights in both models: tests artifact writing/reloading,
    # all parameter names and GPT2 Conv1D transpose, without modifying saved files.
    state = original.state_dict()
    for dst, (src, transpose) in mapping.items():
        value = target.state_dict()[dst]
        state[src] = value.t().contiguous() if transpose else value
    original.load_state_dict(state, strict=True)
    for batch in (1, 3):
        ids = torch.randint(0, config.vocab_size, (batch, 7))
        original.zero_grad()
        target.zero_grad()
        expected = original(ids, use_cache=False).last_hidden_state
        actual = target(ids)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        coefficients = torch.randn_like(actual)
        (actual * coefficients).sum().backward()
        (expected * coefficients).sum().backward()
        original_params = dict(original.named_parameters())
        for name, param in target.named_parameters():
            src, transpose = mapping[name]
            expected_grad = original_params[src].grad
            if transpose:
                expected_grad = expected_grad.t()
            torch.testing.assert_close(param.grad, expected_grad, rtol=3e-4, atol=3e-5)
    assert "import " not in text and built.manifest["training"] is None


@pytest.mark.parametrize(
    "body,match",
    [
        ("linear(in=4,out=4,bias=2)", "0 or 1"),
        ("rmsnorm(dim=4,eps=0)", "positive"),
        ("rope_attention(dim=4,heads=3,kv_heads=2,head_dim=4)", "divisible"),
        ("rope_attention(dim=4,heads=4,kv_heads=2,head_dim=3)", "even"),
        ("gelu(approximate=2)", "0 or 1"),
        ("rope_attention(dim=4,heads=2,kv_heads=1,head_dim=2,qk_norm=3)", "0, 1 or 2"),
    ],
)
def test_new_operator_invalid_options(body, match):
    text = f"wgt 0.3\nmodel bad\ninput x: f32[2,4]\noutput y: f32[2,4]\nlayer first: linear(in=4,out=4)\nlayer body: {body}\ninit seed=0\n"
    with pytest.raises(WgtError, match=match):
        parse(text)


def test_reject_unimplemented_architecture_semantics():
    config = dict(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    with pytest.raises(WgtError, match="default full-dimension"):
        architecture("llama", {**config, "rope_scaling": {"type": "linear", "factor": 2}}, 7)
    with pytest.raises(WgtError, match="another adapter"):
        architecture("llama", {**config, "add_cross_attention": True}, 7)
    with pytest.raises(WgtError, match="no verified adapter"):
        architecture("qwen3_5", config, 7)


def test_inventory_is_complete_and_does_not_claim_migration(hf):
    report = inventory()
    assert report["source_files"] == len(report["records"]) > 100
    assert all(len(row["sha256"]) == 64 and row["status"] == "not_migrated" for row in report["records"])


@pytest.mark.parametrize("family", ["gpt2", "llama", "mistral", "qwen2", "qwen3", "gemma", "olmo", "olmo2", "olmo3", "starcoder2"])
def test_standalone_python_export_of_migrated_models(family, tmp_path):
    import importlib.util
    from pathlib import Path
    from weightlang.python_export import export_python
    from weightlang.artifact import build_module

    source = Path("examples/transformers") / f"{family}.wgt"
    program = parse(source.read_text())
    original = build_module([layer.to_dict() for layer in program.layers]).eval()
    output = tmp_path / "model.py"
    export_python(source, output)
    spec = importlib.util.spec_from_file_location("exported_model", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Our generated standalone source, not user input.
    restored = module.Model().eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    x = torch.randint(0, 32, (2, 8))
    torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)


def test_invalid_serialized_options_are_rejected(tmp_path):
    import json
    source = tmp_path / "source.wgt"
    source.write_text("wgt 0.3\nmodel test\ninput x: f32[4]\noutput y: f32[4]\nlayer fc: linear(in=4,out=4)\ninit seed=0\n")
    built = compile_file(source, tmp_path/"weights", no_train=True)
    manifest = built.manifest
    manifest["layers"][0]["args"]["bias"] = 2
    (built.path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(WgtError,match="0 or 1"):
        load_artifact(built.path,warn_untrained=False)


def test_olmo2_rejects_unsupported_sliding_layers():
    config = dict(vocab_size=32, hidden_size=16, intermediate_size=24,
                  num_hidden_layers=1, num_attention_heads=4,
                  layer_types=["sliding_attention"], sliding_window=3)
    with pytest.raises(WgtError, match="only supports full_attention"):
        architecture("olmo2", config, 8)
