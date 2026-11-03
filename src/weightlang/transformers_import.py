"""Audited architecture adapters and source inventory for Transformers 5.7.0.

No pretrained weights or remote code are loaded. Adapters describe a restricted
single-input, full-sequence base-model interface, not the complete HF Python API.
"""

from __future__ import annotations

import ast
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

from .syntax import WgtError, parse

TESTED_VERSION = "5.7.0"
FAMILIES = {
    "gpt2": "GPT2Model",
    "llama": "LlamaModel",
    "mistral": "MistralModel",
    "qwen2": "Qwen2Model",
    "qwen3": "Qwen3Model",
    "gemma": "GemmaModel",
    "olmo": "OlmoModel",
    "olmo2": "Olmo2Model",
    "olmo3": "Olmo3Model",
    "starcoder2": "Starcoder2Model",
}


def architecture(family: str, config: dict, sequence: int) -> str:
    """Generate base-model hidden states for unpadded tokens at positions 0..T-1."""
    if family not in FAMILIES:
        raise WgtError(f"no verified adapter for {family!r}; supported: {', '.join(FAMILIES)}")
    if not isinstance(config, dict) or type(sequence) is not int or sequence <= 0:
        raise WgtError("expected config object and positive sequence length")
    if config.get("model_type", family) != family:
        raise WgtError("configuration model_type does not match selected family")

    def positive(key, default=None):
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            raise WgtError(f"config {key} must be a positive integer")
        return value

    def flag(key, default=False):
        value = config.get(key, default)
        if type(value) is not bool:
            raise WgtError(f"config {key} must be boolean")
        return int(value)

    def scalar(key, default):
        value = config.get(key, default)
        if type(value) not in (int, float):
            raise WgtError(f"config {key} must be numeric")
        return value

    vocab = positive("vocab_size")
    padding = config.get("pad_token_id")
    if padding is not None and (type(padding) is not int or not 0 <= padding < vocab):
        raise WgtError("pad_token_id must be null or an integer inside the vocabulary")
    dim = positive("n_embd" if family == "gpt2" else "hidden_size")
    depth = positive("n_layer" if family == "gpt2" else "num_hidden_layers")
    heads = positive("n_head" if family == "gpt2" else "num_attention_heads")
    if depth > 1024:
        raise WgtError("model depth exceeds 1024")
    if config.get("pruned_heads") or config.get("add_cross_attention") or config.get("is_encoder_decoder"):
        raise WgtError("pruned heads, cross attention and encoder-decoder models require another adapter")
    lines = [
        "wgt 0.3",
        "use std/hf_decoder",
        f"model {family}_base",
        f"# Audited against Transformers {TESTED_VERSION}: {FAMILIES[family]} base hidden states only.",
        "# 单输入 token 序列，不含 batch；只含结构，无权重值或训练声明。",
        "# 位置从 0 开始，无 padding、外部 mask、KV cache 或语言模型输出头。",
        f"input tokens: i64[time={sequence}]",
        f"output hidden: f32[time={sequence},feature={dim}]",
        f"layer tokens: embedding(vocab={vocab}, dim={dim}"
        + (
            f", padding_idx={config['pad_token_id']}"
            if config.get("pad_token_id") is not None and family != "gpt2"
            else ""
        )
        + ")",
    ]
    if family == "gemma":
        lines.append(f"layer embedding_scale: scale(factor={dim**0.5})")
    if family == "starcoder2":
        lines.append(f"layer embedding_dropout: dropout(p={scalar('embedding_dropout', 0.0)})")
    if family == "gpt2":
        if config.get("activation_function", "gelu_new") not in ("gelu_new", "gelu_pytorch_tanh"):
            raise WgtError("GPT2 adapter requires tanh GELU (gelu_new or gelu_pytorch_tanh)")
        if (
            not flag("scale_attn_weights", True)
            or flag("scale_attn_by_inverse_layer_idx")
            or flag("reorder_and_upcast_attn")
        ):
            raise WgtError("nonstandard GPT2 attention scaling/upcast is unsupported")
        positions = positive("n_positions", 1024)
        if sequence > positions:
            raise WgtError("sequence exceeds position embedding capacity")
        hidden = positive("n_inner", dim * 4) if config.get("n_inner") is not None else dim * 4
        eps = scalar("layer_norm_epsilon", 1e-5)
        ad, rd, ed = (scalar(key, 0.1) for key in ("attn_pdrop", "resid_pdrop", "embd_pdrop"))
        lines += [
            f"layer positions: position_embedding(dim={dim}, max_len={positions})",
            f"layer embedding_dropout: dropout(p={ed})",
        ]
        for i in range(depth):
            lines.append(
                f"layer b{i}: hf_decoder.gpt2_block(dim={dim}, heads={heads}, hidden={hidden}, eps={eps}, attn_dropout={ad}, resid_dropout={rd})"
            )
        lines.append(f"layer norm: layernorm(dim={dim}, eps={eps})")
    else:
        allowed_activation = ("gelu_pytorch_tanh", "gelu_new") if family in ("gemma", "starcoder2") else ("silu",)
        if config.get("hidden_act", allowed_activation[0]) not in allowed_activation:
            raise WgtError(f"{family} requires activation in {allowed_activation}")
        kv = config.get("num_key_value_heads")
        kv = heads if kv is None else kv
        head_dim = config.get("head_dim")
        if head_dim is None and dim % heads:
            raise WgtError("hidden_size must divide heads when head_dim is omitted")
        head_dim = dim // heads if head_dim is None else head_dim
        if type(kv) is not int or type(head_dim) is not int:
            raise WgtError("invalid key/value heads or head dimension")
        hidden = positive("intermediate_size")
        eps = scalar("norm_epsilon", 1e-5) if family == "starcoder2" else scalar("rms_norm_eps", 1e-5 if family == "olmo3" else 1e-6)
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        if not isinstance(rope, dict) or rope.get("rope_type", rope.get("type", "default")) != "default":
            raise WgtError("only default full-dimension RoPE is supported")
        if (
            config.get("partial_rotary_factor", 1.0) != 1.0
            or rope.get("partial_rotary_factor", 1.0) != 1.0
            or set(rope) - {"rope_type", "type", "rope_theta", "partial_rotary_factor"}
        ):
            raise WgtError("scaled or partial RoPE is unsupported")
        theta = rope.get("rope_theta", config.get("rope_theta", 1e6 if family.startswith("qwen") else 10000))
        if type(theta) not in (int, float):
            raise WgtError("rope_theta must be numeric")
        ad = scalar("attention_dropout", 0.0)
        bias = flag("use_bias", True) if family == "starcoder2" else flag("attention_bias")
        qkv_bias, out_bias = (1, 0) if family == "qwen2" else (bias, bias)
        if family == "mistral" and bias:
            raise WgtError("Mistral has no attention bias")
        mlp_bias = flag("mlp_bias") if family == "llama" else 0
        window = config.get("sliding_window", 4096 if family == "olmo3" else None)
        layer_types = config.get("layer_types")
        if layer_types is None:
            if family == "olmo3":
                layer_types = ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(depth)]
            elif family in ("mistral", "starcoder2"):
                layer_types = ["sliding_attention" if window else "full_attention"] * depth
            elif family in ("qwen2", "qwen3") and config.get("use_sliding_window", False):
                layer_types = [
                    "sliding_attention" if i >= config.get("max_window_layers", 28) else "full_attention"
                    for i in range(depth)
                ]
            else:
                layer_types = ["full_attention"] * depth
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != depth
            or any(t not in ("full_attention", "sliding_attention") for t in layer_types)
        ):
            raise WgtError("unsupported decoder layer_types")
        if family in ("gemma", "olmo", "olmo2") and any(t != "full_attention" for t in layer_types):
            raise WgtError(f"{family} adapter only supports full_attention layers")
        for i, kind in enumerate(layer_types):
            sliding = window if kind == "sliding_attention" else 0
            if type(sliding) is not int or sliding < 0 or (kind == "sliding_attention" and sliding == 0):
                raise WgtError("sliding attention requires a positive integer window")
            if family == "gemma":
                lines.append(
                    f"layer b{i}: hf_decoder.gemma_block(dim={dim}, heads={heads}, kv_heads={kv}, head_dim={head_dim}, hidden={hidden}, eps={eps}, theta={theta}, attention_bias={bias}, attn_dropout={ad})"
                )
            elif family in ("olmo2", "olmo3"):
                lines.append(
                    f"layer b{i}: hf_decoder.olmo2_block(dim={dim}, heads={heads}, kv_heads={kv}, head_dim={head_dim}, hidden={hidden}, eps={eps}, theta={theta}, attention_bias={bias}, attn_dropout={ad}, window={sliding})"
                )
            elif family == "olmo":
                clip = config.get("clip_qkv") or 0
                if type(clip) not in (int, float):
                    raise WgtError("clip_qkv must be numeric or null")
                lines.append(
                    f"layer b{i}: hf_decoder.olmo_block(dim={dim}, heads={heads}, kv_heads={kv}, head_dim={head_dim}, hidden={hidden}, theta={theta}, attention_bias={bias}, attn_dropout={ad}, clip={clip})"
                )
            elif family == "starcoder2":
                rd = scalar("residual_dropout", 0.0)
                lines.append(
                    f"layer b{i}: hf_decoder.starcoder2_block(dim={dim}, heads={heads}, kv_heads={kv}, head_dim={head_dim}, hidden={hidden}, eps={eps}, theta={theta}, bias={bias}, window={sliding}, attn_dropout={ad}, resid_dropout={rd})"
                )
            else:
                lines.append(
                    f"layer b{i}: hf_decoder.rope_block(dim={dim}, heads={heads}, kv_heads={kv}, head_dim={head_dim}, hidden={hidden}, eps={eps}, theta={theta}, qk_norm={int(family == 'qwen3')}, qkv_bias={qkv_bias}, out_bias={out_bias}, mlp_bias={mlp_bias}, window={sliding}, attn_dropout={ad})"
                )
        if family == "olmo":
            lines.append(f"layer norm: layernorm(dim={dim}, affine=0)")
        elif family == "starcoder2":
            lines.append(f"layer norm: layernorm(dim={dim}, eps={eps})")
        else:
            lines.append(f"layer norm: rmsnorm(dim={dim}, eps={eps}, offset={int(family == 'gemma')})")
    lines.append("init seed=0")
    text = "\n".join(lines) + "\n"
    parse(text)
    return text


def parameter_mapping(family, config):
    """WGT key -> (upstream key, transpose); for tests/explicit parameter migration."""
    mapping = {"tokens.weight": ("wte.weight" if family == "gpt2" else "embed_tokens.weight", False)}
    if family == "gpt2":
        mapping["positions.weight"] = ("wpe.weight", False)
        for i in range(config["n_layer"]):
            names = {
                "norm1": "ln_1",
                "norm2": "ln_2",
                "attention.qkv": "attn.c_attn",
                "attention.proj": "attn.c_proj",
                "up": "mlp.c_fc",
                "down": "mlp.c_proj",
            }
            for dst, src in names.items():
                for suffix in ("weight", "bias"):
                    mapping[f"b{i}.{dst}.{suffix}"] = (
                        f"h.{i}.{src}.{suffix}",
                        suffix == "weight" and src not in ("ln_1", "ln_2"),
                    )
        for suffix in ("weight", "bias"):
            mapping[f"norm.{suffix}"] = (f"ln_f.{suffix}", False)
    else:
        for i in range(config["num_hidden_layers"]):
            names = {
                "norm1": "input_layernorm",
                "norm2": "post_attention_layernorm",
                "gate.step0": "mlp.gate_proj",
                "value": "mlp.up_proj",
                "down": "mlp.down_proj",
            }
            if family in ("olmo2", "olmo3"):
                names.update(norm1="post_attention_layernorm", norm2="post_feedforward_layernorm")
            if family == "olmo":
                names = {k: v for k, v in names.items() if not k.startswith("norm")}
            if family == "starcoder2":
                names = {
                    "norm1": "input_layernorm",
                    "norm2": "post_attention_layernorm",
                    "up": "mlp.c_fc",
                    "down": "mlp.c_proj",
                }
            for dst, src in names.items():
                mapping[f"b{i}.{dst}.weight"] = (f"layers.{i}.{src}.weight", False)
                if (family == "llama" and config.get("mlp_bias", False) and src.startswith("mlp.")) or (
                    family == "starcoder2" and (src.endswith("layernorm") or config.get("use_bias", True))
                ):
                    mapping[f"b{i}.{dst}.bias"] = (f"layers.{i}.{src}.bias", False)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mapping[f"b{i}.attention.{proj}.weight"] = (f"layers.{i}.self_attn.{proj}.weight", False)
                if (
                    config.get("attention_bias", False)
                    or (family == "qwen2" and proj != "o_proj")
                    or (family == "starcoder2" and config.get("use_bias", True))
                ):
                    mapping[f"b{i}.attention.{proj}.bias"] = (f"layers.{i}.self_attn.{proj}.bias", False)
            if family in ("qwen3", "olmo2", "olmo3"):
                for name in ("q_norm", "k_norm"):
                    mapping[f"b{i}.attention.{name}.weight"] = (f"layers.{i}.self_attn.{name}.weight", False)
        if family != "olmo":
            mapping["norm.weight"] = ("norm.weight", False)
        if family == "starcoder2":
            mapping["norm.bias"] = ("norm.bias", False)
    return mapping


def inventory(root: Path | None = None) -> dict:
    """Inspect every modeling source without importing or allocating its model."""
    dist = distribution("transformers")
    root = root or Path(dist.locate_file("transformers/models"))
    records = []
    for path in sorted(root.rglob("modeling_*.py")):
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        calls = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
        family = path.relative_to(root).parts[0]
        from .python_import import python_architecture

        try:
            python_architecture(path)
            attempt = {"status": "converted", "note": "Static conversion only; numerical equivalence not validated."}
        except (WgtError, SyntaxError, ValueError, TypeError, RecursionError) as exc:
            attempt = {"status": "unsupported", "reason": str(exc).replace(str(root) + "/", "")}
        records.append(
            {
                "source": str(path.relative_to(root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "classes": classes,
                "status": "not_migrated",
                "static_attempt": attempt,
                "adapter_class": FAMILIES.get(family),
                "signals": {
                    key: any(key in call.lower() for call in calls)
                    for key in ("cache", "mask", "conv", "attention", "rmsnorm", "rotary", "expert", "scan")
                },
            }
        )
    return {
        "transformers_version": dist.version,
        "expected_version": TESTED_VERSION,
        "scope": "All installed modeling_*.py files; adapter availability is NOT migration completion.",
        "source_files": len(records),
        "records": records,
    }


def write_architecture(family, config_path, sequence, output):
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise WgtError(f"output already exists: {destination}")
    config = json.loads(Path(config_path).read_text())
    text = architecture(family, config, sequence)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return {
        "output": str(destination),
        "class": FAMILIES[family],
        "trained_weights": False,
        "interface": "unpadded full-sequence base hidden states, no KV cache or LM head",
    }
