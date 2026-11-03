"""Explicit sequence invocation; no persistent per-call mask, cache, or auxiliary loss."""
from dataclasses import dataclass

import torch
from torch import nn

from .dataflow import Flow, Merge, Recur, Scan, ResidualFlow
from .artifact import Residual
from .modern import RopeAttention, SparseMoE


@dataclass
class SequenceOutput:
    output: torch.Tensor
    past_key_values: tuple
    aux_loss: torch.Tensor
    intermediates: dict


def run_sequence(module, input_ids, *, attention_mask=None, past_key_values=None, use_cache=False, capture=()):
    """Execute a sequential/branched decoder with explicit [B,T] tokens/mask.

    Cache entries follow depth-first attention order and contain [B,T,KV,D] K/V.
    A call owns its outputs; nothing is retained on modules. Fixed source shapes
    remain the artifact contract; this API permits variable sequence lengths for
    graphs containing only sequence-polymorphic operators.
    """
    if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or input_ids.shape[1] == 0:
        raise ValueError("sequence input must be nonempty i64[batch,time]")
    devices = {p.device for p in module.parameters()}
    if len(devices) != 1 or input_ids.device not in devices:
        raise ValueError("run_sequence requires tokens and model parameters on one device")
    if attention_mask is not None and attention_mask.device != input_ids.device:
        raise ValueError("attention_mask must be on the token device")
    attentions = [m for m in module.modules() if isinstance(m, RopeAttention)]
    cache = tuple(past_key_values) if past_key_values is not None else (None,) * len(attentions)
    if len(cache) != len(attentions):
        raise ValueError("cache must contain one entry per attention layer")
    lengths = set()
    for attn, entry in zip(attentions, cache):
        if entry is None:
            lengths.add(0)
            continue
        if len(entry) != 2 or entry[0].shape != entry[1].shape or entry[0].ndim != 4:
            raise ValueError("cache must contain equal-shaped [B,T,KV,D] key/value pairs")
        if entry[0].shape[0] != input_ids.shape[0] or entry[0].shape[2:] != (attn.kv_heads, attn.head_dim):
            raise ValueError("cache batch/head dimensions do not match attention")
        if any(t.device != input_ids.device for t in entry):
            raise ValueError("cache tensors must be on the token device")
        lengths.add(entry[0].shape[1])
    if len(lengths) > 1:
        raise ValueError("cache lengths must agree across layers")
    start = next(iter(lengths), 0)
    if attention_mask is not None:
        if attention_mask.shape != (input_ids.shape[0], start + input_ids.shape[1]):
            raise ValueError("attention_mask must cover past and current tokens")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one")
    presents, auxiliary, intermediates = [], [], {}
    cursor = 0

    def execute(node, x, path):
        nonlocal cursor
        if isinstance(node, (Recur, Scan)):
            raise ValueError("sequence invocation does not support recurrent/scan bodies")
        if isinstance(node, RopeAttention):
            past = cache[cursor]
            cursor += 1
            result = node(x, attention_mask=attention_mask, past_key_value=past, use_cache=use_cache)
            if use_cache:
                result, present = result
                presents.append(present)
            return result
        if isinstance(node, SparseMoE):
            result, aux = node(x, return_aux=True)
            auxiliary.append(aux)
            return result
        if isinstance(node, nn.Sequential):
            residual = x
            values = {"input": x}
            routes = {name: refs for name, refs, _ in node._flow_routes} if isinstance(node, Flow) else {}
            for name, child in node._modules.items():
                refs = routes.get(name, [])
                args = [values[ref] for ref in refs] if refs else [x]
                child_path = f"{path}.{name}" if path else name
                x = execute(child, args if isinstance(child, Merge) else args[0], child_path)
                values[name] = x
                if child_path in capture:
                    intermediates[child_path] = x
            return residual + x if isinstance(node, (Residual, ResidualFlow)) else x
        return node(x)

    output = execute(module, input_ids, "")
    aux = sum(auxiliary, output.new_zeros(()))
    return SequenceOutput(output, tuple(presents), aux, intermediates)
