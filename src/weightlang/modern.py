"""Explicit sequence and image operators; all persistent state is float32 weights."""

import math

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_len, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x + self.weight[: x.shape[-2]]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, causal: int = 0, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.causal = bool(causal)
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads).unbind(2)
        out = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal,
            dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).contiguous().reshape(batch, length, dim))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, offset: float = 0.0, cast_after: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), 1.0 - offset))
        self.eps, self.offset, self.cast_after = eps, offset, cast_after

    def forward(self, x):
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        if self.offset:
            return ((self.weight.float() + self.offset) * normalized).to(x.dtype)
        return (self.weight * normalized).to(x.dtype) if self.cast_after else self.weight * normalized.to(x.dtype)


class Scale(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class RopeAttention(nn.Module):
    """Causal GQA with explicit optional padding mask and immutable KV inputs."""
    def __init__(self, dim, heads, kv_heads, head_dim, theta=10000.0, eps=1e-6,
                 qk_norm=0, qkv_bias=0, out_bias=0, window=0, dropout=0.0, clip=0.0, yarn_factor=1.0, yarn_original=2048,
                 yarn_fast=32.0, yarn_slow=1.0, attention_factor=1.0, max_len=32768, flash=1, norm_cast_after=0, rope_fp32=0):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.theta, self.window, self.dropout, self.clip = theta, window, dropout, clip
        self.yarn_factor, self.yarn_original = yarn_factor, yarn_original
        self.yarn_fast, self.yarn_slow = yarn_fast, yarn_slow
        self.attention_factor, self.max_len, self.flash = attention_factor, max_len, flash
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=bool(qkv_bias))
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=bool(qkv_bias))
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=bool(qkv_bias))
        self.o_proj = nn.Linear(heads * head_dim, dim, bias=bool(out_bias))
        self.rope_fp32 = rope_fp32
        self.qk_norm = qk_norm
        self.q_norm = RMSNorm(heads * head_dim if qk_norm == 2 else head_dim, eps, cast_after=norm_cast_after) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(kv_heads * head_dim if qk_norm == 2 else head_dim, eps, cast_after=norm_cast_after) if qk_norm else nn.Identity()

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        batch, length, _ = x.shape
        start = 0 if past_key_value is None else past_key_value[0].shape[1]
        if start + length > self.max_len:
            raise ValueError("sequence exceeds attention max_len")
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.clip:
            q, k, v = (t.clamp(-self.clip, self.clip) for t in (q, k, v))
        if self.qk_norm == 2:
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.reshape(batch, length, self.heads, self.head_dim)
        k = k.reshape(batch, length, self.kv_heads, self.head_dim)
        if self.qk_norm == 1:
            q, k = self.q_norm(q), self.k_norm(k)
        v = v.reshape(batch, length, self.kv_heads, self.head_dim)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim))
        if self.yarn_factor != 1 and self.max_len > self.yarn_original:
            def correction(beta):
                return self.head_dim * math.log(self.yarn_original / (beta * 2 * math.pi)) / (2 * math.log(self.theta))
            low = max(math.floor(correction(self.yarn_fast)), 0)
            high = min(math.ceil(correction(self.yarn_slow)), self.head_dim // 2 - 1)
            ramp = ((torch.arange(self.head_dim // 2, device=x.device).float() - low) / max(high - low, 0.001)).clamp(0, 1)
            inv = inv * (1 - ramp + ramp / self.yarn_factor)
        phases = torch.arange(start, start + length, device=x.device).float()[:, None] * inv[None, :]
        phases = torch.cat((phases, phases), dim=-1)
        cos = (phases.cos() * self.attention_factor).unsqueeze(1)
        sin = (phases.sin() * self.attention_factor).unsqueeze(1)
        if not self.rope_fp32:
            cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        def rotate(t):
            first, second = t.chunk(2, dim=-1)
            return (t * cos + torch.cat((-second, first), dim=-1) * sin).to(t.dtype)
        q, k = rotate(q), rotate(k)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=1)
            v = torch.cat((past_key_value[1], v), dim=1)
        present = (k, v)
        repeats = self.heads // self.kv_heads
        q = q.transpose(1, 2)
        k, v = (t.repeat_interleave(repeats, dim=2).transpose(1, 2) for t in (k, v))
        if self.flash and length > 1 and start == 0 and not self.window and (attention_mask is None or torch.all(attention_mask == 1)):
            values = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            distance = torch.arange(start, start + length, device=x.device)[:, None] - torch.arange(k.shape[2], device=x.device)[None, :]
            allowed = distance >= 0
            if self.window:
                allowed = allowed & (distance < self.window)
            scores = scores.masked_fill(~allowed, float("-inf"))
            if attention_mask is not None:
                if attention_mask.shape != (batch, start + length):
                    raise ValueError("attention_mask must be [batch, past+current time]")
                scores = scores + (1.0 - attention_mask[:, None, None, :].to(scores)) * -1e9
            if getattr(self,'_explicit_mask',False):
                keep = allowed[None,None,:,:] & attention_mask[:,None,None,:]
                scores = scores.masked_fill(~keep,float('-inf'))
                scores = torch.where(keep.any(-1,keepdim=True),scores,torch.zeros_like(scores))
                probs = nn.functional.softmax(scores,dim=-1).masked_fill(~keep,0).to(q.dtype)
            else:
                probs = nn.functional.softmax(scores, dim=-1).to(q.dtype)
            values = nn.functional.dropout(probs, p=self.dropout, training=self.training) @ v
        output = self.o_proj(values.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim))
        return (output, present) if use_cache else output


class SparseMoE(nn.Module):
    """Top-k SwiGLU experts; normalized top-1 uses detached-input straight-through routing."""
    def __init__(self, dim, hidden, experts, top_k, normalize=1, aux_coef=0.0005, activation=0):
        super().__init__()
        self.top_k, self.normalize, self.aux_coef, self.activation = top_k, normalize, aux_coef, activation
        self.gate = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList([nn.ModuleDict({
            "gate_proj": nn.Linear(dim, hidden, bias=False),
            "up_proj": nn.Linear(dim, hidden, bias=False),
            "down_proj": nn.Linear(hidden, dim, bias=False),
        }) for _ in range(experts)])

    def forward(self, x, return_aux=False):
        flat = x.reshape(-1, x.shape[-1])
        scores = self.gate(flat).softmax(-1)
        weights, indices = scores.topk(self.top_k, dim=-1, sorted=False)
        if self.normalize:
            if self.top_k > 1:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            else:
                top1 = self.gate(flat.detach()).softmax(-1).topk(1, dim=-1, sorted=False)[0]
                weights = top1 - top1.detach() + 1.0
        output = torch.zeros_like(flat)
        activation = (nn.functional.silu, nn.functional.relu, nn.functional.gelu)[self.activation]
        for index, expert in enumerate(self.experts):
            mask = indices == index
            if mask.any():
                tokens = mask.any(-1).nonzero().flatten()
                selected = flat[tokens]
                value = expert["down_proj"](activation(expert["gate_proj"](selected)) * expert["up_proj"](selected))
                output.index_add_(0, tokens, (value * weights[mask].view(-1, 1)).to(output.dtype))
            elif self.training:
                output[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        aux = scores.new_zeros((x.shape[0],)) if return_aux == 2 else scores.new_zeros(())
        if self.training and self.aux_coef > 0:
            if return_aux == 2:
                load = nn.functional.one_hot(indices.view(x.shape[0],x.shape[1],self.top_k),len(self.experts)).float().mean(1)
                importance = scores.view(x.shape[0],x.shape[1],len(self.experts)).mean(1)
                aux = (load * importance[:,None,:]).sum((1,2)) * len(self.experts) * self.aux_coef
            else:
                load = nn.functional.one_hot(indices, len(self.experts)).float().mean(0)
                aux = (load * scores.mean(0)).sum() * len(self.experts) * self.aux_coef
        output = output.view_as(x)
        return (output, aux) if return_aux else output


class CachedAttention(RopeAttention):
    def __init__(self,**args):
        super().__init__(**args)
        self._explicit_mask = True

    def forward(self,x,mask,key,value):
        output,(key,value) = super().forward(x,attention_mask=mask,past_key_value=(key,value),use_cache=True)
        return {'output':output,'key':key,'value':value}


class MoEPorts(SparseMoE):
    def forward(self,x):
        output,aux = super().forward(x,return_aux=2)
        return {'output':output,'aux':aux}


class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=-2)


class Last(nn.Module):
    def forward(self, x):
        return x[..., -1, :]


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean(dim=(-2, -1))


class LastDimCrossEntropy(nn.Module):
    def forward(self, prediction, target):
        return nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), target.reshape(-1).long())


class Conv1d(nn.Conv1d):
    def __init__(self, **args):
        self.left_padding = args['dilation'] * (args['kernel']-1) if args['causal'] else 0
        super().__init__(args['in'],args['out'],args['kernel'],stride=args['stride'],
                         padding=args['padding'],dilation=args['dilation'],groups=args['groups'],bias=bool(args['bias']))

    def forward(self, x):
        if self.left_padding:
            x = nn.functional.pad(x,(self.left_padding,0))
        return super().forward(x)


def build_modern(op, args):
    if op in {'rope_attention','sparse_moe'}:
        args = dict(args)
        ports = args.pop('ports',0)
        if ports:
            return (CachedAttention if op == 'rope_attention' else MoEPorts)(**args)
    constructors = {
        "conv1d": lambda: Conv1d(**args),
        "sparse_moe": lambda: SparseMoE(**args),
        "scale": lambda: Scale(**args),
        "rmsnorm": lambda: RMSNorm(**args),
        "rope_attention": lambda: RopeAttention(**args),
        "embedding": lambda: nn.Embedding(args["vocab"], args["dim"], padding_idx=args.get("padding_idx")),
        "position_embedding": lambda: PositionEmbedding(args["dim"], args["max_len"]),
        "self_attention": lambda: SelfAttention(**args),
        "conv2d": lambda: nn.Conv2d(args["in"], args["out"], args["kernel"], args["stride"], args["padding"]),
        "maxpool2d": lambda: nn.MaxPool2d(args["kernel"], args["stride"]),
        "globalavgpool2d": GlobalAvgPool2d,
        "meanpool": MeanPool,
        "last": Last,
        "flatten": lambda: nn.Flatten(start_dim=1),
    }
    return constructors[op]()


def tensor_views(module):
    """Compiler adapter: describe meaningful row windows, not frontend model templates."""
    views = {}
    for name, child in module.named_modules():
        if isinstance(child, SelfAttention):
            dim = child.proj.in_features
            size = dim // child.heads
            records = []
            for part, label in enumerate(("Query", "Key", "Value")):
                records.append({"label": label, "start": part * dim, "end": (part + 1) * dim})
                records.extend(
                    {
                        "label": f"{label} · head {head}",
                        "start": part * dim + head * size,
                        "end": part * dim + (head + 1) * size,
                    }
                    for head in range(child.heads)
                )
            views[f"{name}.qkv.weight"] = records
    return views
