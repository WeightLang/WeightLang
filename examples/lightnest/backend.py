"""WGT parameter kernels with explicit, differentiable dynamic host operations.

This is a hybrid backend, not an assertion that arbitrary Python compiles to WGT.
All emitted sources use the existing local 0.3 implementation.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from weightlang import compile_file, load_artifact


class Kernels:
    def __init__(self, root, *, reload=False, source_root=None):
        self.root = Path(root)
        self.reload = reload
        self.source_root = Path(source_root) if source_root is not None else None
        self.records = []
        self.artifacts = []
        if not reload:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "sources").mkdir()
        self.saved_records = json.loads((self.root / "mapping.json").read_text()) if reload else None

    def make(self, name, operation, input_shape, output_shape, tensors, origins, *, integer=False):
        identifier = name.replace('.', '__')
        if any(r['name'] == name for r in self.records):
            raise ValueError(f'duplicate kernel {name}')
        source = (f'wgt 0.3\nmodel {identifier}\n'
                  '# 单样本形状不含 batch；动态宿主把独立位置折叠到 batch 后调用。\n'
                  '# 参数来自本次架构初始化，不是训练权重；完整动态计算见 backend.py。\n'
                  f'input x: {"i64" if integer else "f32"}[{input_shape}]\n'
                  f'output y: f32[{output_shape}]\n'
                  f'layer op: {operation}\ninit seed=42\n')
        path = self.root / 'components' / identifier
        if self.reload:
            artifact = load_artifact(path, warn_untrained=False)
            source = (self.root / 'sources' / f'{identifier}.wgt').read_text()
            if (path / 'source.wgt').read_text() != source:
                raise ValueError(f'architecture/source mismatch: {name}')
        else:
            if self.source_root is not None:
                source = (self.source_root / f'{identifier}.wgt').read_text()
            source_path = self.root / 'sources' / f'{identifier}.wgt'
            source_path.write_text(source)
            built = compile_file(source_path, self.root / '.scaffold', no_train=True)
            artifact = load_artifact(built.path, warn_untrained=False)
            expected = artifact.module.state_dict()
            if set(expected) != set(tensors):
                raise ValueError(f'parameter mismatch: {name}')
            for key, tensor in tensors.items():
                if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
                    raise ValueError(f'layout/dtype mismatch: {name}.{key}')
            artifact.module.load_state_dict(tensors, strict=True)
            artifact.save(path)
            artifact = load_artifact(path, warn_untrained=False)
            for key, tensor in tensors.items():
                torch.testing.assert_close(artifact.module.state_dict()[key], tensor, rtol=0, atol=0)
            shutil.rmtree(built.path)
        self.artifacts.append(artifact)
        self.records.append({'name': name, 'artifact': str(path.relative_to(self.root)),
                             'mapping': origins, 'operation': operation})
        return artifact.module

    def linear(self, name, weight, bias, *, weight_name=None, bias_name=None):
        width_out, width_in = weight.shape
        value = bias if bias is not None else weight.new_zeros(width_out)
        module = self.make(name, f'linear(in={width_in}, out={width_out})', str(width_in), str(width_out),
                           {'op.weight': weight, 'op.bias': value},
                           {'op.weight': weight_name or name + '.weight',
                            'op.bias': (bias_name or name + '.bias') if bias is not None else 'constant zero; frozen'})
        if bias is None:
            module.op.bias.requires_grad_(False)
        return Linear(module, width_in, width_out)

    def finish(self):
        if self.reload:
            if self.records != self.saved_records:
                raise ValueError('kernel mapping mismatch')
        else:
            (self.root / 'mapping.json').write_text(json.dumps(self.records, indent=2) + '\n')
            (self.root / '.scaffold').rmdir()


class Linear(nn.Module):
    def __init__(self, graph, width_in, width_out):
        super().__init__()
        self.graph, self.in_features, self.out_features = graph, width_in, width_out

    @property
    def weight(self):
        return self.graph.op.weight

    @property
    def bias(self):
        return self.graph.op.bias

    def forward(self, x):
        return self.graph(x.reshape(-1, self.in_features)).reshape(*x.shape[:-1], self.out_features)


class Embedding(nn.Module):
    def __init__(self, graph, original):
        super().__init__()
        self.graph = graph
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        if self.padding_idx is not None:
            # WGT's embedding has no padding option. Keep the same frozen row
            # gradient without changing its forward value or artifact structure.
            self.weight.register_hook(self._mask_padding)

    @property
    def weight(self):
        return self.graph.op.weight

    def _mask_padding(self, grad):
        grad = grad.clone()
        grad[self.padding_idx] = 0
        return grad

    def forward(self, x):
        return self.graph(x.reshape(-1, 1)).reshape(*x.shape, self.embedding_dim)


class Norm(nn.Module):
    def __init__(self, graph, width):
        super().__init__()
        self.graph, self.width = graph, width

    def forward(self, x):
        return self.graph(x.reshape(-1, self.width)).reshape_as(x)


class Conv1d(nn.Module):
    def __init__(self, projection, original):
        super().__init__()
        self.projection = projection
        self.kernel = original.kernel_size[0]
        self.padding = original.padding[0]
        self.stride = original.stride[0]

    def forward(self, x):
        # PyTorch Conv1d is cross-correlation, ordered [channel, kernel].
        patches = F.pad(x, (self.padding, self.padding)).unfold(-1, self.kernel, self.stride)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        return self.projection(patches).transpose(1, 2)


class GRUCell(nn.Module):
    def __init__(self, input_projection, hidden_projection, hidden_size):
        super().__init__()
        self.input_projection, self.hidden_projection = input_projection, hidden_projection
        self.hidden_size = hidden_size

    def forward(self, x, h=None):
        if h is None:
            h = x.new_zeros(*x.shape[:-1], self.hidden_size)
        ir, iz, inn = self.input_projection(x).chunk(3, -1)
        hr, hz, hn = self.hidden_projection(h).chunk(3, -1)
        reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
        # PyTorch reset-after: r * (W_hn h + b_hn), including the bias.
        candidate = (inn + reset * hn).tanh()
        return candidate + update * (h - candidate)


class GRU(nn.Module):
    def __init__(self, cells, original):
        super().__init__()
        self.cells = nn.ModuleList(cells)
        self.hidden_size, self.batch_first = original.hidden_size, original.batch_first
        self.num_layers, self.dropout = original.num_layers, original.dropout

    def forward(self, x, h=None):
        packed = isinstance(x, PackedSequence)
        lengths = None
        if packed:
            x, lengths = pad_packed_sequence(x, batch_first=True)
        elif not self.batch_first:
            x = x.transpose(0, 1)
        if x.ndim != 3:
            raise ValueError('LightNest GRU requires batched sequence inputs')
        states = []
        for index, cell in enumerate(self.cells):
            state = x.new_zeros(x.shape[0], self.hidden_size) if h is None else h[index]
            outputs = []
            for time in range(x.shape[1]):
                candidate = cell(x[:, time], state)
                if lengths is not None:
                    valid = (time < lengths.to(x.device))[:, None]
                    state = torch.where(valid, candidate, state)
                    outputs.append(torch.where(valid, state, torch.zeros_like(state)))
                else:
                    state = candidate
                    outputs.append(state)
            x = torch.stack(outputs, 1)
            states.append(state)
            if index + 1 < self.num_layers:
                x = F.dropout(x, self.dropout, self.training)
        if packed:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        elif not self.batch_first:
            x = x.transpose(0, 1)
        return x, torch.stack(states)


class Attention(nn.Module):
    def __init__(self, q, k, v, output, original):
        super().__init__()
        self.q, self.k, self.v, self.output = q, k, v, output
        self.num_heads, self.batch_first = original.num_heads, original.batch_first
        self.dropout = original.dropout

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        batch, length, width = query.shape
        def heads(x):
            return x.reshape(batch, -1, self.num_heads, width // self.num_heads).transpose(1, 2)
        q, k, v = heads(self.q(query)), heads(self.k(key)), heads(self.v(value))
        scores = q @ k.transpose(-1, -2) / (width // self.num_heads) ** .5
        if is_causal:
            causal = torch.ones(length, key.shape[1], device=query.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, -torch.inf)
        for mask, padding in ((attn_mask, False), (key_padding_mask, True)):
            if mask is None:
                continue
            if padding:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask.reshape(batch, self.num_heads, length, -1)
            scores = scores.masked_fill(mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
        weights = F.dropout(scores.softmax(-1), self.dropout, self.training)
        result = self.output((weights @ v).transpose(1, 2).reshape(batch, length, width))
        if not self.batch_first:
            result = result.transpose(0, 1)
        weights_out = (weights.mean(1) if average_attn_weights else weights) if need_weights else None
        return result, weights_out


class EncoderLayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name, child in original.named_children():
            self.add_module(name, child)
        self.norm_first, self.activation = original.norm_first, original.activation

    def forward(self, x, mask=None, padding=None, is_causal=False):
        def attention(value):
            return self.dropout1(self.self_attn(value, value, value, attn_mask=mask,
                                                key_padding_mask=padding, need_weights=False,
                                                is_causal=is_causal)[0])
        def feed(value):
            return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(value)))))
        if self.norm_first:
            x = x + attention(self.norm1(x))
            return x + feed(self.norm2(x))
        x = self.norm1(x + attention(x))
        return self.norm2(x + feed(x))


class Encoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.layers, self.norm = original.layers, original.norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        for layer in self.layers:
            src = layer(src, mask, src_key_padding_mask, bool(is_causal))
        return self.norm(src) if self.norm is not None else src


def migrate(original, kernels):
    """Copy the host graph, replacing supported learned operations; fail closed."""
    model = copy.deepcopy(original)
    covered = set()

    def visit(module, name):
        def claim():
            covered.update(f'{name}.{key}'.lstrip('.') for key in module.state_dict())
        def cell(suffix='', label=''):
            wi, wh = getattr(module, 'weight_ih' + suffix), getattr(module, 'weight_hh' + suffix)
            bi, bh = getattr(module, 'bias_ih' + suffix, None), getattr(module, 'bias_hh' + suffix, None)
            prefix = name + label
            a = kernels.linear(prefix + '.input', wi, bi, weight_name=name + '.weight_ih' + suffix,
                               bias_name=name + '.bias_ih' + suffix)
            b = kernels.linear(prefix + '.hidden', wh, bh, weight_name=name + '.weight_hh' + suffix,
                               bias_name=name + '.bias_hh' + suffix)
            return GRUCell(a, b, module.hidden_size)
        if isinstance(module, nn.Linear):
            claim()
            return kernels.linear(name, module.weight, module.bias)
        if isinstance(module, nn.Embedding):
            if module.max_norm is not None or module.scale_grad_by_freq or module.sparse:
                raise ValueError('unsupported embedding option')
            claim()
            graph = kernels.make(name, f'embedding(vocab={module.num_embeddings}, dim={module.embedding_dim})',
                                 '1', f'1,{module.embedding_dim}', {'op.weight': module.weight},
                                 {'op.weight': name + '.weight'}, integer=True)
            return Embedding(graph, module)
        if isinstance(module, nn.LayerNorm):
            if len(module.normalized_shape) != 1 or module.eps != 1e-5 or module.bias is None:
                raise ValueError('unsupported LayerNorm contract')
            claim()
            width = module.normalized_shape[0]
            graph = kernels.make(name, f'layernorm(dim={width})', str(width), str(width),
                                 {'op.weight': module.weight, 'op.bias': module.bias},
                                 {'op.weight': name + '.weight', 'op.bias': name + '.bias'})
            return Norm(graph, width)
        if isinstance(module, nn.Conv1d):
            if module.groups != 1 or module.dilation != (1,) or module.padding_mode != 'zeros':
                raise ValueError('unsupported Conv1d contract')
            claim()
            projection = kernels.linear(name, module.weight.flatten(1), module.bias,
                                         weight_name=name + '.weight [out,in,kernel] -> [out,in*kernel]')
            return Conv1d(projection, module)
        if isinstance(module, nn.GRUCell):
            claim()
            return cell()
        if isinstance(module, nn.GRU):
            if module.bidirectional:
                raise ValueError('bidirectional GRU is outside current LightNest')
            claim()
            return GRU([cell(f'_l{i}', f'.layer{i}') for i in range(module.num_layers)], module)
        if isinstance(module, nn.MultiheadAttention):
            if not module._qkv_same_embed_dim or module.bias_k is not None or module.add_zero_attn:
                raise ValueError('unsupported MHA contract')
            claim()
            weights = module.in_proj_weight.chunk(3)
            biases = module.in_proj_bias.chunk(3) if module.in_proj_bias is not None else [None] * 3
            parts = [kernels.linear(name + '.' + tag, w, b,
                                   weight_name=f'{name}.in_proj_weight [{i}*dim:({i}+1)*dim]',
                                   bias_name=f'{name}.in_proj_bias [{i}*dim:({i}+1)*dim]')
                     for i, (tag, w, b) in enumerate(zip(('q', 'k', 'v'), weights, biases))]
            return Attention(*parts, visit(module.out_proj, name + '.out_proj'), module)
        for child_name, child in list(module.named_children()):
            setattr(module, child_name, visit(child, f'{name}.{child_name}'.lstrip('.')))
        # These tensors are data for dynamic host attention, not executable
        # stand-alone networks. They are explicitly labelled in the manifest.
        for key, parameter in list(module.named_parameters(recurse=False)):
            full = f'{name}.{key}'.lstrip('.')
            if full not in {'green.memory.landmark_bias', 'notebook.slot_keys', 'landmark_bias', 'slot_keys'}:
                raise ValueError(f'unhandled parameter: {full}')
            shape = parameter.shape
            table = parameter.reshape(-1, shape[-1])
            graph = kernels.make(full, f'embedding(vocab={table.shape[0]}, dim={table.shape[1]})',
                                 '1', f'1,{table.shape[1]}', {'op.weight': table},
                                 {'op.weight': full + ' reshape; host attention parameter'}, integer=True)
            setattr(module, key, nn.Parameter(graph.op.weight.detach().reshape(shape)))
            covered.add(full)
        if isinstance(module, nn.TransformerEncoderLayer):
            return EncoderLayer(module)
        if isinstance(module, nn.TransformerEncoder):
            return Encoder(module)
        return module

    result = visit(model, '')
    # Persistent non-parameter buffers (e.g. ELF magic) stay in the host snapshot.
    required = set(dict(original.named_parameters()))
    if required - covered:
        raise ValueError(f'unmapped parameters: {sorted(required - covered)}')
    kernels.finish()
    result._wgt_kernels = kernels
    result.train(original.training)
    return result
