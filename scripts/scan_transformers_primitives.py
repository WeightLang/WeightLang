"""Read-only AST survey of installed model sources; never import model code.

Signals are syntactic evidence, not dependency resolution or migration coverage.
A file is counted once per category, even when it defines many classes/calls.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

# Attribute/name spellings are deliberately retained as evidence. A method named
# `update` alone does not prove KV-cache use; cache is detected via forward ports.
CALL_GROUPS = {
    'reshape_axes': {'view', 'reshape', 'flatten', 'unflatten', 'transpose', 'permute', 'unsqueeze', 'squeeze'},
    'broadcast': {'expand', 'expand_as', 'broadcast_to', 'repeat', 'repeat_interleave'},
    'split_concat': {'split', 'chunk', 'unbind', 'cat', 'concat', 'stack'},
    'contraction': {'matmul', 'bmm', 'mm', 'einsum', 'baddbmm', 'addmm'},
    'index_route': {'gather', 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'index_select', 'index_add_', 'index_put_', 'take_along_dim'},
    'selection': {'topk', 'sort', 'argsort', 'argmax', 'argmin', 'nonzero', 'one_hot'},
    'mask_select': {'where', 'masked_fill', 'masked_fill_', 'tril', 'triu'},
    'reduce': {'sum', 'mean', 'amax', 'amin', 'max', 'min', 'prod', 'var', 'norm', 'logsumexp'},
    'prefix_scan': {'cumsum', 'cumprod', 'logcumsumexp'},
    'numeric': {'exp', 'log', 'log1p', 'sqrt', 'rsqrt', 'pow', 'square', 'sin', 'cos', 'abs', 'clamp', 'clamp_', 'softplus'},
    'activation': {'relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax', 'log_softmax'},
    'precision': {'to', 'float', 'half', 'bfloat16', 'type_as', 'autocast'},
    'tensor_creation': {'arange', 'linspace', 'zeros', 'ones', 'full', 'zeros_like', 'ones_like', 'full_like', 'new_zeros', 'new_ones', 'new_full'},
    'normalization': {'LayerNorm', 'RMSNorm', 'GroupNorm', 'BatchNorm1d', 'BatchNorm2d', 'layer_norm', 'rms_norm', 'normalize'},
    'convolution': {'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'conv1d', 'conv2d', 'conv3d'},
    'resample_pool': {'interpolate', 'grid_sample', 'pad', 'max_pool1d', 'max_pool2d', 'avg_pool2d', 'adaptive_avg_pool2d', 'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d'},
    'parameter_definition': {'Parameter', 'register_parameter', 'register_buffer', 'ParameterList', 'ParameterDict'},
    'linear_embedding': {'Linear', 'Embedding', 'linear', 'embedding'},
    'attention_kernel': {'scaled_dot_product_attention', 'eager_attention_forward', 'flash_attn_func', '_flash_attention_forward'},
    'complex_fft': {'fft', 'rfft', 'irfft', 'fft2', 'view_as_complex', 'view_as_real', 'polar'},
}
PORT_GROUPS = {
    'external_mask': {'attention_mask', 'encoder_attention_mask', 'pixel_mask'},
    'explicit_positions': {'position_ids', 'cache_position', 'position_embeddings'},
    'cache_state': {'past_key_values', 'past_key_value', 'cache_params', 'state', 'states'},
    'cross_modal_inputs': {'encoder_hidden_states', 'pixel_values', 'input_features', 'input_values', 'audio_values'},
}


def inspect_source(text: str) -> dict:
    tree = ast.parse(text)
    found: dict[str, list] = {}

    def record(group, node, symbol):
        evidence = found.setdefault(group, [])
        if len(evidence) < 5:
            evidence.append({'line': node.lineno, 'symbol': symbol})

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            symbol = ast.unparse(node.func)
            tail = symbol.rsplit('.', 1)[-1]
            for group, names in CALL_GROUPS.items():
                if tail in names:
                    record(group, node, symbol)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            record('contraction', node, '@')
        elif isinstance(node, ast.Subscript):
            # Could index a Python list/config rather than a tensor. Do not label
            # this as a proven tensor slicing requirement.
            record('subscript_signal', node, 'subscript')
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'forward':
            ports = {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
            for group, names in PORT_GROUPS.items():
                for name in sorted(ports & names):
                    record(group, node, f'forward:{name}')
    return found


def scan(root: Path, version: str) -> dict:
    records = []
    for path in sorted(root.rglob('modeling_*.py')):
        raw = path.read_bytes()
        record = {'source': str(path.relative_to(root)), 'sha256': hashlib.sha256(raw).hexdigest()}
        try:
            record['signals'] = inspect_source(raw.decode('utf-8'))
        except (SyntaxError, UnicodeError, RecursionError) as exc:
            record.update(signals={}, error=str(exc))
        records.append(record)
    groups = sorted(set(CALL_GROUPS) | set(PORT_GROUPS) | {'subscript_signal'})
    return {
        'schema': 1, 'transformers_version': version,
        'method': 'AST spelling/forward-port signals, one count per file/category; at most five evidence locations per file/category. Not semantic resolution or support/coverage proof.',
        'limitations': ['All modeling_*.py backends included; no modular_*.py or imported shared helpers.',
                       'Calls in loss/training/helpers also counted, not just inference forward.',
                       'Aliases/custom fused kernels can be missed; method spellings and Python indexing can produce false positives.',
                       'Generated copies and model variants are not independent architectures; categories overlap.'],
        'source_files': len(records),
        'parse_errors': sum('error' in r for r in records),
        'categories': {g: {'file_count': sum(g in r['signals'] for r in records),
                           'examples': [r['source'] for r in records if g in r['signals']][:8]} for g in groups},
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    if args.root:
        root, version = args.root, 'custom-source'
    else:
        dist = distribution('transformers')
        root, version = Path(dist.locate_file('transformers/models')), dist.version
    if not root.is_dir():
        parser.error(f'not a source directory: {root}')
    report = scan(root, version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
