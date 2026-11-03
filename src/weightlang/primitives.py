"""Tensor atoms: shared declarative signatures and per-sample shape rules."""
from __future__ import annotations

import json
import math
from pathlib import Path

SPECS = json.loads(Path(__file__).with_name('primitive_specs.json').read_text())


def arguments(op, args):
    spec = SPECS[op]
    args = {**spec['defaults'], **args}
    if set(args) != set(spec['required']) | set(spec['defaults']):
        raise ValueError(f'{op}: expected arguments {spec["required"]} with defaults {spec["defaults"]}')
    for key, value in args.items():
        if spec.get('types', {}).get(key, 'int') == 'float':
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{op}: {key} must be a finite number')
        elif type(value) is not int:
            raise ValueError(f'{op}: {key} must be an integer')
    if 'keepdim' in args and args['keepdim'] not in (0, 1):
        raise ValueError('keepdim must be 0 or 1')
    if args.get('step', 1) <= 0:
        raise ValueError('slice step must be positive')
    return args


def contract(op, args, operands):
    def need(ok, message):
        if not ok:
            raise ValueError(f'{op}: {message}')
    need(len(operands) == SPECS[op]['arity'], 'wrong number of tensor inputs')
    dtype, shape = operands[0]
    result = list(shape)
    rank = len(shape)
    def axis(key='axis', insert=False):
        n = rank + int(insert)
        v = args[key]
        need(-n <= v < n, 'axis outside sample rank')
        return v % n
    if op in {'square', 'rsqrt', 'exp', 'log', 'sin', 'cos', 'softplus', 'shift', 'cumsum'}:
        need(dtype == 'f32', 'expects f32')
        if op == 'cumsum':
            axis()
    elif op in {'resize2d','pad2d','avgpool2d'}:
        need(dtype == 'f32' and rank == 3 and all(d > 0 for d in shape), 'expects nonempty f32[channels,height,width]')
        if op == 'resize2d':
            need(args['height'] > 0 and args['width'] > 0, 'target size must be positive')
            need(args['mode'] in (0,1) and args['align_corners'] in (0,1), 'mode and align_corners must be 0 or 1')
            need(args['mode'] == 1 or args['align_corners'] == 0, 'nearest does not support align_corners')
            result[1:] = [args['height'],args['width']]
        elif op == 'pad2d':
            need(args['mode'] in (0,1,2), 'pad mode must be 0=zeros, 1=reflect, 2=replicate')
            need(all(args[k] >= 0 for k in ('left','right','top','bottom')), 'padding must be nonnegative')
            if args['mode'] == 1:
                need(max(args['left'],args['right']) < shape[2] and max(args['top'],args['bottom']) < shape[1], 'reflection padding must be smaller than its input dimension')
            result[1] += args['top'] + args['bottom']
            result[2] += args['left'] + args['right']
        else:
            k,s,p = args['kernel'],args['stride'],args['padding']
            need(k > 0 and s > 0 and 0 <= p <= k//2 and args['count_include_pad'] in (0,1), 'invalid pooling configuration')
            result[1:] = [(d+2*p-k)//s+1 for d in shape[1:]]
            need(all(d > 0 for d in result), 'pooling produces empty output')
    elif op == 'matmul':
        other, rhs = operands[1]
        need(dtype == other == 'f32' and rank >= 2 and len(rhs) == rank, 'requires same-rank f32 matrices')
        need(shape[:-2] == rhs[:-2] and shape[-1] == rhs[-2], 'batch/sample prefixes and contraction dimensions must agree; broadcast explicitly')
        result = [*shape[:-2], shape[-2], rhs[-1]]
    elif op == 'broadcast':
        _, target = operands[1]
        need(rank <= len(target), 'source rank exceeds target')
        padded = (1,) * (len(target)-rank) + tuple(shape)
        need(all(a == b or a == 1 for a,b in zip(padded,target)), 'cannot expand source to target shape')
        result = list(target)
    elif op in {'cat', 'stack'}:
        other, rhs = operands[1]
        need(dtype == other and len(rhs) == rank, 'input ranks and dtypes must agree')
        a = axis(insert=op == 'stack')
        if op == 'stack':
            need(shape == rhs, 'stack shapes must agree')
            result.insert(a,2)
        else:
            need(all(x == y for i,(x,y) in enumerate(zip(shape,rhs)) if i != a), 'cat dimensions must agree except concatenation axis')
            result[a] += rhs[a]
    elif op == 'slice':
        a = axis()
        need(0 <= args['start'] <= args['stop'] <= shape[a], 'slice bounds must satisfy 0 <= start <= stop <= size')
        result[a] = (args['stop']-args['start']+args['step']-1)//args['step']
    elif op == 'transpose_axes':
        a,b = axis('axis1'),axis('axis2')
        result[a],result[b] = result[b],result[a]
    elif op.startswith('reduce_'):
        need(dtype == 'f32', 'reduction expects f32')
        a = axis()
        need(op == 'reduce_sum' or shape[a] > 0, 'mean/max require a nonempty reduction axis')
        if args['keepdim']:
            result[a] = 1
        else:
            result.pop(a)
    elif op in {'gather','scatter_add'}:
        a = axis()
        index_dtype, indices = operands[1]
        need(index_dtype == 'i64' and len(indices) == rank, 'indices must be i64 with matching rank')
        need(all(i <= n for k,(i,n) in enumerate(zip(indices,shape)) if k != a), 'index dimensions exceed input')
        if op == 'gather':
            result = list(indices)
        else:
            need(dtype == 'f32' and operands[2] == ('f32',indices), 'scatter_add source must match indices shape and use f32')
    elif op in {'eq','lt'}:
        need(operands[1] == operands[0], 'comparison requires same shape and dtype')
        need(op != 'lt' or dtype != 'bool', 'lt requires numeric inputs')
        dtype = 'bool'
    elif op == 'masked_softmax':
        axis()
        need(dtype == 'f32' and operands[1] == ('bool',shape), 'mask must be bool with the logits shape; true means keep')
    return dtype, tuple(result)
