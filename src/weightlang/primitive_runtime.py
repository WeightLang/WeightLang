"""Audited tensor primitives; this class is copied into standalone exports."""
import torch
from torch import nn


class TensorPrimitive(nn.Module):
    def __init__(self, op, args, rank):
        super().__init__()
        self.op, self.args, self.rank = op, args, rank

    def forward(self, x, *rest):
        op, args = self.op, self.args
        offset = x.ndim - self.rank
        def axis(key='axis', insert=False):
            return offset + args[key] % (self.rank + int(insert))
        if op in {'square', 'rsqrt', 'exp', 'log', 'sin', 'cos'}:
            return getattr(torch, op)(x)
        if op == 'softplus':
            return torch.nn.functional.softplus(x, beta=1, threshold=20)
        if op == 'shift':
            return x + args['value']
        if op == 'cumsum':
            return torch.cumsum(x, dim=axis())
        if op in {'resize2d','pad2d','avgpool2d'}:
            value = x.unsqueeze(0) if offset == 0 else x
            if op == 'resize2d':
                options = {'mode':'nearest'} if args['mode'] == 0 else {'mode':'bilinear','align_corners':bool(args['align_corners'])}
                out = torch.nn.functional.interpolate(value, size=(args['height'],args['width']), **options)
            elif op == 'pad2d':
                padding = tuple(args[k] for k in ('left','right','top','bottom'))
                out = torch.nn.functional.pad(value, padding, mode=('constant','reflect','replicate')[args['mode']])
            else:
                out = torch.nn.functional.avg_pool2d(value, args['kernel'], args['stride'], args['padding'], count_include_pad=bool(args['count_include_pad']))
            return out.squeeze(0) if offset == 0 else out
        if op == 'matmul':
            return torch.matmul(x, rest[0])
        if op == 'broadcast':
            target = rest[0]
            rank = target.ndim-offset
            shape = (*x.shape[:offset], *((1,)*(rank-self.rank)), *x.shape[offset:])
            return x.reshape(shape).expand_as(target)
        if op in {'cat','stack'}:
            return getattr(torch, op)((x,rest[0]), dim=axis(insert=op == 'stack'))
        if op == 'slice':
            slices = [slice(None)] * x.ndim
            slices[axis()] = slice(args['start'],args['stop'],args['step'])
            return x[tuple(slices)]
        if op == 'transpose_axes':
            return x.transpose(axis('axis1'),axis('axis2'))
        if op.startswith('reduce_'):
            fn = {'reduce_sum':torch.sum,'reduce_mean':torch.mean,'reduce_max':torch.amax}[op]
            return fn(x,dim=axis(),keepdim=bool(args['keepdim']))
        if op in {'gather','scatter_add'}:
            index = rest[0]
            if torch.any(index < 0) or torch.any(index >= x.shape[axis()]):
                raise ValueError(f'{op}: index outside axis bounds')
            return torch.gather(x,axis(),index) if op == 'gather' else x.scatter_add(axis(),index,rest[1])
        if op in {'eq','lt'}:
            return getattr(torch,op)(x,rest[0])
        if op == 'masked_softmax':
            mask = rest[0]
            # Avoid NaN in both forward and backward for completely masked rows.
            valid = mask.any(dim=axis(),keepdim=True)
            logits = x.masked_fill(~mask,float('-inf'))
            logits = torch.where(valid,logits,torch.zeros_like(logits))
            return torch.softmax(logits,dim=axis()).masked_fill(~mask,0)
        raise ValueError(f'unknown tensor primitive {op}')
