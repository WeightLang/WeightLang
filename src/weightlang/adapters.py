"""Declared low-rank Linear adapters, canonical serialization and exact unmerge."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Linear):
    def __init__(self, base, rank, alpha, freeze):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features,base.out_features
        self.weight, self.bias = base.weight,base.bias
        self.lora_rank, self.lora_alpha = rank,alpha
        self.lora_a = nn.Parameter(base.weight.new_empty(rank,self.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(self.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
        self.register_buffer('_unmerged_weight',None,persistent=False)
        if freeze:
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)

    def forward(self,x):
        result = nn.functional.linear(x,self.weight,self.bias)
        if self._unmerged_weight is None:
            result = result + nn.functional.linear(nn.functional.linear(x,self.lora_a),self.lora_b)*(self.lora_alpha/self.lora_rank)
        return result

    def train(self, mode=True):
        if mode and self._unmerged_weight is not None:
            raise ValueError('unmerge adapter before training')
        return super().train(mode)

    def merge(self):
        if self.training:
            raise ValueError('merge requires eval mode')
        if self._unmerged_weight is None:
            with torch.no_grad():
                self._unmerged_weight = self.weight.detach().clone()
                self.weight.add_((self.lora_b @ self.lora_a)*(self.lora_alpha/self.lora_rank))

    def unmerge(self):
        if self._unmerged_weight is not None:
            with torch.no_grad():
                self.weight.copy_(self._unmerged_weight)
            self._unmerged_weight = None

    def _save_to_state_dict(self,destination,prefix,keep_vars):
        super()._save_to_state_dict(destination,prefix,keep_vars)
        if self._unmerged_weight is not None:
            destination[prefix+'weight'] = self._unmerged_weight if keep_vars else self._unmerged_weight.detach()



class DoRALinear(LoRALinear):
    def __init__(self, base, rank, alpha, freeze):
        super().__init__(base, rank, alpha, freeze)
        self.magnitude = nn.Parameter(self.weight.detach().norm(dim=1))

    def effective_weight(self):
        direction = self.weight + (self.lora_b @ self.lora_a) * (self.lora_alpha / self.lora_rank)
        # DoRA treats the row norm as a detached normalization factor.
        norm = direction.norm(dim=1).detach().clamp_min(torch.finfo(direction.dtype).eps)
        return direction * (self.magnitude / norm).unsqueeze(1)

    def forward(self, x):
        weight = self.weight if self._unmerged_weight is not None else self.effective_weight()
        return nn.functional.linear(x, weight, self.bias)

    def merge(self):
        if self.training:
            raise ValueError('merge requires eval mode')
        if self._unmerged_weight is None:
            with torch.no_grad():
                effective = self.effective_weight()
                self._unmerged_weight = self.weight.detach().clone()
                self.weight.copy_(effective)


class PiSSALinear(LoRALinear):
    def __init__(self, base, rank, alpha, freeze):
        super().__init__(base, rank, alpha, freeze)
        if not self.weight.is_meta:
            with torch.no_grad():
                u, singular, vh = torch.linalg.svd(self.weight, full_matrices=False)
                root = (singular[:rank] / (alpha / rank)).sqrt()
                self.lora_a.copy_(root[:, None] * vh[:rank])
                self.lora_b.copy_(u[:, :rank] * root[None, :])
                # Preserve the initial function: frozen residual + principal component.
                self.weight.sub_((self.lora_b @ self.lora_a) * (alpha / rank))


class NF4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scales, out_features, in_features):
        ctx.save_for_backward(packed, scales)
        ctx.shape = (out_features, in_features)
        weight = QLoRALinear.decode(packed, scales, ctx.shape)
        return nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, grad):
        packed, scales = ctx.saved_tensors
        # Reconstruct only this layer's base; do not retain dense weights for backward.
        weight = QLoRALinear.decode(packed, scales, ctx.shape)
        return grad @ weight, None, None, None, None


class QLoRALinear(LoRALinear):
    # Fixed block size 64, two NF4 codes per byte, f32 block absmax.
    codebook = (-1., -.6961928009986877, -.5250730514526367, -.39491748809814453,
                -.28444138169288635, -.18477343022823334, -.09105003625154495, 0.,
                .07958029955625534, .16093020141124725, .24611230194568634,
                .33791524171829224, .44070982933044434, .5626170039176941,
                .7229568362236023, 1.)

    def __init__(self, base, rank, alpha, freeze):
        super().__init__(base, rank, alpha, freeze)
        weight = self._parameters.pop('weight').detach()
        count = self.in_features * self.out_features
        self.register_buffer('packed', torch.empty((count + 1) // 2, dtype=torch.uint8, device=weight.device), persistent=False)
        self.register_buffer('scales', weight.new_empty((count + 63) // 64), persistent=False)
        if not weight.is_meta:
            self.encode(weight)

    def __getattr__(self, name):
        if name == 'weight' and 'weight' not in self._parameters:
            merged = self._buffers.get('_unmerged_weight')
            if merged is not None:
                return self._merged_weight
            return self.decode(self.packed, self.scales, (self.out_features, self.in_features))
        return super().__getattr__(name)

    @staticmethod
    def decode(packed, scales, shape):
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten()[:shape[0] * shape[1]].long()
        table = scales.new_tensor(QLoRALinear.codebook)
        return (table[codes] * scales.repeat_interleave(64)[:codes.numel()]).reshape(shape)

    def encode(self, weight):
        with torch.no_grad():
            flat = weight.detach().flatten()
            padded = nn.functional.pad(flat, (0, (-flat.numel()) % 64)).reshape(-1, 64)
            scales = padded.abs().amax(dim=1)
            normalized = (padded / scales.clamp_min(torch.finfo(weight.dtype).tiny)[:, None]).flatten()[:flat.numel()]
            table = weight.new_tensor(self.codebook)
            # Midpoint lookup avoids an N x 16 temporary during quantization.
            codes = torch.bucketize(normalized, (table[:-1] + table[1:]) / 2).to(torch.uint8)
            codes = nn.functional.pad(codes, (0, codes.numel() % 2))
            self.packed.copy_(codes[0::2] | (codes[1::2] << 4))
            self.scales.copy_(scales)

    def forward(self, x):
        if self._unmerged_weight is not None:
            return nn.functional.linear(x, self._merged_weight, self.bias)
        result = NF4LinearFunction.apply(x, self.packed, self.scales, self.out_features, self.in_features)
        if self.bias is not None:
            result = result + self.bias
        return result + nn.functional.linear(nn.functional.linear(x, self.lora_a), self.lora_b) * (self.lora_alpha / self.lora_rank)

    def merge(self):
        if self.training:
            raise ValueError('merge requires eval mode')
        if self._unmerged_weight is None:
            with torch.no_grad():
                self._unmerged_weight = self.weight.detach()
                self.register_buffer('_merged_weight', self._unmerged_weight + (self.lora_b @ self.lora_a) * (self.lora_alpha / self.lora_rank), persistent=False)

    def unmerge(self):
        self._unmerged_weight = None
        if '_merged_weight' in self._buffers:
            del self._buffers['_merged_weight']

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Existing WGT artifacts are canonical f32. Packed storage is runtime-only.
        weight = self._unmerged_weight if self._unmerged_weight is not None else self.weight
        destination[prefix + 'weight'] = weight if keep_vars else weight.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict = dict(state_dict)
        weight = state_dict.pop(prefix + 'weight', None)
        if weight is None:
            missing_keys.append(prefix + 'weight')
        elif tuple(weight.shape) != (self.out_features, self.in_features) or weight.dtype != torch.float32 or not torch.isfinite(weight).all():
            error_msgs.append('invalid QLoRA base weight')
        else:
            self.unmerge()
            self.encode(weight.to(self.scales.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

def attach_adapters(module, specs):
    if not isinstance(specs,dict):
        raise ValueError('adapters must be a dictionary')
    seen = set()
    parameters = list(module.named_parameters(remove_duplicate=False))
    for name,spec in specs.items():
        if not isinstance(spec,dict) or set(spec) not in ({'target','rank','alpha','freeze'}, {'target','rank','alpha','freeze','method'}):
            raise ValueError('adapter requires target/rank/alpha/freeze')
        method = spec.get('method', 'lora')
        if method not in ('lora', 'dora', 'pissa', 'qlora'):
            raise ValueError('unknown adapter method')
        if method != 'lora' and spec['freeze'] != 1:
            raise ValueError('dora/pissa/qlora require freeze=1')
        target,rank,alpha,freeze = (spec[k] for k in ('target','rank','alpha','freeze'))
        if not isinstance(name,str) or not name.isidentifier() or not isinstance(target,str) or target in seen:
            raise ValueError('invalid/duplicate adapter target or name')
        try:
            base = module.get_submodule(target)
        except AttributeError:
            raise ValueError('unknown adapter target') from None
        if type(base) is not nn.Linear or type(rank) is not int or not 1 <= rank <= min(base.in_features,base.out_features):
            raise ValueError('LoRA target must be Linear with rank in [1,min(in,out)]')
        if type(alpha) not in (float,int) or not math.isfinite(alpha) or alpha <= 0 or type(freeze) is not int or freeze not in (0,1):
            raise ValueError('LoRA alpha must be positive finite and freeze=0/1')
        if sum(parameter is base.weight for _,parameter in parameters) != 1:
            raise ValueError('LoRA targets with shared base weights are unsupported')
        if base.bias is not None and sum(parameter is base.bias for _,parameter in parameters) != 1:
            raise ValueError('adapter targets with shared bias are unsupported')
        seen.add(target)
    for spec in specs.values():
        target = spec['target']
        parent,_,leaf = target.rpartition('.')
        base = module.get_submodule(target)
        module.get_submodule(parent)._modules[leaf] = {'lora': LoRALinear, 'dora': DoRALinear, 'pissa': PiSSALinear, 'qlora': QLoRALinear}[spec.get('method', 'lora')](base,spec['rank'],spec['alpha'],spec['freeze'])
    module._wgt_adapters = dict(specs)


def _adapter(model,name):
    spec = getattr(model,'_wgt_adapters',{}).get(name)
    if spec is None:
        raise ValueError('unknown declared adapter')
    return model.get_submodule(spec['target']),spec


def merge_adapter(model,name):
    layer,_ = _adapter(model,name)
    layer.merge()
    return model


def unmerge_adapter(model,name):
    layer,_ = _adapter(model,name)
    layer.unmerge()
    return model


def _base_hash(layer):
    h = hashlib.sha256()
    state = layer.state_dict()
    for key in ('weight','bias'):
        value = state.get(key)
        if value is not None:
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_adapter(model,name,path,*,force=False):
    """Save only adapter tensors plus an exact base-weight digest, never a model."""
    import os
    import tempfile
    from .syntax import WgtError
    layer,spec = _adapter(model,name)
    path = Path(path)
    if path.exists() and not force:
        raise WgtError('adapter exists; pass force=True to overwrite')
    record = dict(format='weightlang.lora',version=1,rank=spec['rank'],alpha=spec['alpha'],base_sha256=_base_hash(layer),
                  a=layer.lora_a.detach().cpu().clone(),b=layer.lora_b.detach().cpu().clone())
    method = spec.get('method', 'lora')
    if method != 'lora':
        record.update(format='weightlang.adapter', version=2, method=method)
    if method == 'dora':
        record['magnitude'] = layer.magnitude.detach().cpu().clone()
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(record,stream)
        if force:
            os.replace(tmp,path)
        else:
            os.link(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_adapter(model,name,path):
    from .syntax import WgtError
    layer,spec = _adapter(model,name)
    if layer._unmerged_weight is not None:
        raise WgtError('unmerge adapter before loading')
    record = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(record, dict):
        raise WgtError('unsupported adapter artifact')
    method = spec.get('method', 'lora')
    valid = (record.get('format'), record.get('version')) == ('weightlang.lora', 1) if method == 'lora' else (record.get('format'), record.get('version'), record.get('method')) == ('weightlang.adapter', 2, method)
    if not valid:
        raise WgtError('unsupported adapter artifact or method mismatch')
    if (record.get('rank'),record.get('alpha'),record.get('base_sha256')) != (spec['rank'],spec['alpha'],_base_hash(layer)):
        raise WgtError('adapter rank/alpha/base digest mismatch')
    pairs = [('a',layer.lora_a),('b',layer.lora_b)]
    if method == 'dora':
        pairs.append(('magnitude', layer.magnitude))
    for key,target in pairs:
        value = record.get(key)
        if not isinstance(value,torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
            raise WgtError('adapter tensor shape/dtype/value mismatch')
    with torch.no_grad():
        for key, target in pairs:
            target.copy_(record[key])
    return model
