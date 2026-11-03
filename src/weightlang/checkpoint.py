"""Optimizer-bearing checkpoints, distinct from deployable model artifacts."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import tempfile

import torch

from .parameter_identity import parameter_inventory
from .syntax import WgtError


_ALLOWED = {torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW}


def _parameters(model):
    if any(getattr(module,"_unmerged_weight",None) is not None for module in model.modules()):
        raise WgtError("unmerge adapters before saving/restoring a training checkpoint")
    inventory = parameter_inventory(model)
    return inventory, {key:model.get_parameter(info['paths'][0]) for key,info in inventory.items()}


def _optimizer(optimizer, params):
    if type(optimizer) not in _ALLOWED:
        raise WgtError('checkpoint supports torch.optim SGD, Adam and AdamW')
    ids = {id(param):key for key,param in params.items()}
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        keys = []
        for parameter in group['params']:
            key = ids.get(id(parameter))
            if key is None or key in seen:
                raise WgtError('optimizer parameters must be unique model parameters')
            seen.add(key)
            keys.append(key)
        groups.append({**{k:copy.deepcopy(v) for k,v in group.items() if k != 'params'},'params':keys})
    state = {ids[id(p)]:copy.deepcopy(value) for p,value in optimizer.state.items()}
    return dict(kind=type(optimizer).__name__,groups=groups,state=state)


def _adapter_contract(model):
    return {name: {key: spec.get(key, 'lora' if key == 'method' else None)
                   for key in ('method', 'rank', 'alpha', 'freeze')}
            for name, spec in getattr(model, '_wgt_adapters', {}).items()}


def save_checkpoint(model, optimizer, path, *, progress=None, force=False):
    """Atomically save weights, identity-indexed optimizer state, RNG and progress.

    No architecture is embedded. Reconstruct the model/optimizer explicitly before
    restore_checkpoint; this file cannot be loaded as a deployable WGT artifact.
    """
    path = Path(path)
    if path.exists() and not force:
        raise WgtError('checkpoint exists; pass force=True to overwrite')
    progress = copy.deepcopy(progress or {})
    if not isinstance(progress,dict):
        raise WgtError('checkpoint progress must be a JSON dictionary')
    try:
        json.dumps(progress,allow_nan=False)
    except (ValueError,TypeError):
        raise WgtError('checkpoint progress must contain finite JSON values') from None
    inventory,params = _parameters(model)
    record = dict(format='weightlang.training-checkpoint',version=1,inventory=inventory,adapters=_adapter_contract(model),
                  parameters={k:p.detach().cpu().clone() for k,p in params.items()},
                  buffers={k:b.detach().cpu().clone() for k,b in model.named_buffers()},
                  optimizer=_optimizer(optimizer,params),progress=progress,
                  modes={k:m.training for k,m in model.named_modules()},
                  rng=dict(python=random.getstate(),torch=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))
    try:
        import numpy as np
        kind,keys,pos,gaussian,cached = np.random.get_state()
        record['rng']['numpy'] = [kind,keys.tolist(),pos,gaussian,cached]
    except ImportError:
        pass
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(record,stream)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary,path)
        else:
            os.link(temporary,path)  # Atomic no-clobber publication.
        return path
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path):
    try:
        record = torch.load(path,map_location='cpu',weights_only=True)
    except Exception as exc:
        raise WgtError(f'cannot read training checkpoint: {exc}') from None
    if not isinstance(record,dict) or record.get('format') != 'weightlang.training-checkpoint' or record.get('version') != 1:
        raise WgtError('not a supported WeightLang training checkpoint')
    return record


def checkpoint_report(model, path, *, mapping=None):
    return _report(model, _read(path), mapping=mapping)


def _report(model, record, *, mapping=None):
    current,params = _parameters(model)
    mapping = dict(mapping or {})
    saved = record['inventory']
    if set(mapping)-set(current) or any(v not in saved for v in mapping.values()):
        raise WgtError('checkpoint mapping references unknown identities')
    report = dict(reusable=[],conflicts=[],missing=[],unused=[])
    if record.get('adapters', {}) != _adapter_contract(model):
        report['conflicts'].append(dict(target='adapters', source='adapters'))
    used = set()
    for key,spec in current.items():
        source = mapping.get(key,key)
        if source not in saved:
            report['missing'].append(key)
        elif any(spec[field] != saved[source][field] for field in ('shape','dtype','trainable')):
            report['conflicts'].append(dict(target=key,source=source))
        else:
            report['reusable'].append(dict(target=key,source=source))
        if source in saved:
            used.add(source)
    report['unused'] = sorted(set(saved)-used)
    return report


def restore_checkpoint(model, optimizer, path, *, mapping=None, restore_rng=True):
    """Strict all-or-nothing compatibility checks before restoring training state.

    Compatible parameter/module renames use identities; modes require the same
    set of module paths (renamed paths inherit the current caller's mode).
    Optimizer groups may reorder parameters, but must contain the same identities.
    """
    record = _read(path)
    current,params = _parameters(model)
    mapping = dict(mapping or {})
    report = _report(model,record,mapping=mapping)
    if report['conflicts'] or report['missing'] or report['unused']:
        raise WgtError('incompatible training checkpoint: '+json.dumps(report))
    source_keys = [mapping.get(key,key) for key in params]
    if len(set(source_keys)) != len(source_keys):
        raise WgtError('checkpoint parameter mapping must be one-to-one')
    saved = record['optimizer']
    active = _optimizer(optimizer,params)
    if saved['kind'] != active['kind'] or len(saved['groups']) != len(active['groups']):
        raise WgtError('optimizer kind/group count mismatch')
    buffers = dict(model.named_buffers())
    if set(buffers) != set(record['buffers']):
        raise WgtError('checkpoint buffer paths mismatch')
    for name,value in buffers.items():
        stored = record['buffers'][name]
        if not isinstance(stored,torch.Tensor) or (stored.shape,stored.dtype) != (value.shape,value.dtype):
            raise WgtError('checkpoint buffer shape/dtype mismatch')
    for key,param in params.items():
        tensor = record['parameters'].get(mapping.get(key,key))
        if not isinstance(tensor,torch.Tensor) or (tensor.shape,tensor.dtype) != (param.shape,param.dtype):
            raise WgtError('checkpoint parameter payload mismatch')
    restored = dict(state={},param_groups=[])
    index = 0
    for old,group in zip(saved['groups'],active['groups']):
        if set(old['params']) != {mapping.get(k,k) for k in group['params']}:
            raise WgtError('optimizer group identity mismatch')
        indices = []
        for key in group['params']:
            source = mapping.get(key,key)
            value = saved['state'].get(source,{})
            for item in value.values():
                if isinstance(item,torch.Tensor) and item.ndim and item.shape != params[key].shape:
                    raise WgtError('optimizer state shape mismatch')
            restored['state'][index] = copy.deepcopy(value)
            indices.append(index)
            index += 1
        restored['param_groups'].append({**copy.deepcopy(old),'params':indices})
    rng = record['rng']
    if restore_rng:
        try:
            random.Random().setstate(rng['python'])
            torch.Generator().set_state(rng['torch'])
            if rng['cuda'] and len(rng['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA device count differs from checkpoint')
            if 'numpy' in rng:
                import numpy as np
                kind,keys,pos,gaussian,cached = rng['numpy']
                np.random.RandomState().set_state((kind,np.asarray(keys,dtype='uint32'),pos,gaussian,cached))
        except (ValueError,TypeError,RuntimeError,ImportError) as exc:
            raise WgtError(f'incompatible checkpoint RNG: {exc}') from None
    # Validate optimizer loading on an isolated copy so errors do not alter caller state.
    shadow = copy.deepcopy(optimizer)
    shadow.load_state_dict(restored)
    optimizer.load_state_dict(restored)
    with torch.no_grad():
        for key,param in params.items():
            param.copy_(record['parameters'][mapping.get(key,key)])
        for key,buffer in buffers.items():
            buffer.copy_(record['buffers'][key])
    for key,module in model.named_modules():
        if key in record['modes']:
            module.training = record['modes'][key]
    if restore_rng:
        random.setstate(rng['python'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])
        if 'numpy' in rng:
            np.random.set_state((kind,np.asarray(keys,dtype='uint32'),pos,gaussian,cached))
    return copy.deepcopy(record['progress'])
