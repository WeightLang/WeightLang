"""Stable declared parameter identities, aliases and explicit migration reports."""
from __future__ import annotations

import re

import torch


def bind_parameters(module, identities, shares):
    """Bind existing Parameters without allocating/copying; standalone-export safe."""
    if not isinstance(identities, dict) or not isinstance(shares, dict):
        raise ValueError('parameter identities/shares must be dictionaries')
    if any(not isinstance(name,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name) for name in identities):
        raise ValueError('invalid parameter identity')
    parameters = dict(module.named_parameters(remove_duplicate=False))
    if any(not isinstance(path,str) or path not in parameters for path in identities.values()):
        raise ValueError('parameter identity refers to an unknown parameter')
    if len(set(identities.values())) != len(identities):
        raise ValueError('one parameter path cannot have multiple identities')
    owner_paths = set(identities.values())
    for destination, identity in shares.items():
        if destination not in parameters or identity not in identities:
            raise ValueError('share refers to an unknown parameter or identity')
        if destination in owner_paths:
            raise ValueError('a shared destination cannot also declare an identity')
        owner = parameters[identities[identity]]
        other = parameters[destination]
        if owner.shape != other.shape or owner.dtype != other.dtype:
            raise ValueError('shared parameter shape/dtype mismatch')
    declared = {}
    for identity,path in identities.items():
        key = id(parameters[path])
        if key in declared:
            raise ValueError('one parameter object cannot have multiple identities')
        declared[key] = identity
    for destination, identity in shares.items():
        parent, _, leaf = destination.rpartition('.')
        module.get_submodule(parent)._parameters[leaf] = parameters[identities[identity]]
    module._parameter_identities = dict(identities)
    module._parameter_shares = dict(shares)


def parameter_inventory(module):
    """Identity is explicit when declared; undeclared paths remain path-sensitive."""
    owners = getattr(module,'_parameter_identities',{})
    parameters = dict(module.named_parameters())
    declared = {id(parameters[path]): 'id:'+identity for identity,path in owners.items() if path in parameters}
    for name,spec in getattr(module,'_wgt_adapters',{}).items():
        layer = module.get_submodule(spec['target'])
        declared[id(layer.lora_a)] = 'adapter:'+name+':a'
        declared[id(layer.lora_b)] = 'adapter:'+name+':b'
        if spec.get('method') == 'dora':
            declared[id(layer.magnitude)] = 'adapter:'+name+':magnitude'
    result = {}
    objects = {}
    for path, parameter in module.named_parameters(remove_duplicate=False):
        key = objects.setdefault(id(parameter), declared.get(id(parameter),'path:'+path))
        entry = result.setdefault(key,dict(paths=[],shape=list(parameter.shape),dtype=str(parameter.dtype),trainable=parameter.requires_grad))
        entry['paths'].append(path)
    return result


def migrate_parameters(source, target, *, mapping=None, apply=False):
    """Report first; optional apply copies only unambiguous compatible identities.

    mapping maps target inventory IDs to source inventory IDs. No order matching,
    resizing or optimizer migration is performed by this function.
    """
    from .syntax import WgtError
    old,new = parameter_inventory(source),parameter_inventory(target)
    mapping = dict(mapping or {})
    if set(mapping)-set(new) or any(v not in old for v in mapping.values()):
        raise WgtError('parameter mapping references unknown identities')
    report = dict(reusable=[],conflicts=[],new=[],unused=[])
    used = set()
    copies = []
    for identity,entry in new.items():
        origin = mapping.get(identity,identity)
        if origin not in old:
            report['new'].append(identity)
            continue
        used.add(origin)
        if (entry['shape'],entry['dtype']) != (old[origin]['shape'],old[origin]['dtype']):
            report['conflicts'].append(dict(target=identity,source=origin,expected=entry,actual=old[origin]))
        else:
            report['reusable'].append(dict(target=identity,source=origin))
            copies.append((target.get_parameter(entry['paths'][0]),source.get_parameter(old[origin]['paths'][0])))
    report['unused'] = sorted(set(old)-used)
    if apply:
        if report['conflicts']:
            raise WgtError('parameter migration has conflicts; inspect report before applying')
        with torch.no_grad():
            for destination,value in copies:
                destination.copy_(value)
    return report
