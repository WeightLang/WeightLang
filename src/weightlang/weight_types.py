"""Finite weight domains, slice mixtures and differentiable forward projection."""
import math

import torch


def validate_types(types):
    if not isinstance(types, dict):
        raise ValueError('weight types must be a dictionary')
    result = {}
    for name, values in types.items():
        if not isinstance(name, str) or not name.isascii() or not name.isidentifier() or not isinstance(values, list) or not values:
            raise ValueError('wtype requires a name and a nonempty list')
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
            raise ValueError('wtype values must be finite f32 numbers')
        # Validate in the storage dtype, including underflow and rounding collisions.
        import struct
        values = sorted(struct.unpack('f', struct.pack('f', v))[0] for v in values)
        if len(set(values)) != len(values):
            raise ValueError('wtype values must be distinct in f32')
        result[name] = values
    return result


def project_weight(value, rules):
    """Nearest codebook value; ties choose the lower value; identity STE gradient."""
    result = value
    for rule in rules:
        selection = [slice(None)] * value.ndim
        if 'axis' in rule:
            selection[rule['axis']] = slice(rule['start'], rule['stop'])
        selection = tuple(selection)
        part = value[selection]
        book = part.new_tensor(rule['values'])
        index = torch.searchsorted(book, part.detach().contiguous()).clamp(max=len(book)-1)
        lower = (index-1).clamp(min=0)
        projected = torch.where((part.detach()-book[lower]).abs() <= (part.detach()-book[index]).abs(), book[lower], book[index])
        effective = projected.detach() + (part-part.detach())
        if 'axis' not in rule:
            result = effective
        else:
            if result is value:
                result = value.clone()
            result[selection] = effective
    return result


def weight_view(module, cache):
    """Copy only constrained module paths, including weights read by a parent."""
    import copy
    children = {name: weight_view(child, cache) for name, child in module._modules.items()}
    rules = getattr(module, '_weight_rules', {})
    if not rules and all(children[name] is child for name, child in module._modules.items()):
        return module
    view = copy.copy(module)
    forward = module.__dict__.get("forward")
    if getattr(forward, "__self__", None) is module:
        from types import MethodType
        view.forward = MethodType(forward.__func__, view)
    view._modules = children
    view._parameters = module._parameters.copy()
    for name, domain in rules.items():
        parameter = module._parameters[name]
        if id(parameter) not in cache:
            cache[id(parameter)] = project_weight(parameter, domain)
        view._parameters[name] = cache[id(parameter)]
    return view


class WeightTyped:
    """Use a per-call module view; keep master Parameters and state paths intact."""
    _classes = {}

    def forward(self, *args, **kwargs):
        view = weight_view(self, {})
        return super(WeightTyped, view).forward(*args, **kwargs)


def attach_weight_types(module, types, assignments):
    types = validate_types(types)
    if not isinstance(assignments, list):
        raise ValueError('weight assignments must be a list')
    parameters = dict(module.named_parameters(remove_duplicate=False))
    by_identity = {}
    for spec in assignments:
        if not isinstance(spec, dict) or set(spec) not in ({'target','type'}, {'target','type','axis','start','stop'}):
            raise ValueError('weight requires target/type and optionally axis/start/stop')
        target, kind = spec['target'], spec['type']
        if not isinstance(target,str) or target not in parameters or not isinstance(kind,str) or kind not in types:
            raise ValueError('weight refers to an unknown parameter or wtype')
        parameter = parameters[target]
        rule = {'values': types[kind]}
        if 'axis' in spec:
            axis, start, stop = (spec[k] for k in ('axis','start','stop'))
            if any(type(v) is not int for v in (axis,start,stop)) or not 0 <= axis < parameter.ndim or not 0 <= start < stop <= parameter.shape[axis]:
                raise ValueError('weight slice requires a valid axis and 0 <= start < stop <= axis size')
            rule.update(axis=axis,start=start,stop=stop)
        previous = by_identity.setdefault(id(parameter), [])
        for other in previous:
            if 'axis' not in other or 'axis' not in rule or other['axis'] != rule['axis'] or max(other['start'],rule['start']) < min(other['stop'],rule['stop']):
                raise ValueError('weight assignments overlap or mix axes on a shared parameter')
        previous.append(rule)
    # Aliases receive the same domain automatically; no independent master copy.
    targets = []
    for path, parameter in parameters.items():
        if id(parameter) not in by_identity:
            continue
        parent, _, leaf = path.rpartition('.')
        owner = module.get_submodule(parent)
        if hasattr(owner, 'lora_rank'):
            raise ValueError('weight types on LoRA modules are not supported')
        targets.append((owner,leaf,by_identity[id(parameter)]))
    for owner, leaf, rules in targets:
        if '_weight_rules' not in owner.__dict__:
            owner._weight_rules = {}
        owner._weight_rules[leaf] = rules
    if targets and not isinstance(module, WeightTyped):
        base = type(module)
        if base not in WeightTyped._classes:
            WeightTyped._classes[base] = type('WeightTyped'+base.__name__, (WeightTyped,base), {})
        module.__class__ = WeightTyped._classes[base]
