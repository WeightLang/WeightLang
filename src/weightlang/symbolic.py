"""Bounded integer dimension contracts; exhaustive checking, never source evaluation."""
from __future__ import annotations

import copy
import itertools
import math
import re


class DimensionContract:
    """The same closed integer evaluator is copied into standalone exports."""
    LIMIT = 2**31 - 1

    @staticmethod
    def evaluate(expr, bindings, depth=0):
        if depth > 16:
            raise ValueError("dimension expression exceeds depth 16")
        if type(expr) is int:
            value = expr
        elif isinstance(expr, str):
            if expr not in bindings:
                raise ValueError(f"unknown dimension {expr!r}")
            value = bindings[expr]
        elif isinstance(expr, list) and len(expr) == 3:
            op, left, right = expr
            a = DimensionContract.evaluate(left, bindings, depth + 1)
            b = DimensionContract.evaluate(right, bindings, depth + 1)
            if op == '+':
                value = a + b
            elif op == '-':
                value = a - b
            elif op == '*':
                value = a * b
            elif op in ('//', '%') and b > 0:
                value = a // b if op == '//' else a % b
            else:
                raise ValueError("dimension expression requires + - * or // % with a positive divisor")
        else:
            raise ValueError("invalid dimension expression")
        if type(value) is not int or not -DimensionContract.LIMIT <= value <= DimensionContract.LIMIT:
            raise ValueError("dimension arithmetic exceeds signed 31-bit magnitude")
        return value

    @staticmethod
    def satisfies(constraints, bindings):
        result = True
        for constraint in constraints:
            if not isinstance(constraint, list) or len(constraint) != 3:
                raise ValueError("invalid dimension constraint")
            op, left, right = constraint
            a, b = DimensionContract.evaluate(left, bindings), DimensionContract.evaluate(right, bindings)
            if op not in ('==', '!=', '<', '<=', '>', '>='):
                raise ValueError("invalid dimension comparison")
            result = {'==': a == b, '!=': a != b, '<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[op] and result
        return result

    @staticmethod
    def specialize(interface, bindings):
        result = {}
        for direction in ('inputs', 'outputs'):
            result[direction] = []
            for port in interface[direction]:
                spec = dict(port)
                exprs = spec.pop('shape_expr', spec['shape'])
                spec['shape'] = [DimensionContract.evaluate(e, bindings) for e in exprs]
                if any(n < 0 for n in spec['shape']):
                    raise ValueError("symbolic port dimensions must remain nonnegative")
                result[direction].append(spec)
        return result

    @staticmethod
    def bind(interface, values):
        bindings = {}
        for port in interface['inputs']:
            value = values[port['name']]
            rank = len(port['shape'])
            if not hasattr(value, 'shape') or len(value.shape) not in (rank, rank + 1):
                raise ValueError(f"port {port['name']}: invalid rank")
            for expr, size in zip(port.get('shape_expr', port['shape']), value.shape[-rank:]):
                if isinstance(expr, str):
                    if expr in bindings and bindings[expr] != size:
                        raise ValueError(f"dimension {expr}: inconsistent input lengths")
                    bindings[expr] = size
        for name, bounds in interface['dimensions'].items():
            if name not in bindings or not bounds['min'] <= bindings[name] <= bounds['max']:
                raise ValueError(f"dimension {name}: outside declared range")
        if not DimensionContract.satisfies(interface.get('constraints', []), bindings):
            raise ValueError("dimension constraint failed")
        return DimensionContract.specialize(interface, bindings)


def encode(expr, constants, dimensions, error):
    from .expressions import Expression
    if isinstance(expr, Expression):
        if expr.op not in ('+', '-', '*', '//', '%'):
            raise error("symbolic dimensions require integer arithmetic (no /)")
        return [expr.op, encode(expr.left, constants, dimensions, error), encode(expr.right, constants, dimensions, error)]
    if isinstance(expr, str):
        if expr in dimensions:
            return expr
        if expr not in constants:
            raise error(f"unknown dimension or constant {expr!r}")
        expr = constants[expr]
    if type(expr) is not int or abs(expr) > DimensionContract.LIMIT:
        raise error("dimension expression requires bounded integers")
    return expr


def display(expr):
    if isinstance(expr, list):
        return f"({display(expr[1])} {expr[0]} {display(expr[2])})"
    return str(expr)


def environments(record):
    dimensions = record['dimensions']
    if not isinstance(dimensions, dict) or not 1 <= len(dimensions) <= 8:
        raise ValueError("declare between 1 and 8 bounded dimensions")
    dimensions = dict(sorted(dimensions.items()))
    for name, bounds in dimensions.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) or not isinstance(bounds, dict) or set(bounds) != {'min', 'max'}:
            raise ValueError("invalid dimension declaration")
        if any(type(bounds[k]) is not int for k in bounds) or not 0 <= bounds['min'] <= bounds['max'] <= DimensionContract.LIMIT:
            raise ValueError("dimension bounds require 0 <= min <= max <= 2147483647")
    if math.prod(b['max'] - b['min'] + 1 for b in dimensions.values()) > 4096:
        raise ValueError("dimension domain exceeds 4096 combinations; narrow the bounds")
    if not isinstance(record.get('constraints', []), list) or len(record.get('constraints', [])) > 64:
        raise ValueError("at most 64 dimension constraints are supported")
    bound = {e for p in record['inputs'] for e in p.get('shape_expr', p['shape']) if isinstance(e, str)}
    if bound != dimensions.keys():
        raise ValueError("each dimension must appear directly on an input axis")
    found = False
    for sizes in itertools.product(*(range(b['min'], b['max'] + 1) for b in dimensions.values())):
        env = dict(zip(dimensions, sizes))
        if DimensionContract.satisfies(record.get('constraints', []), env):
            found = True
            yield env
    if not found:
        raise ValueError("dimension constraints have no legal assignment")


def check_symbolic(program):
    from .ports import check_ports
    from .syntax import WgtError
    def walk(layers):
        for layer in layers:
            if layer.op in {'rearrange', 'reduce', 'flatten', 'mean', 'sum', 'transpose'}:
                raise ValueError(f"{layer.op} does not yet support bounded dimensions")
            yield layer
            yield from walk(layer.children)
    try:
        count = len(list(walk(program.layers)))
        # Keep the validation workload bounded independently of parameter sizes.
        domains = math.prod(b['max'] - b['min'] + 1 for b in program.dimensions.values())
        if domains * count > 200000:
            raise ValueError("symbolic validation exceeds 200000 node assignments")
        record = {'dimensions': program.dimensions, 'constraints': program.constraints,
                  'inputs': [{'name': p.name, 'shape': list(p.shape), 'shape_expr': list(p.shape_expr)} for p in program.inputs]}
        canonical = None
        for env in environments(record):
            candidate = copy.deepcopy(canonical or program)
            for port in candidate.inputs + candidate.outputs:
                port.shape = tuple(DimensionContract.evaluate(e, env) for e in port.shape_expr)
                if any(n < 0 for n in port.shape):
                    raise ValueError("symbolic port dimensions must remain nonnegative")
            check_ports(candidate)
            if canonical is not None:
                def emissions(layers):
                    return [(node.results['output']['dtype'],node.results['output']['shape'][1:]) for node in walk(layers) if node.op == 'scan_state']
                if emissions(candidate.layers) != emissions(canonical.layers):
                    raise ValueError('scan_state emitted sample shape/dtype must be invariant across bounded dimensions')
            if canonical is None:
                canonical = candidate
        program.layers = canonical.layers
        for port, original in zip(program.inputs + program.outputs, canonical.inputs + canonical.outputs):
            port.shape = original.shape
        program.returns = canonical.returns
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise WgtError(f"invalid bounded dimensions: {exc}", path=program.path) from None
