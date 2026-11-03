"""WGT initialization algorithms: closed syntax interpreted without Python exec."""
import ast
import math
import operator
import re
import shlex

import torch

from .syntax import WgtError

BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
       ast.Pow: operator.pow, ast.MatMult: operator.matmul}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
FUNCTIONS = {
    "zeros", "ones", "full", "eye", "arange", "linspace", "rand", "randn",
    "reshape", "transpose", "unsqueeze", "squeeze", "stack", "cat", "diag",
    "sin", "cos", "tanh", "exp", "log", "sqrt", "abs", "sign", "clamp",
    "sum", "mean", "norm", "where", "matmul", "qr", "svd", "solve",
    "minimum", "maximum", "range", "len", "min", "max", "dim",
}


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class LoopControl(Exception):
    def __init__(self, stop):
        self.stop = stop


def parse_source(text, path):
    """Lower only WGT declarations/let; AST is data, never compiled/executed."""
    lines = text.splitlines()
    code, declarations = [], []
    header = False
    body_allowed = False
    for number, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            code.append(raw)
            continue
        if not header:
            if shlex.split(stripped, comments=True) != ["wgt", "init", "1"]:
                raise WgtError("expected 'wgt init 1'", number, path)
            header = True
            code.append("")
        elif raw == raw.lstrip():
            if stripped.startswith("algorithm "):
                code.append(raw.replace("algorithm ", "def ", 1))
                body_allowed = True
            elif stripped.startswith("tensor "):
                match = re.fullmatch(r'tensor\s+"([^"\\]+)"\s*:\s*(?:#.*)?', stripped)
                if match:
                    name = f"_tensor_{len(declarations)}"
                    declarations.append((number, ["tensor", match[1]], name))
                    code.append(f"def {name}():")
                    body_allowed = True
                else:
                    declarations.append((number, shlex.split(raw, comments=True), None))
                    code.append("")
                    body_allowed = False
            else:
                raise WgtError("expected algorithm or tensor declaration", number, path)
        else:
            if not body_allowed:
                raise WgtError("unexpected indented initialization statement", number, path)
            code.append(re.sub(r"^(\s*)let\s+", r"\1", raw))
    if not header:
        raise WgtError("expected 'wgt init 1'", path=path)
    try:
        tree = ast.parse("\n".join(code), filename=path)
    except SyntaxError as exc:
        raise WgtError(f"invalid initialization algorithm: {exc.msg}", exc.lineno, path) from None
    library = AlgorithmLibrary(tree, path)
    return declarations, library


class AlgorithmLibrary:
    def __init__(self, tree, path):
        self.path = path
        self.functions = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in self.functions or node.name in FUNCTIONS:
                raise WgtError("expected unique algorithm name", node.lineno, path)
            args = node.args
            if (node.decorator_list or node.returns or args.defaults or args.kw_defaults or args.vararg
                    or args.kwarg or args.posonlyargs or args.kwonlyargs
                    or any(arg.annotation for arg in args.args)):
                raise WgtError("algorithm takes plain required positional parameters", node.lineno, path)
            if len({arg.arg for arg in args.args}) != len(args.args):
                raise WgtError("duplicate algorithm parameter", node.lineno, path)
            self.functions[node.name] = node
        allowed = (ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
                   ast.Return, ast.If, ast.For, ast.While, ast.Break, ast.Continue,
                   ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Slice,
                   ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp, ast.Call,
                   ast.Load, ast.Store, ast.And, ast.Or, *BIN, *CMP, *UNARY)
        def check_control(statements, loop=False):
            for statement in statements:
                if isinstance(statement, (ast.Break, ast.Continue)) and not loop:
                    raise WgtError("break/continue require an enclosing loop", statement.lineno, path)
                if isinstance(statement, (ast.For, ast.While)):
                    check_control(statement.body, True)
                elif isinstance(statement, ast.If):
                    check_control(statement.body, loop)
                    check_control(statement.orelse, loop)

        for function in self.functions.values():
            check_control(function.body)
            for node in ast.walk(function):
                if not isinstance(node, allowed) or isinstance(node, ast.FunctionDef) and node is not function:
                    raise WgtError(f"unsupported initialization syntax: {type(node).__name__}",
                                   getattr(node, "lineno", function.lineno), path)
                if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                        or node.func.id not in FUNCTIONS | self.functions.keys() or node.keywords):
                    raise WgtError("call a declared algorithm or tensor primitive with positional arguments", node.lineno, path)
                if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], (ast.Name, ast.Subscript))):
                    raise WgtError("assignment requires one variable or tensor index", node.lineno, path)
                if isinstance(node, ast.For) and (not isinstance(node.target, ast.Name) or node.orelse):
                    raise WgtError("for requires one variable and no else", node.lineno, path)
                if isinstance(node, ast.While) and node.orelse:
                    raise WgtError("while does not support else", node.lineno, path)
                if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool, type(None))):
                    raise WgtError("only numeric, boolean and None literals are supported", node.lineno, path)

    def generate(self, name, spec, seed):
        evaluator = Evaluator(self, spec, seed)
        with torch.no_grad():
            return evaluator.call(name, [])


class Evaluator:
    def __init__(self, library, spec, seed):
        self.library, self.spec = library, spec
        self.budget, self.depth = 100000, 0
        self.globals = {"shape": spec.shape, "rank": len(spec.shape), "seed": seed, "pi": math.pi}
        self.primitives = self.make_primitives()

    def make_primitives(self):
        dtype = self.spec.dtype
        def tensor(value):
            return value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=dtype, device="cpu")
        result = {
            "zeros": lambda shape: torch.zeros(shape, dtype=dtype, device="cpu"),
            "ones": lambda shape: torch.ones(shape, dtype=dtype, device="cpu"),
            "full": lambda shape, value: torch.full(shape, value, dtype=dtype, device="cpu"),
            "eye": lambda rows, cols=None: torch.eye(rows, rows if cols is None else cols, dtype=dtype, device="cpu"),
            "arange": lambda *args: torch.arange(*args, dtype=dtype, device="cpu"),
            "linspace": lambda start, stop, steps: torch.linspace(start, stop, steps, dtype=dtype, device="cpu"),
            "rand": lambda shape: torch.rand(shape, dtype=dtype, device="cpu"),
            "randn": lambda shape: torch.randn(shape, dtype=dtype, device="cpu"),
            "reshape": torch.reshape, "transpose": torch.transpose,
            "unsqueeze": torch.unsqueeze, "squeeze": lambda x, axis=None: torch.squeeze(x) if axis is None else torch.squeeze(x, axis),
            "stack": torch.stack, "cat": torch.cat, "diag": torch.diag,
            "clamp": torch.clamp, "where": torch.where, "matmul": torch.matmul,
            "sum": torch.sum, "mean": torch.mean,
            "norm": torch.linalg.vector_norm, "qr": lambda x: torch.linalg.qr(x, mode="reduced"),
            "svd": lambda x: torch.linalg.svd(x, full_matrices=False), "solve": torch.linalg.solve,
            "minimum": lambda a, b: torch.minimum(tensor(a), tensor(b)),
            "maximum": lambda a, b: torch.maximum(tensor(a), tensor(b)),
            "range": range, "len": len, "min": min, "max": max,
            "dim": lambda axis: self.spec.shape[axis],
        }
        for name in ("sin", "cos", "tanh", "exp", "log", "sqrt", "abs", "sign"):
            result[name] = lambda x, fn=getattr(torch, name): fn(tensor(x))
        return result

    def tick(self, node):
        self.budget -= 1
        if self.budget < 0:
            raise WgtError("initialization algorithm exceeded 100000 evaluation steps", node.lineno, self.library.path)

    def call(self, name, args):
        if name in self.primitives:
            return self.primitives[name](*args)
        fn = self.library.functions[name]
        if len(args) != len(fn.args.args):
            raise WgtError(f"algorithm {name} expects {len(fn.args.args)} arguments", fn.lineno, self.library.path)
        if self.depth >= 64:
            raise WgtError("initialization algorithm exceeded 64 call frames", fn.lineno, self.library.path)
        env = dict(self.globals)
        env.update((arg.arg, value) for arg, value in zip(fn.args.args, args))
        self.depth += 1
        try:
            self.block(fn.body, env)
        except Returned as result:
            return result.value
        finally:
            self.depth -= 1
        raise WgtError(f"algorithm {name} did not return a value", fn.lineno, self.library.path)

    def expression(self, node, env):
        self.tick(node)
        ev = lambda n: self.expression(n, env)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(ev(item) for item in node.elts)
        if isinstance(node, ast.Slice):
            return slice(*(ev(item) if item else None for item in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        if isinstance(node, ast.BinOp):
            return BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for index, (op, right) in enumerate(zip(node.ops, node.comparators)):
                value = ev(right)
                result = CMP[type(op)](left, value)
                if index < len(node.ops) - 1 and not bool(result):
                    return False
                left = value
            return result
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                result = ev(item)
                if isinstance(node.op, ast.And) and not bool(result) or isinstance(node.op, ast.Or) and bool(result):
                    return result
            return result
        if isinstance(node, ast.IfExp):
            return ev(node.body if bool(ev(node.test)) else node.orelse)
        if isinstance(node, ast.Call):
            return self.call(node.func.id, [ev(arg) for arg in node.args])
        raise WgtError("unsupported expression", node.lineno, self.library.path)

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, ast.Subscript):
            self.expression(target.value, env)[self.expression(target.slice, env)] = value
        else:
            raise WgtError("invalid assignment target", target.lineno, self.library.path)

    def block(self, statements, env):
        for node in statements:
            self.tick(node)
            try:
                if isinstance(node, ast.Assign):
                    self.assign(node.targets[0], self.expression(node.value, env), env)
                elif isinstance(node, ast.AugAssign):
                    self.assign(node.target, BIN[type(node.op)](self.expression(node.target, env),
                                                               self.expression(node.value, env)), env)
                elif isinstance(node, ast.Return):
                    raise Returned(self.expression(node.value, env) if node.value else None)
                elif isinstance(node, ast.If):
                    self.block(node.body if bool(self.expression(node.test, env)) else node.orelse, env)
                elif isinstance(node, (ast.For, ast.While)):
                    iterator = iter(self.expression(node.iter, env)) if isinstance(node, ast.For) else None
                    while True:
                        self.tick(node)
                        if iterator is None:
                            if not bool(self.expression(node.test, env)):
                                break
                        else:
                            try:
                                env[node.target.id] = next(iterator)
                            except StopIteration:
                                break
                        try:
                            self.block(node.body, env)
                        except LoopControl as control:
                            if control.stop:
                                break
                elif isinstance(node, (ast.Break, ast.Continue)):
                    raise LoopControl(isinstance(node, ast.Break))
            except (Returned, LoopControl, WgtError):
                raise
            except Exception as exc:
                raise WgtError(f"initialization algorithm failed: {exc}", node.lineno, self.library.path) from exc
