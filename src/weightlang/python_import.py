"""Static PyTorch architecture import. Source files are parsed, never executed."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path

from .syntax import Layer, PARAM_OPS, WgtError, output_contract, parse


@dataclass
class Module:
    name: str
    op: str = ""
    args: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    forward: ast.FunctionDef | None = None


class Converter:
    def __init__(self, path):
        self.path = str(path)
        self.aliases = {}
        self.classes = {}
        self.constants = {}
        self.layers = []
        self.used = set()
        self.depth = 0

    def fail(self, node, message):
        raise WgtError(message, getattr(node, "lineno", 1), self.path)

    def qualified(self, node):
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return self.qualified(node.value) + "." + node.attr
        return ""

    def value(self, node, env):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool, str, type(None)):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.value(n, env) for n in node.elts]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in env.get("self", {}):
                return env["self"][node.attr]
        if isinstance(node, ast.Subscript):
            value, index = self.value(node.value, env), self.value(node.slice, env)
            if isinstance(value, Module) and value.op in ("sequence", "module_list"):
                value = value.children
            if isinstance(value, list) and type(index) is int and -len(value) <= index < len(value):
                return value[index]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.value(node.operand, env)
            if type(value) in (int, float):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.FloorDiv: operator.floordiv,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
            }
            left, right = self.value(node.left, env), self.value(node.right, env)
            if type(left) in (int, float) and type(right) in (int, float) and type(node.op) in ops:
                try:
                    return ops[type(node.op)](left, right)
                except ArithmeticError:
                    self.fail(node, "invalid constant arithmetic")
        self.fail(node, "expected a literal, constructor parameter or constant expression")

    def bind(self, node, names, defaults, env):
        if len(node.args) > len(names) or any(isinstance(a, ast.Starred) for a in node.args):
            self.fail(node, "unsupported positional arguments")
        supplied = dict(zip(names, [self.value(a, env) for a in node.args]))
        for kw in node.keywords:
            if kw.arg not in names or kw.arg in supplied:
                self.fail(node, f"unknown or duplicate argument: {kw.arg}")
            supplied[kw.arg] = self.value(kw.value, env)
        values = {**defaults, **supplied}
        if set(values) != set(names):
            self.fail(node, "missing constructor arguments: " + ", ".join(set(names) - set(values)))
        return values

    def indices(self, node, env):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "range"
            or node.keywords
            or not 1 <= len(node.args) <= 3
        ):
            self.fail(node, "expected a constant range with at most 1024 iterations")
        args = [self.value(arg, env) for arg in node.args]
        if any(type(v) is not int for v in args) or (len(args) == 3 and args[2] == 0):
            self.fail(node, "range arguments must be integers with nonzero step")
        values = range(*args)
        if len(values) > 1024:
            self.fail(node, "static loop exceeds 1024 iterations")
        return values

    def check_binding(self, node, name):
        if name in self.aliases or name in self.classes or name in ("self", "range", "super"):
            self.fail(node, f"shadowing {name!r} is unsupported")

    def constructor(self, node, name, env):
        if not isinstance(node, ast.Call):
            # Aliases preserve object identity, so repeated parameter use is detected.
            value = self.value(node, env)
            if isinstance(value, Module):
                return value
            self.fail(node, "expected a module constructor")
        kind = self.qualified(node.func)
        if kind in self.classes:
            self.depth += 1
            if self.depth > 32:
                self.fail(node, "module nesting exceeds 32 (possibly recursive construction)")
            cls = self.classes[kind]
            methods = {}
            for item in cls.body:
                if self.docstring(item):
                    continue
                if (
                    not isinstance(item, ast.FunctionDef)
                    or item.name not in ("__init__", "forward")
                    or item.name in methods
                ):
                    self.fail(item, "classes support only __init__ and forward methods")
                if item.decorator_list:
                    self.fail(item, "decorated methods are unsupported")
                methods[item.name] = item
            if set(methods) != {"__init__", "forward"}:
                self.fail(cls, "class must define __init__ and forward")
            init = methods["__init__"]
            args = init.args
            if (
                args.posonlyargs
                or args.vararg
                or args.kwarg
                or args.kwonlyargs
                or not args.args
                or args.args[0].arg != "self"
            ):
                self.fail(init, "constructor requires self and ordinary named parameters")
            names = [a.arg for a in args.args[1:]]
            for parameter in names:
                self.check_binding(init, parameter)
            defaults = dict(
                zip(names[len(names) - len(args.defaults) :], [self.value(v, self.constants) for v in args.defaults])
            )
            local = {**self.constants, **self.bind(node, names, defaults, env), "self": {}}
            for item in init.body:
                if self.docstring(item):
                    continue
                if (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Call)
                    and ast.dump(item.value, include_attributes=False)
                    == ast.dump(ast.parse("super().__init__()").body[0].value, include_attributes=False)
                ):
                    continue
                if isinstance(item, ast.Assign) and len(item.targets) == 1:
                    target = item.targets[0]
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        if target.attr in local["self"]:
                            self.fail(item, "reassigned module attributes are unsupported")
                        value = (
                            self.constructor(item.value, name + "_" + target.attr, local)
                            if isinstance(item.value, ast.Call)
                            else self.value(item.value, local)
                        )
                        local["self"][target.attr] = value
                        continue
                    if isinstance(target, ast.Name):
                        self.check_binding(item, target.id)
                        local[target.id] = self.value(item.value, local)
                        continue
                self.fail(item, "unsupported constructor statement; use explicit module declarations")
            forward = methods["forward"]
            fa = forward.args
            if (
                fa.posonlyargs
                or fa.vararg
                or fa.kwarg
                or fa.kwonlyargs
                or fa.defaults
                or [a.arg for a in fa.args[:1]] != ["self"]
                or len(fa.args) != 2
            ):
                self.fail(forward, "forward must take exactly one tensor argument")
            self.check_binding(forward, fa.args[1].arg)
            self.depth -= 1
            return Module(name, attrs=local["self"], forward=forward)
        if kind in ("torch.nn.Sequential", "torch.nn.ModuleList"):
            if node.keywords:
                self.fail(node, "container keyword arguments are unsupported")
            children = node.args
            if kind.endswith("ModuleList"):
                if len(children) == 1 and isinstance(children[0], ast.ListComp):
                    comp = children[0]
                    if (
                        len(comp.generators) != 1
                        or comp.generators[0].ifs
                        or comp.generators[0].is_async
                        or not isinstance(comp.generators[0].target, ast.Name)
                    ):
                        self.fail(comp, "ModuleList comprehension requires one constant range without filters")
                    gen = comp.generators[0]
                    self.check_binding(comp, gen.target.id)
                    return Module(
                        name,
                        "module_list",
                        children=[
                            self.constructor(comp.elt, f"{name}_{i}", {**env, gen.target.id: value})
                            for i, value in enumerate(self.indices(gen.iter, env))
                        ],
                    )
                if len(children) != 1 or not isinstance(children[0], (ast.List, ast.Tuple)):
                    self.fail(node, "ModuleList requires a literal list or constant-range comprehension")
                children = children[0].elts
            return Module(
                name,
                "module_list" if kind.endswith("ModuleList") else "sequence",
                children=[self.constructor(c, f"{name}_{i}", env) for i, c in enumerate(children)],
            )
        if not kind.startswith("torch.nn."):
            self.fail(node, f"unsupported module constructor: {kind}")
        kind = kind.removeprefix("torch.nn.")
        schemas = {
            "Linear": (["in_features", "out_features", "bias"], {"bias": True}),
            "Conv2d": (
                [
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "stride",
                    "padding",
                    "dilation",
                    "groups",
                    "bias",
                    "padding_mode",
                ],
                dict(stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"),
            ),
            "MaxPool2d": (
                ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
                dict(stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False),
            ),
            "LayerNorm": (
                ["normalized_shape", "eps", "elementwise_affine", "bias"],
                dict(eps=1e-5, elementwise_affine=True, bias=True),
            ),
            "Dropout": (["p", "inplace"], dict(p=0.5, inplace=False)),
            "Flatten": (["start_dim", "end_dim"], dict(start_dim=1, end_dim=-1)),
            "Softmax": (["dim"], {}),
            "Embedding": (
                [
                    "num_embeddings",
                    "embedding_dim",
                    "padding_idx",
                    "max_norm",
                    "norm_type",
                    "scale_grad_by_freq",
                    "sparse",
                ],
                dict(padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False),
            ),
            "ReLU": (["inplace"], dict(inplace=False)),
            "SiLU": (["inplace"], dict(inplace=False)),
            "GELU": (["approximate"], dict(approximate="none")),
            "Tanh": ([], {}),
            "Sigmoid": ([], {}),
            "Identity": ([], {}),
        }
        if kind not in schemas:
            self.fail(node, f"unsupported Python layer: {kind}")
        a = self.bind(node, *schemas[kind], env)
        required = {
            "bias": True,
            "inplace": False,
            "dilation": 1,
            "groups": 1,
            "padding_mode": "zeros",
            "return_indices": False,
            "ceil_mode": False,
            "eps": 1e-5,
            "elementwise_affine": True,
            "approximate": "none",
            "padding_idx": None,
            "max_norm": None,
            "norm_type": 2.0,
            "scale_grad_by_freq": False,
            "sparse": False,
        }
        for key, expected in required.items():
            if (
                key in a
                and a[key] != expected
                and not (kind in ("Linear", "LayerNorm") and key in ("bias", "eps", "elementwise_affine"))
                and not (kind == "GELU" and key == "approximate" and a[key] == "tanh")
                and not (kind == "Embedding" and key == "padding_idx")
            ):
                self.fail(node, f"{kind} {key}={a[key]!r} is not representable by WeightLang")

        def square(key):
            v = a[key]
            if isinstance(v, list) and len(v) == 2 and v[0] == v[1]:
                v = v[0]
            if type(v) is not int:
                self.fail(node, f"{kind} {key} must be an integer or equal pair")
            return v

        if kind == "Linear":
            op, args = "linear", {"in": a["in_features"], "out": a["out_features"]}
        elif kind == "Conv2d":
            op, args = (
                "conv2d",
                {
                    "in": a["in_channels"],
                    "out": a["out_channels"],
                    "kernel": square("kernel_size"),
                    "stride": square("stride"),
                    "padding": square("padding"),
                },
            )
        elif kind == "MaxPool2d":
            if a["padding"] != 0:
                self.fail(node, "MaxPool2d padding is unsupported")
            a["stride"] = a["kernel_size"] if a["stride"] is None else a["stride"]
            op, args = "maxpool2d", {"kernel": square("kernel_size"), "stride": square("stride")}
        elif kind == "LayerNorm":
            dim = a["normalized_shape"]
            if isinstance(dim, list) and len(dim) == 1:
                dim = dim[0]
            if type(dim) is not int:
                self.fail(node, "LayerNorm supports only the final feature dimension")
            op, args = "layernorm", {"dim": dim}
        elif kind == "Dropout":
            op, args = "dropout", {"p": a["p"]}
        elif kind == "Flatten":
            if a != {"start_dim": 1, "end_dim": -1}:
                self.fail(node, "Flatten must preserve batch: start_dim=1, end_dim=-1")
            op, args = "flatten", {}
        elif kind == "Softmax":
            if a["dim"] != -1:
                self.fail(node, "Softmax supports dim=-1 only")
            op, args = "softmax", {}
        elif kind == "Embedding":
            op, args = "embedding", {"vocab": a["num_embeddings"], "dim": a["embedding_dim"]}
        else:
            op, args = kind.lower(), {}
        if kind in ("Linear", "LayerNorm"):
            if type(a["bias"]) is not bool:
                self.fail(node, "bias must be a boolean")
            if not a["bias"]:
                args["bias"] = 0
        if kind == "LayerNorm" and not a["elementwise_affine"]:
            args["affine"] = 0
        if kind == "Embedding" and a["padding_idx"] is not None:
            padding = a["padding_idx"]
            if type(padding) is not int or not -a["num_embeddings"] <= padding < a["num_embeddings"]:
                self.fail(node, "padding_idx must be inside embedding vocabulary")
            args["padding_idx"] = padding % a["num_embeddings"]
        if kind == "LayerNorm" and a["eps"] != 1e-5:
            args["eps"] = a["eps"]
        if kind == "GELU" and a["approximate"] == "tanh":
            args["approximate"] = 1
        return Module(name, op, args)

    @staticmethod
    def docstring(node):
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)

    def emit(self, node, op, args, refs, name="value"):
        if len(self.layers) >= 10000:
            self.fail(node, "converted graph exceeds 10000 nodes")
        name = f"n{len(self.layers)}_{name}"
        self.layers.append(Layer(name, op, args, getattr(node, "lineno", 1), self.path, inputs=refs))
        return name

    def call(self, module, ref, node):
        if module.forward:
            env = {**self.constants, "self": module.attrs, module.forward.args.args[1].arg: ref}
            result = self.statements(module.forward.body, env)
            if result is None:
                self.fail(module.forward, "forward must return one tensor")
            return result
        if module.op == "sequence":
            for child in module.children:
                ref = self.call(child, ref, node)
            return ref
        if module.op == "module_list":
            self.fail(node, "ModuleList must be iterated, not called")
        if module.op in PARAM_OPS and not (module.op == "layernorm" and module.args.get("affine") == 0):
            if id(module) in self.used:
                self.fail(node, "shared module called repeatedly; refusing to duplicate tied parameters")
            self.used.add(id(module))
        return self.emit(node, module.op, module.args, [ref], module.name)

    def expression(self, node, env):
        if isinstance(node, ast.Name) and isinstance(env.get(node.id), str):
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            return self.emit(
                node,
                "add" if isinstance(node.op, ast.Add) else "mul",
                {},
                [self.expression(node.left, env), self.expression(node.right, env)],
            )
        if isinstance(node, ast.Call):
            kind = self.qualified(node.func)
            if kind in ("torch.cat", "torch.concat"):
                if not 1 <= len(node.args) <= 2 or not isinstance(node.args[0], (ast.List, ast.Tuple)):
                    self.fail(node, "cat requires an explicit list of tensors")
                # Reuse strict binding to reject duplicate/unknown kwargs.
                opts = ast.Call(func=node.func, args=node.args[1:], keywords=node.keywords)
                a = self.bind(opts, ["dim"], {"dim": 0}, env)
                if a["dim"] != -1:
                    self.fail(node, "cat supports dim=-1 only")
                return self.emit(node, "concat", {}, [self.expression(n, env) for n in node.args[0].elts])
            functions = {
                "torch.relu": "ReLU",
                "torch.sigmoid": "Sigmoid",
                "torch.tanh": "Tanh",
                **{
                    "torch.nn.functional." + k: v
                    for k, v in dict(
                        relu="ReLU", sigmoid="Sigmoid", tanh="Tanh", silu="SiLU", gelu="GELU", softmax="Softmax"
                    ).items()
                },
            }
            if kind in functions or kind == "torch.flatten":
                if not node.args:
                    self.fail(node, "functional operation requires a tensor")
                opkind = functions.get(kind, "Flatten")
                clone = ast.copy_location(
                    ast.Call(func=ast.Name(id="torch.nn." + opkind), args=node.args[1:], keywords=node.keywords), node
                )
                # torch.flatten defaults to start_dim=0, unlike nn.Flatten.
                if (
                    kind == "torch.flatten"
                    and len(node.args) == 1
                    and not any(k.arg == "start_dim" for k in node.keywords)
                ):
                    self.fail(node, "torch.flatten requires start_dim=1 to preserve batch")
                return self.call(self.constructor(clone, opkind.lower(), env), self.expression(node.args[0], env), node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "flatten":
                clone = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="torch.flatten"), args=[node.func.value, *node.args], keywords=node.keywords
                    ),
                    node,
                )
                return self.expression(clone, env)
            module = self.value(node.func, env)
            if not isinstance(module, Module) or len(node.args) != 1 or node.keywords:
                self.fail(node, "module calls require exactly one tensor argument")
            return self.call(module, self.expression(node.args[0], env), node)
        self.fail(node, "unsupported forward expression; expected a module call, activation, add, mul or cat")

    def statements(self, body, env, in_loop=False):
        for index, stmt in enumerate(body):
            if self.docstring(stmt):
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                self.check_binding(stmt, stmt.targets[0].id)
                env[stmt.targets[0].id] = self.expression(stmt.value, env)
            elif isinstance(stmt, ast.Return) and not in_loop:
                if index != len(body) - 1:
                    self.fail(stmt, "statements after return are unsupported")
                return self.expression(stmt.value, env)
            elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and not stmt.orelse:
                self.check_binding(stmt, stmt.target.id)
                if isinstance(stmt.iter, ast.Call):
                    children = self.indices(stmt.iter, env)
                else:
                    modules = self.value(stmt.iter, env)
                    if not isinstance(modules, Module) or modules.op not in ("module_list", "sequence"):
                        self.fail(stmt, "loops support ModuleList/Sequential or constant range only")
                    children = modules.children
                for child in children:
                    env[stmt.target.id] = child
                    self.statements(stmt.body, env, in_loop=True)
            else:
                self.fail(stmt, "unsupported forward statement (dynamic control flow is not converted)")
        return None

    def convert(self, path, class_name=None, init_args=None, input_shape=None, input_dtype="f32"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        model = None
        for node in tree.body:
            if self.docstring(node):
                continue
            if isinstance(node, ast.Import):
                for item in node.names:
                    if item.name not in ("torch", "torch.nn", "torch.nn.functional"):
                        self.fail(node, "only torch imports are supported; input Python is never executed")
                    self.aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else "torch"
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module in ("torch", "torch.nn", "torch.nn.functional")
                and not node.level
            ):
                for item in node.names:
                    if item.name == "*":
                        self.fail(node, "wildcard imports are unsupported")
                    self.aliases[item.asname or item.name] = node.module + "." + item.name
            elif isinstance(node, ast.ClassDef):
                if (
                    node.decorator_list
                    or node.keywords
                    or len(node.bases) != 1
                    or self.qualified(node.bases[0]) != "torch.nn.Module"
                    or node.name in self.classes
                ):
                    self.fail(node, "expected an undecorated nn.Module class with no custom inheritance")
                self.classes[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.check_binding(node, node.targets[0].id)
                if node.targets[0].id in self.constants:
                    self.fail(node, "reassigned global constants are unsupported")
                if isinstance(node.value, ast.Call):
                    if model is not None:
                        self.fail(node, "multiple model instances are ambiguous; use a definitions-only file")
                    model = node
                else:
                    self.constants[node.targets[0].id] = self.value(node.value, self.constants)
            else:
                self.fail(
                    node, "unsupported top-level statement; provide model definitions without training/application code"
                )
        if class_name is not None or model is None:
            if class_name is None:
                if len(self.classes) != 1:
                    self.fail(tree, "select one model class with --class-name")
                class_name = next(iter(self.classes))
            if class_name not in self.classes:
                self.fail(tree, f"unknown model class: {class_name}")
            if not isinstance(init_args or {}, dict):
                self.fail(tree, "--init-args must be a JSON object")
            call = ast.Call(
                func=ast.Name(id=class_name),
                args=[],
                keywords=[
                    ast.keyword(arg=k, value=ast.parse(repr(v), mode="eval").body) for k, v in (init_args or {}).items()
                ],
            )
            module = self.constructor(call, class_name, self.constants)
            name = class_name
        else:
            if init_args:
                self.fail(model, "--init-args requires --class-name for files with a model instance")
            name = model.targets[0].id
            module = self.constructor(model.value, name, self.constants)
        result = self.call(module, "input", tree)
        if not self.layers or result != self.layers[-1].name:
            result = self.emit(tree, "identity", {}, [result], "output")
        if input_shape is None:
            weighted = next((layer for layer in self.layers if layer.op in PARAM_OPS), None)
            if weighted is None or weighted.op != "linear":
                self.fail(tree, "input dimensions cannot be inferred; supply --input-shape without batch")
            input_shape = (weighted.args["in"],)
        if (
            not input_shape
            or any(type(n) is not int or n <= 0 for n in input_shape)
            or input_dtype not in ("f32", "i64")
        ):
            self.fail(tree, "input shape must contain positive integers; dtype must be f32 or i64")
        dtype, shape = output_contract(self.layers, input_dtype, tuple(input_shape))

        def dims(values):
            return ",".join(map(str, values))

        lines = [
            "wgt 0.3",
            f"model {name}",
            "# 静态转换 Python 定义，未执行源码；没有导入权重，构建时重新初始化。",
            "# 输入形状不含 batch；Linear 作用于最后一维，卷积输入为 channel,height,width。",
            f"input x: {input_dtype}[{dims(input_shape)}]",
            f"output y: {dtype}[{dims(shape)}]",
            "# 每个参数层独立；分支显式引用输入，add/mul 不进行广播。",
        ]
        for layer in self.layers:
            if layer.op in ("add", "mul", "concat"):
                expr = f"{layer.op}({', '.join(layer.inputs)})"
            else:
                args = ", ".join(f"{k}={v}" for k, v in layer.args.items())
                expr = (f"{layer.op}({args})" if args else layer.op) + " from " + layer.inputs[0]
            lines.append(f"layer {layer.name}: {expr}")
        text = "\n".join([*lines, "init seed=0", ""])
        parse(text)
        return text


def python_architecture(path: Path, **options) -> str:
    return Converter(path).convert(path, **options)
