"""Parser and static checks for WeightLang source files (language versions 0.1 through 0.3).

The format is line oriented: one statement per line, ``#`` starts a comment.
Indented lines belong to the preceding ``data`` block (rows) or ``block``
definition (layers). See ``docs/language.md``.

A *model file* declares one model. A *library file* (e.g. the bundled
``std/nn.wgt``) contains only ``use`` and ``block`` statements.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import WgtError, label as diagnostic_label, suggestion
from .expressions import Expression, evaluate, logical_lines, parse_expression
from .primitives import SPECS as PRIMITIVES

LANGUAGE_VERSION = "0.3"
SUPPORTED_VERSIONS = ("0.1", "0.2", "0.3")
STD_DIR = Path(__file__).resolve().parent / "std"

# Parameter-free element-wise ops.
PARAM_FREE_OPS = {"relu", "tanh", "sigmoid", "gelu", "silu", "softmax", "identity"}
# Ops with arguments: name -> {arg: type}.
ARG_OPS: dict[str, dict[str, type]] = {
    "linear": {"in": int, "out": int},
    "rmsnorm": {"dim": int},
    "scale": {"factor": float},
    "rope_attention": {"dim": int, "heads": int, "kv_heads": int, "head_dim": int},
    "layernorm": {"dim": int},
    "dropout": {"p": float},
}
_V03_OPS = {
    "embedding",
    "position_embedding",
    "self_attention",
    "conv2d",
    "maxpool2d",
    "globalavgpool2d",
    "meanpool",
    "flatten",
    "last",
    "rearrange",
    "reduce",
}
ARG_OPS.update(
    {
        "embedding": {"vocab": int, "dim": int},
        "position_embedding": {"dim": int, "max_len": int},
        "self_attention": {"dim": int, "heads": int, "causal": int},
        "conv2d": {"in": int, "out": int, "kernel": int, "stride": int, "padding": int},
        "maxpool2d": {"kernel": int, "stride": int},
    }
)
OP_DEFAULTS = {"self_attention": {"causal": 0}, "conv2d": {"stride": 1, "padding": 0}}
PARAM_FREE_OPS |= {"globalavgpool2d", "meanpool", "flatten", "last"}
BUILTIN_OPS = PARAM_FREE_OPS | set(ARG_OPS) | {"rearrange", "reduce"}
PARAM_OPS = {
    "linear",
    "layernorm",
    "embedding",
    "position_embedding",
    "self_attention",
    "conv2d",
}  # ops that own trainable tensors
GRAPH_OPS = {"add", "mul", "concat", "mix", "where"}
LOOP_OPS = {"repeat", "recur", "scan", "scan_state", "ponder"}
STRUCTURAL_OPS = {"seq", "residual", "recur", "scan", "scan_state", "ponder", "route"}  # produced by block calls and residual(...)
LOSSES = {"mse", "bce", "cross_entropy"}
OPTIMIZERS = {"sgd", "adam", "adamw"}
DTYPES = {"f32", "i64", "bool"}

# Optional fields do not alter legacy serialized defaults when omitted.
OP_OPTIONALS = {
    "linear": {"bias": int, "tie_embedding": int},
    "layernorm": {"eps": float, "bias": int, "affine": int},
    "gelu": {"approximate": int},
    "rmsnorm": {"eps": float, "offset": float, "cast_after": int},
    "rope_attention": {"ports": int, "theta": float, "eps": float, "qk_norm": int,
                       "qkv_bias": int, "out_bias": int, "window": int, "dropout": float, "clip": float,
                       "yarn_factor": float, "yarn_original": int, "yarn_fast": float,
                       "yarn_slow": float, "attention_factor": float, "max_len": int, "flash": int, "norm_cast_after": int, "rope_fp32": int},
    "self_attention": {"dropout": float},
    "embedding": {"padding_idx": int},
}
_V03_OPS |= {"rmsnorm", "rope_attention", "scale"}
PARAM_OPS |= {"rmsnorm", "rope_attention"}

ARG_OPS["sparse_moe"] = {"dim": int, "hidden": int, "experts": int, "top_k": int}
OP_OPTIONALS["sparse_moe"] = {"ports": int,"normalize": int, "aux_coef": float, "activation": int}
BUILTIN_OPS.add("sparse_moe")
_V03_OPS.add("sparse_moe")
PARAM_OPS.add("sparse_moe")

ARG_OPS.update({op: {k: float if spec.get('types', {}).get(k) == 'float' else int
                     for k in spec['required']} for op, spec in PRIMITIVES.items()})
OP_DEFAULTS.update({op: spec['defaults'] for op, spec in PRIMITIVES.items()})
BUILTIN_OPS.update(PRIMITIVES)
_V03_OPS.update(PRIMITIVES)

ARG_OPS['conv1d'] = {k: int for k in ('in','out','kernel','stride','padding','dilation','groups','bias','causal')}
OP_DEFAULTS['conv1d'] = dict(stride=1,padding=0,dilation=1,groups=1,bias=1,causal=0)
BUILTIN_OPS.add('conv1d')
PARAM_OPS.add('conv1d')
_V03_OPS.add('conv1d')

# Ops that need `wgt 0.2`.
_V02_OPS = {"gelu", "silu", "softmax", "identity", "layernorm", "dropout"}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<string>"[^"\\]*")
  | (?P<arrow>->)
  | (?P<number>-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.]))
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<word>[A-Za-z0-9_]+)
  | (?P<punct>==|!=|<=|>=|<|>|//|[:=\[\](),./+*%\-])
    """,
    re.VERBOSE,
)




Value = int | float | str | Expression  # str = reference to a block parameter


@dataclass
class Port:
    name: str
    dtype: str
    shape: tuple[int, ...]
    line: int
    axes: tuple[str | None, ...] = ()
    shape_expr: tuple = ()


@dataclass
class Spec:
    """An unresolved layer expression: ``op``, ``op(k=v)``, ``ns.block(k=v)``, ``residual(<spec>)``."""

    op: str
    ns: str | None
    args: dict[str, Value]
    inner: Spec | None
    line: int
    path: str | None
    inputs: list[str] = field(default_factory=list)
    chain: list[Spec] = field(default_factory=list)
    input_spans: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class Layer:
    """A resolved layer. ``seq`` and ``residual`` layers have children."""

    name: str
    op: str
    args: dict[str, object]
    line: int
    path: str | None = None
    children: list[Layer] = field(default_factory=list)
    block: str | None = None  # e.g. "std/nn.mlp" when produced by a block call
    input_spec: dict | None = None
    output_spec: dict | None = None
    inputs: list[str] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    input_spans: list[tuple[int, int, int]] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    returns: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "op": self.op, "args": dict(self.args)}
        if self.op in STRUCTURAL_OPS:
            d["layers"] = [c.to_dict() for c in self.children]
        if self.returns:
            d["returns"] = self.returns
        if self.results:
            d["results"] = self.results
        if self.inputs:
            d["inputs"] = list(self.inputs)
        if self.block:
            d["block"] = self.block
        if self.input_spec is not None:
            d["input"] = self.input_spec
            d["output"] = self.output_spec
        return d

    @property
    def has_params(self) -> bool:
        return self.op in PARAM_OPS or any(c.has_params for c in self.children)


@dataclass
class Block:
    name: str
    params: list[str]
    body: list[tuple[str, Spec]]
    line: int
    path: str | None
    scope: Scope | None = None
    defaults: dict[str, Value] = field(default_factory=dict)
    returns: dict = field(default_factory=dict)


@dataclass
class Scope:
    """A file's namespace: its blocks and the libraries it uses."""

    name: str  # "std/nn", a library path as written, or "<model>"
    path: str | None
    blocks: dict[str, Block] = field(default_factory=dict)
    uses: dict[str, Scope] = field(default_factory=dict)
    constants: dict[str, int | float] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    inputs: list[list[float]]
    targets: list[list[float]]
    line: int
    file: str | None = None
    sha256: str | None = None
    skip: int = 0
    license: str | None = None
    source: str | None = None
    format: str | None = None
    dataset: str | None = None
    python: str | None = None
    entrypoint: str | None = None
    loaded: bool = False
    loader_record: dict | None = None


@dataclass
class ImportStmt:
    target: str  # dotted layer path in this model
    source: str  # artifact path or package path
    source_layer: str  # dotted layer path in the source artifact
    sha256: str | None
    freeze: bool
    line: int


@dataclass
class FreezeStmt:
    target: str
    frozen: bool
    line: int


@dataclass
class TrainStmt:
    data: str
    loss: str
    optimizer: str | None
    lr: float
    epochs: int
    line: int
    batch: int | None = None
    validate: str | None = None
    weight_decay: float = 0.0
    trainer: str = "backprop"
    python: str | None = None
    entrypoint: str | None = None


@dataclass
class Program:
    version: str
    model: str
    input: Port
    output: Port
    layers: list[Layer]
    datasets: dict[str, Dataset]
    seed: int
    initializer: dict[str, str] = field(default_factory=dict)
    imports: list[ImportStmt] = field(default_factory=list)
    train: TrainStmt | None = None
    path: str | None = None
    uses: dict[str, str] = field(default_factory=dict)  # alias -> library name

    freezing: list[FreezeStmt] = field(default_factory=list)
    inherit: ImportStmt | None = None
    derivation: object | None = None
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    returns: dict[str, str] = field(default_factory=dict)
    dimensions: dict = field(default_factory=dict)
    constraints: list = field(default_factory=list)

    weight_types: dict = field(default_factory=dict)
    weight_assignments: list = field(default_factory=list)
    adapters: dict = field(default_factory=dict)
    config_schema: dict = field(default_factory=dict)
    config_values: dict = field(default_factory=dict)
    parameter_ids: dict[str, str] = field(default_factory=dict)
    parameter_shares: dict[str, str] = field(default_factory=dict)

    def find(self, dotted: str) -> Layer | None:
        return find_layer(self.layers, dotted)


def find_layer(layers: list[Layer], dotted: str) -> Layer | None:
    head, _, rest = dotted.partition(".")
    for layer in layers:
        if layer.name == head:
            return layer if not rest else find_layer(layer.children, rest)
    return None


# ---------------------------------------------------------------- tokens


class _Tokens(list):
    def __init__(self):
        super().__init__()
        self.spans = []


def _tokenize(text: str, line: int, path: str | None) -> list[tuple[str, str]]:
    tokens = _Tokens()
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise WgtError(f"unexpected character {text[pos]!r}", line, path, code="E0100", labels=[diagnostic_label(path, line, column=pos + 1, end_column=pos + 2)])
        kind = m.lastgroup
        assert kind is not None
        if kind != "ws":
            tokens.append((kind, m.group()))
            tokens.spans.append((line, m.start() + 1, m.end() + 1))
        pos = m.end()
    return tokens


def _strip_comment(raw: str) -> str:
    # Keep '#' inside quoted strings (paths may contain it).
    out = []
    in_string = False
    for ch in raw:
        if ch == '"':
            in_string = not in_string
        if ch == "#" and not in_string:
            break
        out.append(ch)
    return "".join(out).rstrip()


class _Cursor:
    def __init__(self, tokens: list[tuple[str, str]], line: int, path: str | None):
        self.tokens = tokens
        self.version: str | None = None
        self.i = 0
        self.line = line
        self.path = path

    def error(self, message: str, index=None) -> WgtError:
        spans = getattr(self.tokens, "spans", [])
        index = max(0, self.i - 1) if index is None else index
        if spans:
            line, column, end = spans[min(index, len(spans) - 1)]
            if index >= len(spans):
                column, end = end, end + 1
            return WgtError(message, line, self.path, code="E0100",
                            labels=[diagnostic_label(self.path, line, column=column, end_column=end)])
        return WgtError(message, self.line, self.path, code="E0100")

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self, what: str) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise self.error(f"expected {what}, found end of line", self.i)
        self.i += 1
        return tok

    def expect(self, kind: str, value: str | None = None, what: str | None = None) -> str:
        desc = what or (repr(value) if value else kind)
        tok = self.next(desc)
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise self.error(f"expected {desc}, found {tok[1]!r}")
        return tok[1]

    def ident(self, what: str) -> str:
        return self.expect("ident", what=what)

    def dotted(self, what: str) -> str:
        parts = [self.ident(what)]
        while self.peek() == ("punct", "."):
            self.next(".")
            parts.append(self.ident(what))
        return ".".join(parts)

    def integer(self, what: str) -> int:
        text = self.expect("number", what=what)
        if not re.fullmatch(r"\d+", text):
            raise self.error(f"{what} must be a non-negative integer, found {text!r}")
        return int(text)

    def done(self) -> None:
        tok = self.peek()
        if tok is not None:
            raise self.error(f"unexpected {tok[1]!r} at end of statement", self.i)

    def at_end(self) -> bool:
        return self.peek() is None


def _number(text: str) -> int | float:
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def _parse_port(cur: _Cursor, constants: dict[str, int | float], dimensions=None) -> tuple[str, str, tuple[int, ...], tuple[str | None, ...], tuple]:
    name = cur.ident("port name")
    cur.expect("punct", ":")
    dtype = cur.ident("dtype")
    if dtype not in DTYPES:
        supported = "f32, i64 and bool" if cur.version == "0.3" else "f32"
        raise cur.error(f"unsupported dtype {dtype!r}; WeightLang supports only {supported}")
    cur.expect("punct", "[")

    expressions = []

    def dimension() -> int:
        if cur.version != "0.3":
            return cur.integer("dimension")
        expr = parse_expression(cur)
        if dimensions:
            from .symbolic import encode
            expressions.append(encode(expr, constants, dimensions, cur.error))
            return 1  # Concrete representative is selected by exhaustive checking.
        value = evaluate(expr, constants, cur.error)
        if not isinstance(value, int):
            raise cur.error("dimension must be an integer")
        return value

    axes = []

    def named_dimension():
        axis = None
        if cur.peek() and cur.peek()[0] == "ident" and cur.tokens[cur.i + 1:cur.i + 2] == [("punct", "=")]:
            if cur.version != "0.3":
                raise cur.error("named axes require 'wgt 0.3'")
            axis = cur.ident("axis name")
            cur.expect("punct", "=")
            if axis in axes:
                raise cur.error(f"duplicate axis {axis!r}")
        axes.append(axis)
        return dimension()

    dims = [] if cur.version == "0.3" and cur.peek() == ("punct", "]") else [named_dimension()]
    while cur.peek() == ("punct", ","):
        cur.next(",")
        dims.append(named_dimension())
    cur.expect("punct", "]")
    if any(dim < (0 if cur.version == "0.3" else 1) for dim in dims):
        raise cur.error("dimensions must be nonnegative in wgt 0.3")
    return name, dtype, tuple(dims), tuple(axes) if any(axes) else (), tuple(expressions)


def _parse_kv(cur: _Cursor) -> dict[str, str]:
    result: dict[str, str] = {}
    while not cur.at_end():
        key = cur.ident("key")
        cur.expect("punct", "=")
        kind, value = cur.next(f"value for {key}")
        if kind == "string":
            value = value[1:-1]
        elif kind not in ("ident", "number", "word"):
            raise cur.error(f"invalid value {value!r} for {key}")
        if key in result:
            raise cur.error(f"duplicate key {key!r}")
        result[key] = value
    return result


def _parse_vector(cur: _Cursor) -> list[float]:
    cur.expect("punct", "[")
    values: list[float] = []
    while True:
        values.append(float(cur.expect("number", what="number")))
        tok = cur.next("',' or ']'")
        if tok == ("punct", "]"):
            return values
        if tok != ("punct", ","):
            raise cur.error(f"expected ',' or ']', found {tok[1]!r}")


def _parse_spec(cur: _Cursor) -> Spec:
    first = _parse_atom(cur)
    chain = [first]
    while cur.peek() == ("arrow", "->"):
        if cur.version != "0.3":
            raise cur.error("pipelines require 'wgt 0.3'")
        cur.next("->")
        chain.append(_parse_atom(cur))
    if len(chain) == 1:
        return first
    if any(item.inputs for item in chain):
        raise cur.error("merge operations need named layers, not an inline pipeline")
    return Spec("seq", None, {}, None, first.line, first.path, chain=chain)


def _parse_atom(cur: _Cursor) -> Spec:
    line = cur.line
    name = cur.ident("layer operation or block")
    if cur.peek() in {("punct", "+"), ("punct", "*")}:
        if cur.version != "0.3":
            raise cur.error("tensor arithmetic requires 'wgt 0.3'")
        spans = [cur.tokens.spans[cur.i - 1]]
        operator = cur.next("tensor operator")[1]
        other = cur.dotted("tensor name")
        spans.append(cur.tokens.spans[cur.i - 1])
        if cur.peek() in {("punct", "+"), ("punct", "*")}:
            raise cur.error("tensor arithmetic takes two names; use separate named layers for chained expressions")
        return Spec("add" if operator == "+" else "mul", None, {}, None,
                    line, cur.path, [name, other], input_spans=spans)
    ns = None
    if cur.peek() == ("punct", "."):
        cur.next(".")
        ns, name = name, cur.ident("block name")
    args: dict[str, Value] = {}
    inner = None
    inputs = []
    input_spans = []
    if ns is None and name in GRAPH_OPS | LOOP_OPS | {"route"} and cur.version != "0.3":
        raise cur.error(f"{name} requires 'wgt 0.3'")
    if cur.peek() == ("punct", "("):
        cur.next("(")
        if ns is None and name in {"mean", "sum", "transpose"}:
            if cur.version != "0.3":
                raise cur.error("named axis shorthand requires 'wgt 0.3'")
            if name == "transpose":
                names = [cur.ident("axis name")]
                while cur.peek() == ("punct", ","):
                    cur.next(",")
                    if cur.peek() == ("punct", ")"):
                        break
                    names.append(cur.ident("axis name"))
                args["axes"] = " ".join(names)
            else:
                cur.expect("ident", "axis")
                cur.expect("punct", "=")
                args["axis"] = cur.ident("axis name")
                if cur.peek() == ("punct", ","):
                    cur.next(",")
        elif ns is None and name == "linear" and cur.peek() != ("punct", ")") and cur.tokens[cur.i + 1:cur.i + 2] != [("punct", "=")]:
            if cur.version != "0.3":
                raise cur.error("positional linear requires 'wgt 0.3'")
            args["out"] = parse_expression(cur)
            if cur.peek() == ("punct", ","):
                cur.next(",")
            if cur.peek() != ("punct", ")"):
                raise cur.error("linear(out) takes one positional argument; use named arguments for other options")
        elif ns is None and name in {"rearrange", "reduce"}:
            if cur.version != "0.3":
                raise cur.error(f"{name} requires 'wgt 0.3'")
            args["pattern"] = cur.expect("string", what="quoted axis pattern")[1:-1]
            while cur.peek() == ("punct", ","):
                cur.next(",")
                if cur.peek() == ("punct", ")"):
                    break
                key = cur.ident("axis size or reduction op")
                cur.expect("punct", "=")
                if key in args:
                    raise cur.error(f"duplicate argument {key!r}")
                args[key] = cur.ident("mean or sum") if name == "reduce" and key == "op" else parse_expression(cur)
            if name == "reduce" and args.get("op") not in {"mean", "sum"}:
                raise cur.error("reduce requires op=mean or op=sum")
        elif ns is None and name in GRAPH_OPS:
            while cur.peek() != ("punct", ")"):
                inputs.append(cur.dotted("tensor name"))
                input_spans.append(cur.tokens.spans[cur.i - 1])
                if cur.peek() != ("punct", ","):
                    break
                cur.next(",")
        elif ns is None and name == "route":
            branches = [_parse_spec(cur)]
            while cur.peek() == ("punct", ","):
                cur.next(",")
                branches.append(_parse_spec(cur))
            cur.expect("punct", ")")
            if len(branches) < 2:
                raise cur.error("route needs at least two branches")
            return Spec("route", None, {}, None, line, cur.path, chain=branches)
        elif ns is None and name == "scan_state":
            inner = _parse_spec(cur)
        elif ns is None and name in LOOP_OPS:
            key = "hidden" if name == "scan" else "steps"
            if name == "scan":
                cur.expect("ident", "hidden")
                cur.expect("punct", "=")
            args[key] = parse_expression(cur)
            cur.expect("punct", ",")
            inner = _parse_spec(cur)
            while cur.peek() == ("punct", ","):
                cur.next(",")
                if cur.peek() == ("punct", ")"):
                    break
                option = cur.ident("loop option")
                if name not in {"recur", "ponder"} or option not in ({"until", "compact"} if name == "recur" else {"until"}) or option in args:
                    raise cur.error("invalid or duplicate loop option")
                cur.expect("punct", "=")
                args[option] = cur.ident("stop probability layer") if option == "until" else parse_expression(cur)
        elif name == "residual" and ns is None:
            inner = _parse_spec(cur)
        else:
            while cur.peek() != ("punct", ")"):
                key = cur.ident("argument name")
                cur.expect("punct", "=")
                value = parse_expression(cur)
                if key in args:
                    raise cur.error(f"duplicate argument {key!r}")
                args[key] = value
                if cur.peek() == ("punct", ","):
                    cur.next(",")
        cur.expect("punct", ")")
    elif name == "residual" and ns is None:
        raise cur.error("residual needs a block: residual(<block>(...))")
    if ns is None and name == "linear" and "in" not in args and cur.version != "0.3":
        raise cur.error("linear requires in and out; inferred input width requires 'wgt 0.3'")
    if ns is None and cur.version != "0.3" and set(args) & set(OP_OPTIONALS.get(name, {})):
        raise cur.error("extended operator options require wgt 0.3")
    spec = Spec(name, ns, args, inner, line, cur.path, inputs, input_spans=input_spans)
    return spec


def _parse_layer_spec(cur):
    spec = _parse_spec(cur)
    if cur.peek() == ("ident", "from"):
        if cur.version != "0.3":
            raise cur.error("tensor references require 'wgt 0.3'")
        if spec.inputs:
            raise cur.error("merge operations already specify their inputs")
        cur.next("from")
        spec.inputs = [cur.dotted("tensor name")]
        spec.input_spans = [cur.tokens.spans[cur.i - 1]]
        while cur.peek() == ("punct", ","):
            cur.next(",")
            spec.inputs.append(cur.dotted("tensor name"))
            spec.input_spans.append(cur.tokens.spans[cur.i - 1])
    return spec


# ---------------------------------------------------------------- files


class _FileParser:
    """Parses one file into a Scope plus, for model files, the model statements."""

    def __init__(self, source: str, path: str | None, loader: _Loader, *, library: bool, config=None):
        self.source = source
        self.path = path
        self.loader = loader
        self.library = library
        self.scope = Scope("<model>" if not library else (path or "<library>"), path)
        self.version: str | None = None
        self.model: str | None = None
        self.ports: dict[str, Port] = {}
        self.named_ports = {"input": [], "output": []}
        self.returns = {}
        self.dimensions = {}
        self.constraints = []
        self.weight_types = {}
        self.weight_assignments = []
        self.adapters = {}
        self.config_overrides = dict(config or {})
        self.config_schema = {}
        self.config_values = {}
        self.parameter_ids = {}
        self.parameter_shares = {}
        self.layer_specs: list[tuple[str, Spec]] = []
        self.datasets: dict[str, Dataset] = {}
        self.seed: int | None = None
        self.initializer: dict[str, str] = {}
        self.imports: list[ImportStmt] = []
        self.train: TrainStmt | None = None
        self.freezing: list[FreezeStmt] = []
        self.inherit: ImportStmt | None = None
        self.uses: dict[str, str] = {}

    def cursor(self, text, lineno):
        positions = []
        for index, raw in enumerate(self.source.splitlines()[lineno - 1:], lineno):
            segment = _strip_comment(raw)
            offset = 0 if index == lineno else len(segment) - len(segment.lstrip())
            segment = segment if index == lineno else segment.strip()
            if positions:
                positions.append((index, offset + 1))
            positions.extend((index, offset + col + 1) for col in range(len(segment)))
            if len(positions) >= len(text):
                break
        try:
            tokens = _tokenize(text, lineno, self.path)
        except WgtError as exc:
            if exc.labels:
                index = exc.labels[0]["column"] - 1
                if index < len(positions):
                    line, column = positions[index]
                    exc.line = line
                    exc.labels = [diagnostic_label(self.path, line, column=column, end_column=column + 1)]
            raise
        tokens.spans = [(positions[start - 1][0], positions[start - 1][1],
                         positions[end - 2][1] + 1) for _, start, end in tokens.spans]
        return _Cursor(tokens, lineno, self.path)

    def need_v02(self, what: str, line: int) -> None:
        if self.version == "0.1":
            raise WgtError(f"{what} requires 'wgt 0.2'", line, self.path)

    def run(self) -> None:
        open_data: Dataset | None = None
        open_block: Block | None = None
        for lineno, text, multiline in logical_lines(
            self.source, _strip_comment, lambda message, line: WgtError(message, line, self.path)
        ):
            if multiline and self.version != "0.3":
                raise WgtError("multiline statements require 'wgt 0.3'", lineno, self.path)
            if not text.strip():
                continue
            if text[0] in " \t":
                cur = self.cursor(text, lineno)
                cur.version = self.version
                if open_data is not None:
                    x = _parse_vector(cur)
                    cur.expect("arrow", what="'->'")
                    y = _parse_vector(cur)
                    cur.done()
                    open_data.inputs.append(x)
                    open_data.targets.append(y)
                elif open_block is not None:
                    if cur.peek() == ('ident','return'):
                        if self.version != '0.3' or open_block.returns:
                            raise cur.error('block return requires 0.3 and must occur once')
                        cur.next('return')
                        while not cur.at_end():
                            field = cur.ident('result name')
                            cur.expect('punct','=')
                            ref = cur.dotted('result tensor')
                            if field in open_block.returns:
                                raise cur.error('duplicate block result')
                            open_block.returns[field] = ref
                        if 'output' not in open_block.returns:
                            raise cur.error('block return requires output=<tensor>')
                        continue
                    if open_block.returns:
                        raise cur.error('block return must be the final statement')
                    cur.expect("ident", "layer", what="'layer' or 'return'")
                    name = cur.ident("layer name")
                    cur.expect("punct", ":")
                    spec = _parse_layer_spec(cur)
                    cur.done()
                    if any(n == name for n, _ in open_block.body):
                        raise cur.error(f"duplicate layer name {name!r} in block {open_block.name!r}")
                    open_block.body.append((name, spec))
                else:
                    raise WgtError(
                        "unexpected indentation (only data rows and block layers are indented)", lineno, self.path
                    )
                continue
            if open_block is not None and not open_block.body:
                raise WgtError(f"block {open_block.name!r} has no layers", open_block.line, self.path)
            open_data = None
            open_block = None
            cur = self.cursor(text, lineno)
            cur.version = self.version
            keyword = cur.ident("statement keyword")
            if self.version is None and keyword != "wgt":
                raise cur.error("file must start with a version header: wgt 0.2")
            if getattr(self, "data_only", False) and keyword not in ("wgt", "data"):
                raise cur.error("data files only allow wgt and data statements")
            handler = getattr(self, f"_stmt_{keyword}", None)
            if handler is None:
                raise cur.error(f"unknown statement {keyword!r}")
            std_import = keyword == "import" and cur.tokens[cur.i:cur.i + 2] == [("ident", "std"), ("punct", "/")]
            if self.library and keyword not in ("wgt", "use", "block", "const") and not std_import:
                raise cur.error(f"{keyword!r} is not allowed in a library file (only use, block and const)")
            result = handler(cur, lineno)
            if isinstance(result, Dataset) and not (result.file or result.python or result.entrypoint):
                open_data = result
            elif isinstance(result, Block):
                open_block = result
        if open_block is not None and not open_block.body:
            raise WgtError(f"block {open_block.name!r} has no layers", open_block.line, self.path)
        if self.version is None:
            raise WgtError("empty file; expected header: wgt 0.2", None, self.path)

    # -- statements

    def _stmt_wgt(self, cur: _Cursor, line: int) -> None:
        if self.version is not None:
            raise cur.error("duplicate wgt header")
        version = cur.expect("number", what="language version")
        cur.done()
        if version not in SUPPORTED_VERSIONS:
            raise cur.error(
                f"unsupported language version {version}; this weightc supports {', '.join(SUPPORTED_VERSIONS)}"
            )
        if self.library and version == "0.1":
            raise cur.error("library files require 'wgt 0.2'")
        self.version = version

    def _stmt_wtype(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("wtype requires wgt 0.3")
        name = cur.ident("weight type name")
        cur.expect("punct", "=")
        cur.expect("punct", "[")
        values = []
        while True:
            values.append(evaluate(parse_expression(cur), self.scope.constants, cur.error))
            token = cur.next("',' or ']'")
            if token == ("punct", "]"):
                break
            if token != ("punct", ","):
                raise cur.error("wtype values must be separated by commas")
        cur.done()
        if name in self.weight_types:
            raise cur.error("duplicate wtype")
        from .weight_types import validate_types
        try:
            self.weight_types.update(validate_types({name: values}))
        except ValueError as exc:
            raise cur.error(str(exc)) from None

    def _stmt_weight(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("weight requires wgt 0.3")
        target = cur.dotted("parameter path")
        cur.expect("punct", ":")
        spec = {"target": target, "type": cur.ident("weight type name")}
        while not cur.at_end():
            key = cur.ident("slice option")
            if key not in {"axis", "start", "stop"} or key in spec:
                raise cur.error("weight slice options are axis/start/stop, each once")
            cur.expect("punct", "=")
            spec[key] = evaluate(parse_expression(cur), self.scope.constants, cur.error)
        self.weight_assignments.append(spec)

    def _stmt_adapter(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("adapter requires wgt 0.3")
        name = cur.ident("adapter name")
        cur.expect("punct", ":")
        method = cur.ident("adapter method")
        if method not in ("lora", "dora", "pissa", "qlora"):
            raise cur.error("unknown adapter method")
        args = {'freeze':1}
        seen = set()
        while not cur.at_end():
            key = cur.ident("adapter option")
            cur.expect("punct", "=")
            if key in seen:
                raise cur.error("duplicate adapter option")
            seen.add(key)
            args[key] = cur.dotted("target layer") if key == 'target' else evaluate(parse_expression(cur),self.scope.constants,cur.error)
        if name in self.adapters or set(args) != {'target','rank','alpha','freeze'}:
            raise cur.error("adapter needs a unique name, target/rank/alpha and optional freeze=1")
        if method != "lora":
            args["method"] = method
        self.adapters[name] = args

    def _stmt_config(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3" or self.ports or self.layer_specs:
            raise cur.error("config requires wgt 0.3 and must precede ports and layers")
        name = cur.ident("configuration name")
        cur.expect("punct", ":")
        kind = cur.ident("configuration type")
        options = _parse_kv(cur)
        if name in self.scope.constants or name in self.dimensions:
            raise cur.error("duplicate configuration/constant/dimension")
        from .configuration import validate
        try:
            default = options['default']
            value = {'int':int,'float':float,'bool':lambda v:v == 'true'}[kind](default)
            validate(name,kind,options,value)
            value = validate(name,kind,options,self.config_overrides.get(name,value))
        except (ValueError,KeyError,OverflowError) as exc:
            raise cur.error(f"invalid config {name}: {exc}") from None
        self.config_schema[name] = dict(type=kind,**options,line=line,end_line=cur.tokens.spans[-1][0])
        self.config_values[name] = value
        self.scope.constants[name] = int(value) if type(value) is bool else value

    def _stmt_parameter(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("parameter requires wgt 0.3")
        name = cur.ident("parameter identity")
        cur.expect("punct", "=")
        path = cur.dotted("parameter path")
        cur.done()
        if name in self.parameter_ids:
            raise cur.error("duplicate parameter identity")
        self.parameter_ids[name] = path

    def _stmt_share(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("share requires wgt 0.3")
        path = cur.dotted("parameter destination")
        cur.expect("punct", "=")
        name = cur.ident("parameter identity")
        cur.done()
        if path in self.parameter_shares:
            raise cur.error("duplicate shared destination")
        self.parameter_shares[path] = name

    def _stmt_const(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("const requires 'wgt 0.3'")
        name = cur.ident("constant name")
        cur.expect("punct", "=")
        value = parse_expression(cur)
        cur.done()
        if name in self.scope.constants or name in self.dimensions:
            raise cur.error(f"duplicate constant/dimension {name!r}")
        self.scope.constants[name] = evaluate(value, self.scope.constants, cur.error)

    def _stmt_dim(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3" or self.ports:
            raise cur.error("dim requires wgt 0.3 and must precede all ports")
        name = cur.ident("dimension name")
        options = _parse_kv(cur)
        if set(options) != {"min", "max"} or any(not re.fullmatch(r"[0-9]+", v) for v in options.values()):
            raise cur.error("dim requires min=<nonnegative integer> max=<nonnegative integer>")
        if name in self.scope.constants or name in self.dimensions:
            raise cur.error(f"duplicate constant/dimension {name!r}")
        bounds = {k: int(v) for k, v in options.items()}
        if not 0 <= bounds["min"] <= bounds["max"] <= 2147483647:
            raise cur.error("dimension bounds require 0 <= min <= max <= 2147483647")
        self.dimensions[name] = bounds

    def _stmt_require(self, cur: _Cursor, line: int) -> None:
        from .symbolic import encode
        if self.version != "0.3" or not (self.dimensions or self.scope.constants):
            raise cur.error("require needs declared dimensions or constants in wgt 0.3")
        left = encode(parse_expression(cur), self.scope.constants, self.dimensions, cur.error)
        comparison = cur.expect("punct", what="integer comparison")
        if comparison not in ("==", "!=", "<", "<=", ">", ">="):
            raise cur.error("require expects == != < <= > >=")
        right = encode(parse_expression(cur), self.scope.constants, self.dimensions, cur.error)
        cur.done()
        if self.dimensions:
            self.constraints.append([comparison, left, right])
        else:
            from .symbolic import DimensionContract
            try:
                valid = DimensionContract.satisfies([[comparison,left,right]],{})
            except ValueError as exc:
                raise cur.error(str(exc)) from None
            if not valid:
                raise cur.error("configuration integer constraint failed")

    def _stmt_model(self, cur: _Cursor, line: int) -> None:
        if self.model is not None:
            raise cur.error("only one model per file is supported")
        self.model = cur.ident("model name")
        cur.done()

    def _stmt_input(self, cur: _Cursor, line: int) -> None:
        self._port("input", cur, line)

    def _stmt_output(self, cur: _Cursor, line: int) -> None:
        self._port("output", cur, line)

    def _port(self, kind: str, cur: _Cursor, line: int) -> None:
        if kind in self.ports and self.version != "0.3":
            raise cur.error(f"only one {kind} is supported")
        name, dtype, shape, axes, shape_expr = _parse_port(cur, self.scope.constants, self.dimensions)
        if self.version != "0.3" and (dtype != "f32" or len(shape) != 1):
            raise cur.error("only 1-D f32 ports in this version; this interface requires 'wgt 0.3'")
        if any(p.name == name for ports in self.named_ports.values() for p in ports):
            raise cur.error(f"duplicate port name {name!r}; input and output must have different names")
        if kind == "output" and cur.peek() == ("ident", "from"):
            if self.version != "0.3":
                raise cur.error("output references require 'wgt 0.3'")
            cur.next("from")
            self.returns[name] = cur.dotted("output tensor reference")
        cur.done()
        port = Port(name, dtype, shape, line, axes, shape_expr)
        self.named_ports[kind].append(port)
        self.ports.setdefault(kind, port)

    def _stmt_layer(self, cur: _Cursor, line: int) -> None:
        name = cur.ident("layer name")
        cur.expect("punct", ":")
        spec = _parse_layer_spec(cur)
        cur.done()
        if any(n == name for n, _ in self.layer_specs):
            raise cur.error(f"duplicate layer name {name!r}")
        self.layer_specs.append((name, spec))

    def _stmt_use(self, cur: _Cursor, line: int) -> None:
        self.need_v02("use", line)
        tok = cur.peek()
        if tok is not None and tok[0] == "string":
            ref = cur.next("path")[1][1:-1]
            cur.expect("ident", "as", what="'as <alias>' after a file path")
            alias = cur.ident("alias")
        else:
            root = cur.ident("library path")
            if root != "std":
                raise cur.error("use expects std/<name> or a quoted .wgt path with 'as <alias>'")
            cur.expect("punct", "/")
            name = cur.ident("library name")
            ref = f"std/{name}"
            alias = name
            if cur.peek() == ("ident", "as"):
                cur.next("as")
                alias = cur.ident("alias")
        cur.done()
        if alias in self.scope.uses:
            raise cur.error(f"duplicate use alias {alias!r}")
        lib = self.loader.load_library(ref, self.path, line)
        self.scope.uses[alias] = lib
        self.uses[alias] = lib.name

    def _stmt_block(self, cur: _Cursor, line: int) -> Block:
        self.need_v02("block", line)
        name = cur.ident("block name")
        params: list[str] = []
        defaults: dict[str, Value] = {}
        cur.expect("punct", "(")
        while cur.peek() != ("punct", ")"):
            p = cur.ident("parameter name")
            if p in params:
                raise cur.error(f"duplicate parameter {p!r}")
            params.append(p)
            if cur.peek() == ("punct", "="):
                if self.version != "0.3":
                    raise cur.error("parameter defaults require 'wgt 0.3'")
                cur.next("=")
                defaults[p] = parse_expression(cur)
            elif defaults:
                raise cur.error("required parameters must precede default parameters")
            if cur.peek() == ("punct", ","):
                cur.next(",")
        cur.expect("punct", ")")
        cur.expect("punct", ":")
        cur.done()
        if name in BUILTIN_OPS | GRAPH_OPS | LOOP_OPS | STRUCTURAL_OPS:
            raise cur.error(f"block name {name!r} shadows a built-in op")
        if name in self.scope.blocks:
            raise cur.error(f"duplicate block {name!r}")
        block = Block(name, params, [], line, self.path, self.scope, defaults)
        self.scope.blocks[name] = block
        return block

    def _stmt_data(self, cur: _Cursor, line: int) -> Dataset:
        name = cur.ident("dataset name")
        cur.expect("punct", ":")
        kv = _parse_kv(cur)
        if name in self.datasets:
            raise cur.error(f"duplicate dataset {name!r}")
        unknown = set(kv) - {"file", "sha256", "skip", "license", "source", "format", "dataset", "python", "entrypoint"}
        if unknown:
            raise cur.error(f"unknown data option(s): {', '.join(sorted(unknown))}")
        if kv:
            self.need_v02("data options", line)
        ds = Dataset(name, [], [], line, license=kv.get("license"), source=kv.get("source"))
        new_options = {"format", "dataset", "python", "entrypoint"} & set(kv)
        if new_options and self.version != "0.3":
            raise cur.error("external data options require 'wgt 0.3'")
        if sum(key in kv for key in ("file", "python", "entrypoint")) > 1:
            raise cur.error("choose only one of file, python or entrypoint")
        if ("format" in kv or "dataset" in kv) and "file" not in kv:
            raise cur.error("format and dataset require file")
        ds.format, ds.dataset = kv.get("format"), kv.get("dataset")
        ds.python, ds.entrypoint = kv.get("python"), kv.get("entrypoint")
        if getattr(self, "data_only", False) and (ds.python or ds.entrypoint):
            raise cur.error("declare Python loaders in the model file")
        if any(key in kv and not kv[key] for key in ("file", "python", "entrypoint", "format", "dataset")):
            raise cur.error("data source options must not be empty")
        if self.version != "0.3" and "file" in kv:
            ds.format = "csv"
        if ds.format and ds.format not in {"csv", "tsv", "json", "jsonl", "npz", "wgt"}:
            raise cur.error("unsupported data format")
        if "file" in kv:
            if "sha256" not in kv:
                raise cur.error("data files need sha256=<hex> so the training data is pinned")
            if not re.fullmatch(r"[0-9a-f]{64}", kv["sha256"]):
                raise cur.error("sha256 must be 64 lowercase hex characters")
            if "skip" in kv and not kv["skip"].isdigit():
                raise cur.error("skip must be a non-negative integer")
            ds.file, ds.sha256, ds.skip = kv["file"], kv["sha256"], int(kv.get("skip", 0))
        elif "sha256" in kv or "skip" in kv:
            raise cur.error("sha256= and skip= only apply to file= data")
        self.datasets[name] = ds
        return ds

    def _stmt_init(self, cur: _Cursor, line: int) -> None:
        if self.seed is not None:
            raise cur.error("duplicate init statement")
        kv = _parse_kv(cur)
        if set(kv) - {"seed", "file", "python"} or not kv:
            raise cur.error("init accepts seed=, file= or python=")
        sources = set(kv) & {"file", "python"}
        if len(sources) > 1 or any(not kv[key] for key in sources):
            raise cur.error("init requires one non-empty file= or python= source")
        if sources and self.version != "0.3":
            raise cur.error("custom initialization requires 'wgt 0.3'")
        seed = kv.get("seed", "0")
        if not seed.isdigit() or int(seed) >= 2**64:
            raise cur.error("init seed must be an integer in [0, 2**64)")
        self.seed = int(seed)
        self.initializer = {key: kv[key] for key in sources}
        if "file" in sources:
            from .initialization import read_rules
            read_rules(self.initializer["file"], self.path)

    def _stmt_import(self, cur: _Cursor, line: int) -> None:
        if cur.tokens[cur.i:cur.i + 2] == [("ident", "std"), ("punct", "/")]:
            if self.version != "0.3":
                raise cur.error("standard-library import requires 'wgt 0.3'; use 'use std/<name>' in 0.2")
            self._stmt_use(cur, line)
            return
        target = cur.dotted("layer name")
        cur.expect("ident", "from", what="'from'")
        src = cur.expect("string", what="quoted artifact path")[1:-1]
        source_layer = target
        sha = None
        freeze = False
        while not cur.at_end():
            key = cur.ident("import option")
            if key == "freeze":
                freeze = True
            elif key == "layer":
                cur.expect("punct", "=")
                source_layer = cur.dotted("source layer")
            elif key == "sha256":
                cur.expect("punct", "=")
                sha = cur.next("sha256 value")[1]
                if not re.fullmatch(r"[0-9a-f]{64}", sha):
                    raise cur.error("sha256 must be 64 lowercase hex characters")
            else:
                raise cur.error(f"unknown import option(s): {key}")
        if "." in target or "." in source_layer:
            self.need_v02("dotted layer paths in import", line)
        if any(i.target == target for i in self.imports):
            raise cur.error(f"layer {target!r} is imported twice")
        self.imports.append(ImportStmt(target, src, source_layer, sha, freeze, line))

    def _stmt_inherit(self, cur: _Cursor, line: int) -> None:
        if self.version != "0.3":
            raise cur.error("inherit requires 'wgt 0.3'")
        if self.inherit is not None:
            raise cur.error("only one inherit statement is supported")
        cur.expect("ident", "from")
        src = cur.expect("string", what="quoted artifact path")[1:-1]
        kv = _parse_kv(cur)
        if set(kv) != {"sha256"} or not re.fullmatch(r"[0-9a-f]{64}", kv["sha256"]):
            raise cur.error("inherit requires sha256=<64 lowercase hex characters>")
        self.inherit = ImportStmt("", src, "", kv["sha256"], False, line)

    def _freeze_rule(self, cur: _Cursor, line: int, frozen: bool) -> None:
        if self.version != "0.3":
            raise cur.error("standalone freeze/unfreeze requires 'wgt 0.3'")
        target = cur.dotted("layer, parameter or all")
        cur.done()
        self.freezing.append(FreezeStmt(target, frozen, line))

    def _stmt_freeze(self, cur: _Cursor, line: int) -> None:
        self._freeze_rule(cur, line, True)

    def _stmt_unfreeze(self, cur: _Cursor, line: int) -> None:
        self._freeze_rule(cur, line, False)

    def _stmt_train(self, cur: _Cursor, line: int) -> None:
        if self.train is not None:
            raise cur.error("only one train statement is supported")
        kv = _parse_kv(cur)
        if "python" in kv and "entrypoint" in kv:
            raise cur.error("choose python or entrypoint")
        if ("python" in kv or "entrypoint" in kv) and "trainer" in kv:
            raise cur.error("python/entrypoint cannot be combined with trainer")
        if ("python" in kv or "entrypoint" in kv) and self.version != "0.3":
            raise cur.error("Python training requires 'wgt 0.3'")
        if any(key in kv and not kv[key] for key in ("python", "entrypoint")):
            raise cur.error("Python training source must not be empty")
        trainer = kv.get("trainer", "python" if "python" in kv or "entrypoint" in kv else "backprop")
        if "trainer" in kv and self.version != "0.3":
            raise cur.error("trainer selection requires 'wgt 0.3'")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", trainer):
            raise cur.error("trainer must be an identifier")
        custom = trainer != "backprop"
        required = {"data", "loss", "epochs"} if custom else {"data", "loss", "optimizer", "lr", "epochs"}
        optional = {"batch", "validate", "trainer", "lr"} if custom else {"batch", "validate", "weight_decay", "trainer"}
        optional |= {"python", "entrypoint"}
        missing = required - set(kv)
        if missing:
            raise cur.error(f"train is missing: {', '.join(sorted(missing))}")
        unknown = set(kv) - required - optional
        if unknown:
            raise cur.error(f"unknown train option(s): {', '.join(sorted(unknown))}")
        if kv["loss"] not in LOSSES:
            raise cur.error(f"unknown loss {kv['loss']!r}; supported: {', '.join(sorted(LOSSES))}")
        if not custom and kv["optimizer"] not in OPTIMIZERS:
            raise cur.error(f"unknown optimizer {kv['optimizer']!r}; supported: {', '.join(sorted(OPTIMIZERS))}")
        if kv["loss"] == "cross_entropy":
            self.need_v02("loss=cross_entropy", line)
        if kv.get("optimizer") == "adamw":
            self.need_v02("optimizer=adamw", line)
        for opt in sorted(optional & set(kv)):
            self.need_v02(f"train option {opt}=", line)
        try:
            lr = float(kv.get("lr", "0.01"))
            wd = float(kv.get("weight_decay", 0))
        except ValueError:
            raise cur.error("lr and weight_decay must be numbers") from None
        if not kv["epochs"].isdigit() or int(kv["epochs"]) <= 0:
            raise cur.error("epochs must be a positive integer")
        if not math.isfinite(lr) or lr <= 0:
            raise cur.error("lr must be positive")
        if not math.isfinite(wd) or wd < 0:
            raise cur.error("weight_decay must not be negative")
        batch = None
        if "batch" in kv:
            if not kv["batch"].isdigit() or int(kv["batch"]) <= 0:
                raise cur.error("batch must be a positive integer")
            batch = int(kv["batch"])
        self.train = TrainStmt(
            kv["data"], kv["loss"], kv.get("optimizer"), lr, int(kv["epochs"]), line, batch, kv.get("validate"), wd, trainer, kv.get("python"), kv.get("entrypoint")
        )


class _Loader:
    """Loads library files once per parse and detects use cycles."""

    def __init__(self) -> None:
        self.cache: dict[str, Scope] = {}
        self.loading: list[str] = []

    def load_library(self, ref: str, from_path: str | None, line: int) -> Scope:
        if ref.startswith("std/"):
            file = STD_DIR / f"{ref[4:]}.wgt"
            if not file.is_file():
                available = ", ".join(sorted(f"std/{p.stem}" for p in STD_DIR.glob("*.wgt")))
                raise WgtError(f"unknown standard library {ref!r}; available: {available}", line, from_path)
        else:
            if not ref.endswith(".wgt"):
                raise WgtError("library paths must end in .wgt", line, from_path)
            base = Path(from_path).parent if from_path else Path(".")
            file = (base / ref).resolve()
            if not file.is_file():
                raise WgtError(f"library file not found: {ref}", line, from_path)
        key = str(file)
        if key in self.cache:
            return self.cache[key]
        if key in self.loading:
            raise WgtError(f"use cycle: {' -> '.join(self.loading + [key])}", line, from_path)
        self.loading.append(key)
        try:
            fp = _FileParser(file.read_text(encoding="utf-8"), str(file), self, library=True)
            fp.run()
        finally:
            self.loading.pop()
        fp.scope.name = ref
        self.cache[key] = fp.scope
        return fp.scope


# ---------------------------------------------------------------- resolution


def _value(v: Value, env: dict[str, int | float], spec: Spec) -> int | float:
    return evaluate(v, env, lambda message: WgtError(message, spec.line, spec.path))


def _resolve(name: str, spec: Spec, scope: Scope, env: dict[str, int | float], stack: tuple[str, ...], budget: list[int] | None = None) -> Layer:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > 10000 or len(stack) > 64:
        raise WgtError("architecture expansion exceeds 10000 nodes or 64 nested blocks", spec.line, spec.path)
    try:
        layer = _resolve_body(name, spec, scope, env, stack, budget)
    except WgtError as exc:
        if (spec.ns is not None or spec.op in scope.blocks) and (exc.path, exc.line) != (spec.path, spec.line):
            if not exc.labels and exc.line:
                exc.labels.append(diagnostic_label(exc.path, exc.line, exc.message))
            exc.labels.append(diagnostic_label(spec.path, spec.line, f"while expanding `{name}: {spec.op}`"))
        raise
    layer.inputs = list(spec.inputs)
    layer.input_spans = list(spec.input_spans)
    if layer.block:
        frame = diagnostic_label(spec.path, spec.line, f"while expanding `{name}: {layer.block}`")
        def annotate(children):
            for child in children:
                child.frames.append(frame)
                annotate(child.children)
        annotate(layer.children)
    return layer


def _resolve_body(name: str, spec: Spec, scope: Scope, env: dict[str, int | float], stack: tuple[str, ...], budget: list[int]) -> Layer:
    env = {**scope.constants, **env}
    if spec.op == "route" and spec.ns is None:
        children = [_resolve(f"branch{i}", child, scope, env, stack, budget) for i, child in enumerate(spec.chain)]
        return Layer(name, "route", {}, spec.line, spec.path, children)
    if spec.chain:
        children = [_resolve(f"step{i}", child, scope, env, stack, budget) for i, child in enumerate(spec.chain)]
        return Layer(name, "seq", {}, spec.line, spec.path, children)
    if spec.ns is None and spec.op in GRAPH_OPS:
        expected = 3 if spec.op in ("mix", "where") else 2
        if len(spec.inputs) < expected or (spec.op in ("mix", "where") and len(spec.inputs) != 3):
            raise WgtError(f"{spec.op} needs {'exactly' if expected == 3 else 'at least'} {expected} tensor names", spec.line, spec.path)
        return Layer(name, spec.op, {}, spec.line, spec.path)
    if spec.ns is None and spec.op == 'scan_state':
        if spec.inner is None:
            raise WgtError('scan_state needs a block call',spec.line,spec.path)
        child = _resolve('cell',spec.inner,scope,env,stack,budget)
        if child.op != 'seq' or not child.returns:
            raise WgtError('scan_state needs a block with an explicit return',spec.line,spec.path)
        return Layer(name,'scan_state',{},spec.line,spec.path,child.children,returns=child.returns)
    if spec.ns is None and spec.op in LOOP_OPS:
        key = "hidden" if spec.op == "scan" else "steps"
        value = _value(spec.args.get(key, 0), env, spec)
        if type(value) is not int or (value < 1 or (key == "steps" and value > 1024)) or spec.inner is None:
            raise WgtError(f"{spec.op} {key} must be a positive integer (steps <= 1024) and needs a body", spec.line, spec.path)
        if spec.op == "repeat":
            children = [_resolve(f"step{i}", spec.inner, scope, env, stack, budget) for i in range(value)]
            for index, child in enumerate(children):
                if child.inputs:
                    # Each repetition has its own input scope, including a bare merge body.
                    original_name = child.name
                    child.name = "value"
                    children[index] = Layer(original_name, "seq", {}, spec.line, spec.path, [child])
            return Layer(name, "seq", {}, spec.line, spec.path, children)
        child = _resolve("cell", spec.inner, scope, env, stack, budget)
        if spec.op == "scan" and child.op != "seq":
            raise WgtError("scan needs a block call", spec.line, spec.path)
        # The cell's children live directly under the loop, keeping weight paths short.
        children = child.children if child.op == "seq" else [child]
        options = {key: value, **({"until": spec.args["until"]} if "until" in spec.args else {})}
        if spec.op == "ponder" and "until" not in options:
            raise WgtError("ponder requires until=probability_layer", spec.line, spec.path)
        if "compact" in spec.args:
            compact = _value(spec.args["compact"], env, spec)
            if type(compact) is not int or compact not in (0, 1) or "until" not in options:
                raise WgtError("compact must be 0 or 1 and requires until", spec.line, spec.path)
            options["compact"] = compact
        if child.returns:
            raise WgtError("recur/ponder/scan body must return a single tensor; use scan_state for records", spec.line, spec.path)
        return Layer(name, spec.op, options, spec.line, spec.path, children)
    if spec.op == "residual" and spec.ns is None:
        assert spec.inner is not None
        inner = _resolve(name, spec.inner, scope, env, stack, budget)
        if inner.op != "seq":
            raise WgtError("residual(...) takes a block call or pipeline, e.g. residual(linear(out=8) -> relu)", spec.line, spec.path)
        return Layer(name, "residual", {}, spec.line, spec.path, inner.children, inner.block)

    if spec.ns is None and spec.op in {"mean", "sum", "transpose"}:
        if not spec.args:
            raise WgtError(f"{spec.op} requires named axes", spec.line, spec.path)
        return Layer(name, spec.op, dict(spec.args), spec.line, spec.path)

    if spec.ns is None and spec.op in {"rearrange", "reduce"}:
        if "pattern" not in spec.args:
            raise WgtError(f"{spec.op} requires a quoted axis pattern", spec.line, spec.path)
        sizes = {k: _value(v, env, spec) for k, v in spec.args.items() if k not in {"pattern", "op"}}
        if any(type(v) is not int or v <= 0 for v in sizes.values()):
            raise WgtError("axis sizes must be positive integers", spec.line, spec.path)
        args = {"pattern": spec.args["pattern"], "sizes": sizes}
        if spec.op == "reduce":
            args["reduction"] = spec.args.get("op")
        return Layer(name, spec.op, args, spec.line, spec.path)
    if spec.ns is None and spec.op in BUILTIN_OPS:
        args = {**OP_DEFAULTS.get(spec.op, {}), **{k: _value(v, env, spec) for k, v in spec.args.items()}}
        args = _checked_operator_args(spec.op, args, spec.line, spec.path)
        return Layer(name, spec.op, args, spec.line, spec.path)

    # Block call.
    if spec.ns is None:
        target = scope
        if spec.op not in target.blocks:
            raise WgtError(
                f"unknown layer operation or block {spec.op!r}; built-in ops: {', '.join(sorted(BUILTIN_OPS))}",
                spec.line,
                spec.path,
                code="E0201", labels=[diagnostic_label(spec.path, spec.line, "unknown operation", needle=spec.op)],
                help=suggestion(spec.op, BUILTIN_OPS | GRAPH_OPS | LOOP_OPS | set(target.blocks)),
            )
    else:
        if spec.ns not in scope.uses:
            raise WgtError(
                f"unknown library {spec.ns!r}; add 'use std/{spec.ns}' or check the alias", spec.line, spec.path
            )
        target = scope.uses[spec.ns]
        if spec.op not in target.blocks:
            raise WgtError(
                f"library {target.name!r} has no block {spec.op!r}; it has: {', '.join(sorted(target.blocks))}",
                spec.line,
                spec.path,
            )
    block = target.blocks[spec.op]
    label = block.name if target.name == "<model>" else f"{target.name}.{block.name}"
    if label in stack:
        raise WgtError(f"block recursion: {' -> '.join(stack + (label,))}", spec.line, spec.path)
    args = {k: _value(v, env, spec) for k, v in spec.args.items()}
    if set(args) - set(block.params) or set(block.params) - set(args) - set(block.defaults):
        raise WgtError(
            f"block {block.name} takes ({', '.join(block.params)}), got ({', '.join(args)})", spec.line, spec.path
        )
    block_args: dict[str, int | float] = {}
    for k in block.params:
        v = args[k] if k in args else evaluate(
            block.defaults[k], {**target.constants, **block_args},
            lambda message: WgtError(message, block.line, block.path),
        )
        if type(v) not in (int, float) or not math.isfinite(v):
            raise WgtError(f"block parameter {k} must be a finite number", spec.line, spec.path)
        block_args[k] = v
    assert block.scope is not None
    children = [_resolve(lname, lspec, block.scope, block_args, stack + (label,), budget) for lname, lspec in block.body]
    return Layer(name, "seq", {}, spec.line, spec.path, children, label, returns=dict(block.returns))


def _checked_operator_args(op, args, line, path):
    """One validation path for parsed sources and serialized architecture records."""
    if op in PRIMITIVES:
        from .primitives import arguments
        try:
            return arguments(op, args)
        except ValueError as exc:
            raise WgtError(str(exc), line, path) from None
    args = dict(args)
    signature = dict(ARG_OPS.get(op, {}))
    if op == "linear" and "in" not in args:
        signature = {"out": int}
    signature.update({k: t for k, t in OP_OPTIONALS.get(op, {}).items() if k in args})
    if set(args) != set(signature):
        if not signature:
            raise WgtError(f"{op} takes no arguments", line, path)
        want = ", ".join(f"{k}=<{t.__name__}>" for k, t in signature.items())
        raise WgtError(f"{op} requires exactly ({want})", line, path)
    for k, t in signature.items():
        if t is int:
            minimum = 0 if k in ("ports", "padding", "causal", "bias", "approximate", "qk_norm", "qkv_bias", "out_bias", "window", "affine", "padding_idx", "tie_embedding", "normalize", "activation", "flash", "cast_after", "norm_cast_after", "rope_fp32") else 1
            if type(args[k]) is not int or args[k] < minimum:
                raise WgtError(f"{op} {k} must be a positive integer", line, path)
        else:
            if type(args[k]) not in (int, float) or not math.isfinite(args[k]):
                raise WgtError(f"{op} {k} must be a finite number", line, path)
            args[k] = float(args[k])
    if "qk_norm" in args and args["qk_norm"] not in (0, 1, 2):
        raise WgtError("rope_attention qk_norm must be 0, 1 or 2", line, path)
    for key in ("ports", "bias", "approximate", "qkv_bias", "out_bias", "affine", "normalize", "flash", "cast_after", "norm_cast_after", "rope_fp32"):
        if key in args and args[key] not in (0, 1):
            raise WgtError(f"{op} {key} must be 0 or 1", line, path)
    for key in ("eps", "theta"):
        if key in args and (not math.isfinite(args[key]) or args[key] <= 0):
            raise WgtError(f"{op} {key} must be finite and positive", line, path)
    for key in ("offset", "factor", "clip"):
        if key in args and not math.isfinite(args[key]):
            raise WgtError(f"{op} {key} must be finite", line, path)
    if "clip" in args and args["clip"] < 0:
        raise WgtError("attention clip must be nonnegative", line, path)
    if op == "conv1d":
        if args['in'] % args['groups'] or args['out'] % args['groups']:
            raise WgtError("conv1d channels must be divisible by groups", line, path)
        if args['causal'] not in (0,1) or (args['causal'] and args['padding']):
            raise WgtError("conv1d causal must be 0/1 and requires padding=0", line, path)
    if op == "sparse_moe":
        if args["top_k"] > args["experts"] or args.get("aux_coef", 0) < 0 or args.get("activation", 0) not in (0, 1, 2):
            raise WgtError("invalid sparse_moe top_k, aux_coef or activation", line, path)
    for key in ("yarn_factor", "yarn_fast", "yarn_slow", "attention_factor"):
        if key in args and args[key] <= 0:
            raise WgtError(f"{key} must be positive", line, path)
    if op == "embedding" and args.get("padding_idx", 0) >= args["vocab"]:
        raise WgtError("embedding padding_idx outside vocabulary", line, path)
    if "dropout" in args and not 0 <= args["dropout"] < 1:
        raise WgtError("attention dropout must be in [0,1)", line, path)
    if op == "rope_attention" and args.get("yarn_factor", 1) != 1 and args.get("theta", 10000) <= 1:
        raise WgtError("YaRN requires theta > 1", line, path)
    if op == "rope_attention" and (args["heads"] % args["kv_heads"] or args["head_dim"] % 2):
        raise WgtError("rope_attention requires heads divisible by kv_heads and even head_dim", line, path)
    if op == "self_attention":
        if args["dim"] % args["heads"] or args["causal"] not in (0, 1):
            raise WgtError("self_attention needs dim divisible by heads and causal=0 or 1", line, path)
    if op == "dropout" and not 0 <= args["p"] < 1:
        raise WgtError("dropout p must be in [0, 1)", line, path)
    return args


def output_width(layers: list[Layer], width: int) -> int:
    """Propagate the feature width through layers, raising on mismatches."""

    for layer in layers:
        if layer.op == "linear":
            if layer.args["in"] != width:
                raise WgtError(
                    f"shape mismatch: layer {layer.name!r} expects in={layer.args['in']} but receives {width}",
                    layer.line,
                    layer.path,
                )
            width = int(layer.args["out"])
        elif layer.op in ("layernorm", "rmsnorm"):
            if layer.args["dim"] != width:
                raise WgtError(
                    f"shape mismatch: layer {layer.name!r} normalises dim={layer.args['dim']} but receives {width}",
                    layer.line,
                    layer.path,
                )
        elif layer.op == "seq":
            width = output_width(layer.children, width)
        elif layer.op == "residual":
            out = output_width(layer.children, width)
            if out != width:
                raise WgtError(
                    f"residual {layer.name!r} must keep the width ({width}) but its block produces {out}",
                    layer.line,
                    layer.path,
                )
    return width


def _last_leaf(layers: list[Layer]) -> Layer:
    last = layers[-1]
    return _last_leaf(last.children) if last.op == "seq" else last


# ---------------------------------------------------------------- entry points


def parse(source: str, path: str | None = None, *, check_training: bool = True, config=None) -> Program:
    try:
        return _parse(source, path, check_training=check_training, config=config)
    except WgtError as exc:
        if exc.path == path:
            exc.source = source
        raise


def _parse(source: str, path: str | None = None, *, check_training: bool = True, config=None) -> Program:
    """Parse and statically check a model file."""

    from .derivation import is_derived, parse as parse_derived

    if is_derived(source):
        if config:
            raise WgtError("derived sources do not declare configuration")
        return parse_derived(source, path, check_training=check_training)
    if source.startswith("wgt tensors "):
        raise WgtError("tensor inventory is not executable: a forward graph and supported operators are required", path=path)
    fp = _FileParser(source, path, _Loader(), library=False, config=config)
    fp.run()
    if set(fp.config_overrides)-set(fp.config_schema):
        raise WgtError("unknown configuration keys: " + ", ".join(sorted(set(fp.config_overrides)-set(fp.config_schema))),path=path)
    if fp.model is None:
        raise WgtError("missing 'model <name>' statement", None, path)
    for kw in ("input", "output"):
        if kw not in fp.ports:
            raise WgtError(f"missing '{kw}' statement", None, path)
    if not fp.layer_specs:
        raise WgtError("model has no layers", None, path)
    if fp.seed is None:
        raise WgtError("missing 'init seed=<int>'; every parameter must have an explicit initialization", None, path)

    layers = []
    budget = [0]
    for name, spec in fp.layer_specs:
        if fp.version == "0.1" and (spec.op in _V02_OPS or spec.op == "residual" or spec.ns is not None):
            raise WgtError(f"{spec.op!r} requires 'wgt 0.2'", spec.line, path)
        resolved = _resolve(name, spec, fp.scope, {}, (), budget)

        def needs_v03(layer):
            return bool(layer.inputs) or layer.op in _V03_OPS | GRAPH_OPS | LOOP_OPS or any(needs_v03(c) for c in layer.children)

        if fp.version != "0.3" and needs_v03(resolved):
            raise WgtError(f"{spec.op!r} requires 'wgt 0.3'", spec.line, path)
        if fp.version == "0.3":
            resolved.inputs = ["input" if ref == fp.ports["input"].name and ref not in {layer.name for layer in layers} else ref for ref in resolved.inputs]
        layers.append(resolved)

    for ds in fp.datasets.values():
        if ds.file and check_training:
            from .datasets import load_file
            load_file(ds, math.prod(fp.ports["input"].shape), path)

    if fp.inherit:
        if fp.imports:
            raise WgtError(
                "inherit cannot be combined with import; use imports for partial reuse", fp.inherit.line, path
            )
        fp.imports = [
            ImportStmt(layer.name, fp.inherit.source, layer.name, fp.inherit.sha256, False, fp.inherit.line)
            for layer in layers
            if layer.has_params
        ]

    program = Program(
        version=fp.version or LANGUAGE_VERSION,
        model=fp.model,
        input=fp.ports["input"],
        output=fp.ports["output"],
        layers=layers,
        datasets=fp.datasets,
        seed=fp.seed,
        initializer=fp.initializer,
        imports=fp.imports,
        train=fp.train,
        path=path,
        uses=fp.uses,
        freezing=fp.freezing,
        inherit=fp.inherit,
        inputs=fp.named_ports["input"],
        outputs=fp.named_ports["output"],
        returns=fp.returns,
        dimensions=fp.dimensions,
        constraints=fp.constraints,
        weight_types=fp.weight_types,
        weight_assignments=fp.weight_assignments,
        adapters=fp.adapters,
        config_schema=fp.config_schema,
        config_values=fp.config_values,
        parameter_ids=fp.parameter_ids,
        parameter_shares=fp.parameter_shares,
    )
    check(program, check_training=check_training)
    return program


def parse_library(source: str, path: str | None = None) -> Scope:
    """Parse a library file (``use`` and ``block`` only)."""

    fp = _FileParser(source, path, _Loader(), library=True)
    fp.run()
    return fp.scope


def _load_csv(ds: Dataset, input_width: int, path: str | None) -> None:
    from .datasets import load_file
    load_file(ds, input_width, path)


def check(program: Program, *, check_training: bool = True) -> None:
    """Shape and reference checks that do not need weights."""

    path = program.path
    if program.input.name == program.output.name:
        raise WgtError("input and output must have different names", program.output.line, path)
    if program.version != "0.3" and not any(layer.has_params for layer in program.layers):
        raise WgtError("model has no parameters (add at least one linear layer)", program.layers[0].line, path)
    if program.dimensions:
        from .symbolic import check_symbolic
        check_symbolic(program)
    elif program.version == "0.3" and (program.weight_types or program.weight_assignments or program.adapters or program.parameter_ids or program.parameter_shares or any(not p.shape or 0 in p.shape for p in program.inputs + program.outputs) or len(program.inputs) > 1 or len(program.outputs) > 1 or program.returns):
        from .ports import check_ports
        check_ports(program)
    elif program.version == "0.3":
        dtype, shape = output_contract(program.layers, program.input.dtype, program.input.shape, axes=program.input.axes)
        if dtype != program.output.dtype or shape != program.output.shape:
            raise WgtError(
                f"shape/dtype mismatch: network produces {dtype}{list(shape)}, output declares "
                f"{program.output.dtype}{list(program.output.shape)}",
                program.output.line,
                path, code="E0301",
                labels=[diagnostic_label(path, program.output.line, "declared output"),
                        diagnostic_label(program.layers[-1].path, program.layers[-1].line,
                                         f"produces {dtype}{list(shape)}")],
                help="match the output declaration to the final layer, or adjust the final layer",
            )
    else:
        width = output_width(program.layers, program.input.shape[0])
        if width != program.output.shape[0]:
            raise WgtError(
                f"shape mismatch: network produces {width} features but output declares {program.output.shape[0]}",
                program.output.line,
                path,
            )
    if program.output.axes and not (program.dimensions or len(program.inputs) > 1 or len(program.outputs) > 1 or program.returns):
        inferred = program.layers[-1].output_spec.get("axes", [None] * len(program.output.shape))
        if any(a and b and a != b for a, b in zip(inferred, program.output.axes)):
            raise WgtError(f"output axis mismatch: inferred {inferred}, declared {list(program.output.axes)}", program.output.line, path)
    input_size = math.prod(program.input.shape)
    output_size = math.prod(program.output.shape)

    for ds in program.datasets.values() if check_training else ():
        if (ds.python or ds.entrypoint) and not ds.loaded:
            continue
        if len(ds.inputs) != len(ds.targets):
            raise WgtError("inputs and targets must have equal row counts", ds.line, path)
        if not ds.inputs:
            raise WgtError(f"dataset {ds.name!r} has no rows", ds.line, path)
        twidth = len(ds.targets[0])
        for x, y in zip(ds.inputs, ds.targets):
            if len(x) != input_size or len(y) != twidth:
                raise WgtError(
                    f"dataset {ds.name!r} rows must be [{input_size} values] -> [{twidth} values]",
                    ds.line,
                    path,
                )
        for row in ds.inputs + ds.targets:
            if any(not math.isfinite(v) for v in row):
                raise WgtError("dataset values must be finite", ds.line, path)
        if program.input.dtype == "i64":
            for row in ds.inputs:
                if any(v != int(v) or not 0 <= v < 2**63 for v in row):
                    raise WgtError("i64 inputs must be non-negative integers below 2**63", ds.line, path)
            first = program.layers[0]
            while first.op == "seq":
                first = first.children[0]
            if first.op == "embedding" and any(v >= first.args["vocab"] for row in ds.inputs for v in row):
                raise WgtError("token index outside embedding vocabulary", ds.line, path)
        if program.train is None or ds.name not in (program.train.data, program.train.validate):
            if twidth != output_size and twidth != math.prod(program.output.shape[:-1]):
                raise WgtError(
                    f"dataset {ds.name!r} has {twidth} target values per row but output declares {output_size}",
                    ds.line,
                    path,
                )

    for imp in program.imports:
        layer = program.find(imp.target)
        if layer is None:
            raise WgtError(f"import targets unknown layer {imp.target!r}", imp.line, path)
        if not layer.has_params:
            raise WgtError(f"import target {imp.target!r} has no parameters", imp.line, path)

    if program.weight_types or program.weight_assignments or program.freezing or program.adapters or program.parameter_ids or program.parameter_shares:
        # Resolve against the same parameter names used by the runtime, including
        # packed attention projections. Do not silently accept an empty selector.
        from .artifact import build_module
        import torch

        with torch.device("meta"):
            module = build_module([layer.to_dict() for layer in program.layers])
            from .parameter_identity import bind_parameters
            try:
                bind_parameters(module,program.parameter_ids,program.parameter_shares)
                from .adapters import attach_adapters
                attach_adapters(module,program.adapters)
                from .weight_types import attach_weight_types
                attach_weight_types(module, program.weight_types, program.weight_assignments)
            except ValueError as exc:
                raise WgtError(str(exc),path=program.path) from None
        apply_freezing(program, module, require_trainable=check_training)

    if not check_training:
        return

    train = program.train
    if train is None:
        frozen = [i for i in program.imports if i.freeze] + program.freezing
        if frozen:
            raise WgtError("freeze only makes sense together with a train statement", frozen[0].line, path)
        return

    for role, name in (("train", train.data), ("validate", train.validate)):
        if name is None:
            continue
        if name not in program.datasets:
            raise WgtError(f"{role} refers to unknown dataset {name!r}", train.line, path)
        _check_targets(program, program.datasets[name], train)
    if train.validate is not None and train.validate == train.data:
        raise WgtError("validate must be a different dataset from data", train.line, path)

    last = _last_leaf(program.layers)
    if train.loss == "bce" and last.op != "sigmoid":
        raise WgtError("loss=bce requires the last layer to be sigmoid", train.line, path)
    if train.loss == "cross_entropy" and last.op in ("softmax", "sigmoid"):
        raise WgtError(f"loss=cross_entropy expects raw logits; remove the final {last.op} layer", train.line, path)

    frozen_prefixes = [i.target for i in program.imports if i.freeze]

    def has_trainable(layers: list[Layer], prefix: str) -> bool:
        for layer in layers:
            full = f"{prefix}{layer.name}"
            if any(full == f or full.startswith(f + ".") for f in frozen_prefixes):
                continue
            if layer.op in PARAM_OPS or has_trainable(layer.children, full + "."):
                return True
        return False

    if not program.freezing and not has_trainable(program.layers, ""):
        raise WgtError("train has nothing to update: every parameter layer is frozen", train.line, path)


def _check_targets(program: Program, ds: Dataset, train: TrainStmt) -> None:
    if (ds.python or ds.entrypoint) and not ds.loaded:
        return
    out = math.prod(program.output.shape)
    width = len(ds.targets[0])
    if train.loss == "cross_entropy":
        out = program.output.shape[-1]
        if width != math.prod(program.output.shape[:-1]):
            raise WgtError(f"loss=cross_entropy needs one class index per row in {ds.name!r}", ds.line, program.path)
        for row in ds.targets:
            for y in row:
                if not math.isfinite(y) or y != int(y) or not 0 <= y < out:
                    raise WgtError(
                        f"dataset {ds.name!r}: class index {y} is not an integer in [0, {out})", ds.line, program.path
                    )
    elif width != out:
        raise WgtError(
            f"dataset {ds.name!r} has {width} target values per row but output declares {out}", ds.line, program.path
        )


def parse_file(path: str | Path, *, check_training: bool = True, config=None) -> Program:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise WgtError(f"cannot read source: {exc.strerror}", None, str(p)) from None
    return parse(text, str(p), check_training=check_training, config=config)


def is_library_source(text: str) -> bool:
    """A file without a model statement is treated as a library."""
    return not any(_strip_comment(line).split()[:1] == ["model"] for line in text.splitlines())


def output_contract(layers: list[Layer], dtype: str, shape: tuple[int, ...], bindings=None, axes=(), binding_axes=None) -> tuple[str, tuple[int, ...]]:
    """Static per-sample contracts: batch is always an optional leading runtime dimension."""
    values = {"input": (dtype, shape), **(bindings or {})}
    axes = list(axes or [None] * len(shape))
    if len(axes) != len(shape) or any(a is not None and (not isinstance(a, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", a)) for a in axes) or len([a for a in axes if a]) != len(set(a for a in axes if a)):
        raise WgtError("invalid or duplicate axis names")
    axis_values = {"input": axes, **{k: [None] * len(v[1]) for k, v in (bindings or {}).items()}}
    axis_values.update(binding_axes or {})
    origins = {}
    previous = None
    for layer in layers:
        if layer.name in values:
            raise WgtError(f"duplicate or reserved tensor name {layer.name!r}", layer.line, layer.path)
        if not isinstance(layer.inputs, list) or not all(isinstance(ref, str) for ref in layer.inputs):
            raise WgtError("tensor inputs must be a list of names", layer.line, layer.path)
        if layer.op not in GRAPH_OPS and layer.op not in PRIMITIVES and not (layer.op in {"route", "scan_state"} or (layer.op == "seq" and layer.returns)) and not (layer.op == "rope_attention" and layer.args.get("ports")) and not (layer.op == "scan" and len(layer.inputs) == 2) and len(layer.inputs) > 1:
            raise WgtError("unary layers take one tensor input", layer.line, layer.path)
        if layer.op in STRUCTURAL_OPS and not layer.children:
            raise WgtError("structural layer needs a nonempty body", layer.line, layer.path)
        def reference_label(index, ref, message):
            if index < len(layer.input_spans):
                line, column, end = layer.input_spans[index]
                return diagnostic_label(layer.path, line, message, column=column, end_column=end)
            return diagnostic_label(layer.path, layer.line, message, needle=ref)

        operands = []
        for index, ref in enumerate(layer.inputs):
            if ref not in values:
                raise WgtError(f"unknown or forward tensor reference {ref!r}", layer.line, layer.path,
                    code="E0201", labels=[reference_label(index, ref, "not defined in this scope")],
                    help=suggestion(ref, values) or "declare this tensor before using it; `input` is the block input")
            operands.append(values[ref])
        if operands:
            dtype, shape = operands[0]
            axes = list(axis_values[layer.inputs[0]])
        before_axes = list(axes)
        layer.input_spec = {"dtype": dtype, "shape": list(shape)}
        args, op = layer.args, layer.op
        if op in BUILTIN_OPS and op not in {"rearrange", "reduce"}:
            args = layer.args = _checked_operator_args(op, args, layer.line, layer.path)

        def require(ok, message):
            if not ok:
                labels = [diagnostic_label(layer.path, layer.line, message, needle=layer.op)]
                refs = layer.inputs or ([previous] if previous else ["input"])
                notes = []
                for index, ref in enumerate(refs):
                    contract = values.get(ref)
                    if contract:
                        kind, dims = contract
                        desc = f"{ref}: {kind}[{','.join(map(str, dims))}]"
                        if layer.inputs:
                            labels.append(reference_label(index, ref, desc))
                        if ref in origins:
                            origin = origins[ref]
                            labels.append(diagnostic_label(origin.path, origin.line, desc, needle=origin.name))
                        else:
                            notes.append(desc)
                labels.extend(layer.frames)
                raise WgtError(f"{layer.name}: {message}", layer.line, layer.path, code="E0301",
                               labels=labels, notes=notes,
                               help="make the producer dimensions match this operation; no implicit broadcasting is performed")

        # Empty tensors and scalar samples have defined behavior only for audited ops.
        scalar_ops = {"identity", "relu", "tanh", "sigmoid", "gelu", "silu", "scale", "dropout", "add", "mul", "mix", "where", "seq", "residual", "recur", "ponder", "route", "scan_state", "rearrange", "reduce"}
        require(bool(shape) or op in scalar_ops or op in PRIMITIVES, f"{op} requires a non-scalar sample")
        if 0 in shape:
            empty_ops = scalar_ops - {"rearrange", "reduce"} | {"linear", "embedding", "scan", "concat"}
            require(op in empty_ops or op in PRIMITIVES, f"{op} does not support empty sample dimensions")
        if op == "concat":
            require(all(bool(s) for _, s in operands), "concat requires non-scalar inputs")

        # Resolve source-only axis sugar after its actual input axes are known.
        # Artifacts contain only the existing explicit transform operators.
        if op in {"mean", "sum", "transpose"}:
            require(all(a is not None for a in axes), "named axis shorthand requires all input axes to be named")
            if op == "transpose":
                target = args["axes"].split()
                require(len(target) == len(axes) and set(target) == set(axes),
                        "transpose must list each input axis exactly once")
                resolved = {"pattern": " ".join(axes) + " -> " + " ".join(target), "sizes": {}}
                op = "rearrange"
            else:
                require(args["axis"] in axes, f"unknown reduction axis {args['axis']!r}")
                target = [a for a in axes if a != args["axis"]]
                resolved = {"pattern": " ".join(axes) + " -> " + " ".join(target), "sizes": {}, "reduction": op}
                op = "reduce"
            layer.op, layer.args = op, resolved
            args = resolved

        layer.results = {}
        if op == 'rope_attention' and args.get('ports'):
            require(len(operands) == 4, "attention ports require x,mask,old_key,old_value")
            require(dtype == 'f32' and len(shape) == 2 and shape[0] > 0 and shape[1] == args['dim'], "attention ports require nonempty f32[T,dim]")
            old_dtype,old_shape = operands[2]
            require(old_dtype == 'f32' and len(old_shape) == 3 and old_shape[1:] == (args['kv_heads'],args['head_dim']), "cache must be f32[P,kv_heads,head_dim]")
            require(operands[3] == operands[2], "key/value cache shapes and dtypes must match")
            total = old_shape[0] + shape[0]
            require(operands[1] == ('bool',(total,)), "attention mask must be bool[P+T]")
            require(total <= args.get('max_len',32768), "cache and tokens exceed max_len")
            for field in ('key','value'):
                layer.results[field] = {'dtype':'f32','shape':[total,*old_shape[1:]]}
        if op in PRIMITIVES:
            from .primitives import contract
            try:
                dtype, shape = contract(op, args, operands or [(dtype, shape)])
            except ValueError as exc:
                require(False, str(exc))
            axes = [None] * len(shape)
        elif op in {"rearrange", "reduce"}:
            from .axes import plan
            require(dtype == "f32" or op == "rearrange", "reduce expects f32 tensors")
            require(op != "reduce" or args.get("reduction") in {"mean", "sum"}, "reduce requires mean or sum")
            try:
                transform = plan(args["pattern"], shape, args.get("sizes", {}), args.get("reduction") if op == "reduce" else None)
            except (ValueError, TypeError, KeyError) as exc:
                require(False, str(exc))
            require(all(a is None or b is None or a == b for a, b in zip(axes, transform["input_axes"])),
                    f"axis pattern does not match input axes {axes}")
            shape = tuple(transform["output_shape"])
            axes = transform["output_axes"]
        elif op in GRAPH_OPS:
            for ref in layer.inputs[1:]:
                other = axis_values[ref]
                require(len(axes) == len(other) and all(a is None or b is None or a == b for a, b in zip(axes, other)),
                        f"axis mismatch between {layer.inputs[0]} {axes} and {ref} {other}")
                axes = [a or b for a, b in zip(axes, other)]
                require(len([a for a in axes if a]) == len(set(a for a in axes if a)), "duplicate inferred axis names")
            expected = 3 if op in ("mix", "where") else 2
            require(len(operands) >= expected and (expected == 2 or len(operands) == 3), "invalid merge input count")
            if op == "where":
                require(operands[0][0] in {"f32", "bool"} and all(t == "f32" for t, _ in operands[1:]), "where expects bool/f32 condition and f32 values")
                dtype = "f32"
            else:
                require(all(t == "f32" for t, _ in operands), "merge expects f32 tensors")
            if op == "concat":
                require(all(s[:-1] == shape[:-1] for _, s in operands), "concat requires matching dimensions except the last")
                shape = (*shape[:-1], sum(s[-1] for _, s in operands))
            else:
                require(all(s == shape for _, s in operands), f"{op} requires identical shapes (no implicit broadcasting)")
        elif op == 'scan_state' or (op == 'seq' and layer.returns):
            actual = operands or [(dtype,shape)]
            is_scan = op == 'scan_state'
            require(not is_scan or len(shape) >= 1, 'scan_state requires a sequence sample axis')
            item_shape = shape[1:] if is_scan else shape
            bindings = {f'state{i}' if is_scan else f'arg{i+1}':v for i,v in enumerate(actual[1:])}
            if is_scan:
                require(set(layer.returns) == {'output',*bindings}, 'scan_state return must contain output and exactly state0..stateN')
            output_contract(layer.children,dtype,item_shape,bindings)
            local = {'input':(dtype,item_shape),**bindings}
            for child in layer.children:
                local[child.name] = (child.output_spec['dtype'],tuple(child.output_spec['shape']))
                local.update({f'{child.name}.{field}':(spec['dtype'],tuple(spec['shape'])) for field,spec in child.results.items()})
            for field,ref in layer.returns.items():
                require(ref in local, f'unknown block result {ref!r}')
                result_dtype,result_shape = local[ref]
                if is_scan and field != 'output':
                    require((result_dtype,result_shape) == bindings[field], f'{field} must preserve shape and dtype')
                if is_scan and field == 'output':
                    result_shape = (shape[0],*result_shape)
                layer.results[field] = {'dtype':result_dtype,'shape':list(result_shape)}
            dtype,shape = layer.results['output']['dtype'],tuple(layer.results['output']['shape'])
            axes = [None]*len(shape)
        elif op == "scan":
            require(dtype == "f32" and len(shape) == 2 and shape[-1] > 0, "scan expects f32[sequence, features]")
            hidden = args.get("hidden")
            require(type(hidden) is int and hidden > 0, "scan hidden must be a positive integer")
            if len(operands) == 2:
                require(operands[1] == (dtype, (hidden,)), "scan initial state must be f32[hidden]")
            result = output_contract(layer.children, dtype, (shape[-1],), {"state": (dtype, (hidden,))})
            require(result == (dtype, (hidden,)), "scan cell must output f32[hidden]")
            shape = (shape[0], hidden)
        elif op == "route":
            require(len(operands) == 2 and operands[0][0] in {"i64", "bool"} and operands[0][1] == (),
                    "route requires a scalar i64/bool selector and an f32 state")
            require(len(layer.children) >= 2 and (dtype != "bool" or len(layer.children) == 2), "bool route requires exactly two branches")
            dtype, shape = operands[1]
            require(dtype == "f32", "route state must be f32")
            axes = axis_values[layer.inputs[1]]
            for branch in layer.children:
                require(not branch.inputs and not branch.returns, "route branches must be single-input single-output computations")
                result = output_contract([branch], dtype, shape, axes=axes)
                require(result == (dtype, shape) and not branch.results, "route branches must preserve state shape and dtype")
                result_axes = branch.output_spec.get("axes", [None] * len(shape))
                require(all(a is None or b is None or a == b for a,b in zip(axes,result_axes)), "route branches must preserve axes")
        elif op in {"recur", "ponder"}:
            require(type(args.get("steps")) is int and 1 <= args["steps"] <= 1024, "recur steps must be in [1, 1024]")
            require("compact" not in args or (op == "recur" and type(args["compact"]) is int and args["compact"] in (0,1) and "until" in args), "invalid compact option")
            require(op != "ponder" or "until" in args, "ponder requires until")
            result = output_contract(layer.children, dtype, shape, axes=axes)
            require(result == (dtype, shape), "recur body must preserve complete shape and dtype")
            if "until" in args:
                require(isinstance(args["until"], str), "recur until must name a direct body layer")
                gate = next((c for c in layer.children if c.name == args["until"]), None)
                require(gate is not None, "recur until must name a direct body layer")
                require(dtype == "f32" and gate.output_spec["dtype"] == "f32" and gate.output_spec["shape"] == [1],
                        "recur until probability must be f32[1] and body must be f32")
            result_axes = layer.children[-1].output_spec.get("axes", [None] * len(shape))
            require(all(a is None or b is None or a == b for a, b in zip(axes, result_axes)), "recur body must preserve axis names")
            if op == "ponder":
                layer.results = {
                    "states": {"dtype":"f32", "shape":[args["steps"], *shape]},
                    "probabilities": {"dtype":"f32", "shape":[args["steps"], 1]},
                    "halt_weights": {"dtype":"f32", "shape":[args["steps"], 1]},
                    "expected_steps": {"dtype":"f32", "shape":[]},
                }
        elif op in STRUCTURAL_OPS:
            new_dtype, new_shape = output_contract(layer.children, dtype, shape, axes=axes)
            if op == "residual":
                require((new_dtype, new_shape) == (dtype, shape), "residual must preserve complete shape and dtype")
                result_axes = layer.children[-1].output_spec.get("axes", [None] * len(shape))
                require(all(a is None or b is None or a == b for a, b in zip(axes, result_axes)), "residual must preserve axis names")
            dtype, shape = new_dtype, new_shape
        elif op == "embedding":
            require(dtype == "i64" and len(shape) == 1, "embedding expects i64[sequence]")
            dtype, shape = "f32", (*shape, int(args["dim"]))
        else:
            require(op in BUILTIN_OPS, f"unknown operation {op!r}")
            require(dtype == "f32" or op == "identity", f"{op} expects f32; convert tokens with embedding first")
            if op == "linear":
                # Resolve once during source checking; saved weights always carry an explicit width.
                require(shape[-1] > 0, "linear feature width must be positive")
                args.setdefault("in", shape[-1])
                require(shape[-1] == args["in"], f"linear expects {axes[-1] or 'last dimension'}={args['in']}, got {shape[-1]}")
                if args.get("tie_embedding", 0):
                    embeddings = [v for v in origins.values() if v.op == "embedding"]
                    index = args["tie_embedding"] - 1
                    require(index < len(embeddings), "tie_embedding must name an earlier embedding by 1-based scope index")
                    target = embeddings[index]
                    require(args["in"] == target.args["dim"] and args["out"] == target.args["vocab"] and args.get("bias", 1) == 0,
                            "tied projection must match embedding dimensions and have bias=0")
                shape = (*shape[:-1], int(args["out"]))
            elif op in ("layernorm", "rmsnorm"):
                require(shape[-1] == args["dim"], "layernorm dimension mismatch")
            elif op in ("self_attention", "rope_attention", "position_embedding", "sparse_moe"):
                require(len(shape) == 2 and shape[-1] == args["dim"], f"{op} expects f32[sequence, dim]")
                if op == "position_embedding":
                    require(shape[0] <= args["max_len"], "sequence exceeds max_len")
            elif op in ("meanpool", "last"):
                require(len(shape) == 2, f"{op} expects f32[sequence, dim]")
                shape = (shape[-1],)
            elif op == "flatten":
                shape = (math.prod(shape),)
            elif op == "conv1d":
                require(len(shape) == 2 and shape[0] == args['in'], "conv1d expects f32[channels,length] with matching in")
                effective = args['dilation'] * (args['kernel']-1)
                padding = effective if args['causal'] else 2*args['padding']
                length = (shape[1]+padding-effective-1)//args['stride']+1
                require(length > 0, "conv1d kernel produces an empty length")
                shape = (args['out'],length)
            elif op in ("conv2d", "maxpool2d", "globalavgpool2d"):
                require(len(shape) == 3, f"{op} expects f32[channels, height, width]")
                channels, height, width = shape
                if op == "globalavgpool2d":
                    shape = (channels,)
                else:
                    if op == "conv2d":
                        require(channels == args["in"], "conv2d input channels mismatch")
                    padding = int(args.get("padding", 0))
                    kernel, stride = int(args["kernel"]), int(args["stride"])
                    height, width = (
                        (height + 2 * padding - kernel) // stride + 1,
                        (width + 2 * padding - kernel) // stride + 1,
                    )
                    require(height > 0 and width > 0, "kernel produces an empty spatial dimension")
                    shape = (int(args.get("out", channels)), height, width)
        if op in {"residual", "recur", "ponder"}:
            # A body can supply names for an unnamed input. Check that boundary
            # again with those names: otherwise a square transpose can silently
            # swap meanings on residual addition or on the next recurrence.
            result_axes = layer.children[-1].output_spec.get("axes", [None] * len(shape))
            refined_axes = [a or b for a, b in zip(before_axes, result_axes)]
            if refined_axes != before_axes:
                output_contract(layer.children, dtype, shape, axes=refined_axes)

        # Preserve semantic names when an operation preserves axis identity. New
        # feature axes remain unnamed until an explicit pattern supplies a name.
        if op in STRUCTURAL_OPS and op != "route" and not layer.returns:
            axes = list(layer.children[-1].output_spec.get("axes", [None] * len(layer.children[-1].output_spec["shape"])))
            if op == "scan":
                axes = [before_axes[0], *axes]
        elif op == "embedding":
            axes = [*before_axes, None]
        elif op in {"meanpool", "last"}:
            axes = before_axes[-1:]
        elif op == "globalavgpool2d":
            axes = before_axes[:1]
        elif op == "flatten":
            axes = [None]
        if any(before_axes):
            layer.input_spec["axes"] = before_axes
        if len(axes) != len(shape):
            axes = [None] * len(shape)
        layer.output_spec = {"dtype": dtype, "shape": list(shape)}
        if any(axes):
            layer.output_spec["axes"] = list(axes)
        if op == 'sparse_moe' and args.get('ports'):
            layer.results['aux'] = {'dtype':'f32','shape':[]}
        if layer.results:
            layer.results['output'] = dict(layer.output_spec)
            for field,spec in layer.results.items():
                values[f'{layer.name}.{field}'] = (spec['dtype'],tuple(spec['shape']))
                axis_values[f'{layer.name}.{field}'] = list(spec.get('axes',[None]*len(spec['shape'])))
        axis_values[layer.name] = list(axes)
        values[layer.name] = (dtype, shape)
        origins[layer.name] = layer
        previous = layer.name
    return dtype, shape


def apply_freezing(program: Program, module, *, require_trainable: bool = True) -> None:
    """Imports establish defaults; ordered explicit rules override them."""
    parameters = dict(module.named_parameters())
    for imp in program.imports:
        if imp.freeze:
            for name, param in parameters.items():
                if name == imp.target or name.startswith(imp.target + "."):
                    param.requires_grad_(False)
    for rule in program.freezing:
        matches = [
            p
            for name, p in parameters.items()
            if rule.target == "all" or name == rule.target or name.startswith(rule.target + ".")
        ]
        if not matches:
            raise WgtError(f"freeze selector {rule.target!r} matches no parameters", rule.line, program.path)
        for param in matches:
            param.requires_grad_(not rule.frozen)
    if require_trainable and program.train and not any(p.requires_grad for p in parameters.values()):
        raise WgtError("train has nothing to update: every parameter is frozen", program.train.line, program.path)
