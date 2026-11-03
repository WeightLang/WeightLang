"""Parser and static checks for WeightLang source files (language versions 0.1 and 0.2).

The format is line oriented: one statement per line, ``#`` starts a comment.
Indented lines belong to the preceding ``data`` block (rows) or ``block``
definition (layers). See ``docs/language.md``.

A *model file* declares one model. A *library file* (e.g. the bundled
``std/nn.wgt``) contains only ``use`` and ``block`` statements.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

LANGUAGE_VERSION = "0.3"
SUPPORTED_VERSIONS = ("0.1", "0.2", "0.3")
STD_DIR = Path(__file__).resolve().parent / "std"

# Parameter-free element-wise ops.
PARAM_FREE_OPS = {"relu", "tanh", "sigmoid", "gelu", "silu", "softmax", "identity"}
# Ops with arguments: name -> {arg: type}.
ARG_OPS: dict[str, dict[str, type]] = {
    "linear": {"in": int, "out": int},
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
PARAM_FREE_OPS |= {"globalavgpool2d", "meanpool", "flatten"}
BUILTIN_OPS = PARAM_FREE_OPS | set(ARG_OPS)
PARAM_OPS = {
    "linear",
    "layernorm",
    "embedding",
    "position_embedding",
    "self_attention",
    "conv2d",
}  # ops that own trainable tensors
STRUCTURAL_OPS = {"seq", "residual"}  # produced by block calls and residual(...)
LOSSES = {"mse", "bce", "cross_entropy"}
OPTIMIZERS = {"sgd", "adam", "adamw"}
DTYPES = {"f32", "i64"}

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
  | (?P<punct>[:=\[\](),./])
    """,
    re.VERBOSE,
)


class WgtError(Exception):
    """A user-facing error, optionally tied to a source line."""

    def __init__(self, message: str, line: int | None = None, path: str | None = None):
        self.message = message
        self.line = line
        self.path = path
        super().__init__(str(self))

    def __str__(self) -> str:
        where = self.path or "<source>"
        if self.line is not None:
            return f"{where}:{self.line}: {self.message}"
        return f"{where}: {self.message}"


Value = int | float | str  # str = reference to a block parameter


@dataclass
class Port:
    name: str
    dtype: str
    shape: tuple[int, ...]
    line: int


@dataclass
class Spec:
    """An unresolved layer expression: ``op``, ``op(k=v)``, ``ns.block(k=v)``, ``residual(<spec>)``."""

    op: str
    ns: str | None
    args: dict[str, Value]
    inner: Spec | None
    line: int
    path: str | None


@dataclass
class Layer:
    """A resolved layer. ``seq`` and ``residual`` layers have children."""

    name: str
    op: str
    args: dict[str, int | float]
    line: int
    path: str | None = None
    children: list[Layer] = field(default_factory=list)
    block: str | None = None  # e.g. "std/nn.mlp" when produced by a block call
    input_spec: dict | None = None
    output_spec: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "op": self.op, "args": dict(self.args)}
        if self.op in STRUCTURAL_OPS:
            d["layers"] = [c.to_dict() for c in self.children]
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


@dataclass
class Scope:
    """A file's namespace: its blocks and the libraries it uses."""

    name: str  # "std/nn", a library path as written, or "<model>"
    path: str | None
    blocks: dict[str, Block] = field(default_factory=dict)
    uses: dict[str, Scope] = field(default_factory=dict)


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


@dataclass
class ImportStmt:
    target: str  # dotted layer path in this model
    source: str  # artifact path or package path
    source_layer: str  # dotted layer path in the source artifact
    sha256: str | None
    freeze: bool
    line: int


@dataclass
class TrainStmt:
    data: str
    loss: str
    optimizer: str
    lr: float
    epochs: int
    line: int
    batch: int | None = None
    validate: str | None = None
    weight_decay: float = 0.0


@dataclass
class Program:
    version: str
    model: str
    input: Port
    output: Port
    layers: list[Layer]
    datasets: dict[str, Dataset]
    seed: int
    imports: list[ImportStmt] = field(default_factory=list)
    train: TrainStmt | None = None
    path: str | None = None
    uses: dict[str, str] = field(default_factory=dict)  # alias -> library name

    def find(self, dotted: str) -> Layer | None:
        return find_layer(self.layers, dotted)


def find_layer(layers: list[Layer], dotted: str) -> Layer | None:
    head, _, rest = dotted.partition(".")
    for layer in layers:
        if layer.name == head:
            return layer if not rest else find_layer(layer.children, rest)
    return None


# ---------------------------------------------------------------- tokens


def _tokenize(text: str, line: int, path: str | None) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise WgtError(f"unexpected character {text[pos]!r}", line, path)
        kind = m.lastgroup
        assert kind is not None
        if kind != "ws":
            tokens.append((kind, m.group()))
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
        self.i = 0
        self.line = line
        self.path = path

    def error(self, message: str) -> WgtError:
        return WgtError(message, self.line, self.path)

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self, what: str) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise self.error(f"expected {what}, found end of line")
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
            raise self.error(f"unexpected {tok[1]!r} at end of statement")

    def at_end(self) -> bool:
        return self.peek() is None


def _number(text: str) -> int | float:
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def _parse_port(cur: _Cursor) -> tuple[str, str, tuple[int, ...]]:
    name = cur.ident("port name")
    cur.expect("punct", ":")
    dtype = cur.ident("dtype")
    if dtype not in DTYPES:
        raise cur.error(f"unsupported dtype {dtype!r}; WeightLang supports only f32 and i64 (i64 requires wgt 0.3)")
    cur.expect("punct", "[")
    dims = [cur.integer("dimension")]
    while cur.peek() == ("punct", ","):
        cur.next(",")
        dims.append(cur.integer("dimension"))
    cur.expect("punct", "]")
    cur.done()
    if any(dim <= 0 for dim in dims):
        raise cur.error("dimensions must be positive")
    return name, dtype, tuple(dims)


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
    line = cur.line
    name = cur.ident("layer operation or block")
    ns = None
    if cur.peek() == ("punct", "."):
        cur.next(".")
        ns, name = name, cur.ident("block name")
    args: dict[str, Value] = {}
    inner = None
    if cur.peek() == ("punct", "("):
        cur.next("(")
        if name == "residual" and ns is None:
            inner = _parse_spec(cur)
        else:
            while cur.peek() != ("punct", ")"):
                key = cur.ident("argument name")
                cur.expect("punct", "=")
                kind, text = cur.next(f"value for {key}")
                if kind == "number":
                    value: Value = _number(text)
                elif kind == "ident":
                    value = text
                else:
                    raise cur.error(f"invalid value {text!r} for {key}")
                if key in args:
                    raise cur.error(f"duplicate argument {key!r}")
                args[key] = value
                if cur.peek() == ("punct", ","):
                    cur.next(",")
        cur.expect("punct", ")")
    elif name == "residual" and ns is None:
        raise cur.error("residual needs a block: residual(<block>(...))")
    return Spec(name, ns, args, inner, line, cur.path)


# ---------------------------------------------------------------- files


class _FileParser:
    """Parses one file into a Scope plus, for model files, the model statements."""

    def __init__(self, source: str, path: str | None, loader: _Loader, *, library: bool):
        self.source = source
        self.path = path
        self.loader = loader
        self.library = library
        self.scope = Scope("<model>" if not library else (path or "<library>"), path)
        self.version: str | None = None
        self.model: str | None = None
        self.ports: dict[str, Port] = {}
        self.layer_specs: list[tuple[str, Spec]] = []
        self.datasets: dict[str, Dataset] = {}
        self.seed: int | None = None
        self.imports: list[ImportStmt] = []
        self.train: TrainStmt | None = None
        self.uses: dict[str, str] = {}

    def need_v02(self, what: str, line: int) -> None:
        if self.version == "0.1":
            raise WgtError(f"{what} requires 'wgt 0.2'", line, self.path)

    def run(self) -> None:
        open_data: Dataset | None = None
        open_block: Block | None = None
        for lineno, raw in enumerate(self.source.splitlines(), start=1):
            text = _strip_comment(raw)
            if not text.strip():
                continue
            if text[0] in " \t":
                cur = _Cursor(_tokenize(text, lineno, self.path), lineno, self.path)
                if open_data is not None:
                    x = _parse_vector(cur)
                    cur.expect("arrow", what="'->'")
                    y = _parse_vector(cur)
                    cur.done()
                    open_data.inputs.append(x)
                    open_data.targets.append(y)
                elif open_block is not None:
                    cur.expect("ident", "layer", what="'layer' (only layers belong in a block)")
                    name = cur.ident("layer name")
                    cur.expect("punct", ":")
                    spec = _parse_spec(cur)
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
            cur = _Cursor(_tokenize(text, lineno, self.path), lineno, self.path)
            keyword = cur.ident("statement keyword")
            if self.version is None and keyword != "wgt":
                raise cur.error("file must start with a version header: wgt 0.2")
            handler = getattr(self, f"_stmt_{keyword}", None)
            if handler is None:
                raise cur.error(f"unknown statement {keyword!r}")
            if self.library and keyword not in ("wgt", "use", "block"):
                raise cur.error(f"{keyword!r} is not allowed in a library file (only use and block)")
            result = handler(cur, lineno)
            if isinstance(result, Dataset) and not result.file:
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
        if kind in self.ports:
            raise cur.error(f"only one {kind} is supported")
        name, dtype, shape = _parse_port(cur)
        if self.version != "0.3" and (dtype != "f32" or len(shape) != 1):
            raise cur.error("only 1-D f32 ports in this version; this interface requires 'wgt 0.3'")
        self.ports[kind] = Port(name, dtype, shape, line)

    def _stmt_layer(self, cur: _Cursor, line: int) -> None:
        name = cur.ident("layer name")
        cur.expect("punct", ":")
        spec = _parse_spec(cur)
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
        cur.expect("punct", "(")
        while cur.peek() != ("punct", ")"):
            p = cur.ident("parameter name")
            if p in params:
                raise cur.error(f"duplicate parameter {p!r}")
            params.append(p)
            if cur.peek() == ("punct", ","):
                cur.next(",")
        cur.expect("punct", ")")
        cur.expect("punct", ":")
        cur.done()
        if name in BUILTIN_OPS or name == "residual":
            raise cur.error(f"block name {name!r} shadows a built-in op")
        if name in self.scope.blocks:
            raise cur.error(f"duplicate block {name!r}")
        block = Block(name, params, [], line, self.path, self.scope)
        self.scope.blocks[name] = block
        return block

    def _stmt_data(self, cur: _Cursor, line: int) -> Dataset:
        name = cur.ident("dataset name")
        cur.expect("punct", ":")
        kv = _parse_kv(cur)
        if name in self.datasets:
            raise cur.error(f"duplicate dataset {name!r}")
        unknown = set(kv) - {"file", "sha256", "skip", "license", "source"}
        if unknown:
            raise cur.error(f"unknown data option(s): {', '.join(sorted(unknown))}")
        if kv:
            self.need_v02("data options", line)
        ds = Dataset(name, [], [], line, license=kv.get("license"), source=kv.get("source"))
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
        if set(kv) != {"seed"} or not kv["seed"].isdigit():
            raise cur.error("init requires exactly seed=<non-negative int>")
        self.seed = int(kv["seed"])

    def _stmt_import(self, cur: _Cursor, line: int) -> None:
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

    def _stmt_train(self, cur: _Cursor, line: int) -> None:
        if self.train is not None:
            raise cur.error("only one train statement is supported")
        kv = _parse_kv(cur)
        required = {"data", "loss", "optimizer", "lr", "epochs"}
        optional = {"batch", "validate", "weight_decay"}
        missing = required - set(kv)
        if missing:
            raise cur.error(f"train is missing: {', '.join(sorted(missing))}")
        unknown = set(kv) - required - optional
        if unknown:
            raise cur.error(f"unknown train option(s): {', '.join(sorted(unknown))}")
        if kv["loss"] not in LOSSES:
            raise cur.error(f"unknown loss {kv['loss']!r}; supported: {', '.join(sorted(LOSSES))}")
        if kv["optimizer"] not in OPTIMIZERS:
            raise cur.error(f"unknown optimizer {kv['optimizer']!r}; supported: {', '.join(sorted(OPTIMIZERS))}")
        if kv["loss"] == "cross_entropy":
            self.need_v02("loss=cross_entropy", line)
        if kv["optimizer"] == "adamw":
            self.need_v02("optimizer=adamw", line)
        for opt in sorted(optional & set(kv)):
            self.need_v02(f"train option {opt}=", line)
        try:
            lr = float(kv["lr"])
            wd = float(kv.get("weight_decay", 0))
        except ValueError:
            raise cur.error("lr and weight_decay must be numbers") from None
        if not kv["epochs"].isdigit() or int(kv["epochs"]) <= 0:
            raise cur.error("epochs must be a positive integer")
        if lr <= 0:
            raise cur.error("lr must be positive")
        if wd < 0:
            raise cur.error("weight_decay must not be negative")
        batch = None
        if "batch" in kv:
            if not kv["batch"].isdigit() or int(kv["batch"]) <= 0:
                raise cur.error("batch must be a positive integer")
            batch = int(kv["batch"])
        self.train = TrainStmt(
            kv["data"], kv["loss"], kv["optimizer"], lr, int(kv["epochs"]), line, batch, kv.get("validate"), wd
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


def _value(v: Value, env: dict[str, int], spec: Spec) -> int | float:
    if isinstance(v, str):
        if v not in env:
            raise WgtError(f"unknown parameter {v!r}", spec.line, spec.path)
        return env[v]
    return v


def _resolve(name: str, spec: Spec, scope: Scope, env: dict[str, int], stack: tuple[str, ...]) -> Layer:
    if spec.op == "residual" and spec.ns is None:
        assert spec.inner is not None
        inner = _resolve(name, spec.inner, scope, env, stack)
        if inner.op != "seq":
            raise WgtError("residual(...) takes a block call, e.g. residual(nn.mlp(...))", spec.line, spec.path)
        return Layer(name, "residual", {}, spec.line, spec.path, inner.children, inner.block)

    if spec.ns is None and spec.op in BUILTIN_OPS:
        args = {**OP_DEFAULTS.get(spec.op, {}), **{k: _value(v, env, spec) for k, v in spec.args.items()}}
        signature = ARG_OPS.get(spec.op, {})
        if set(args) != set(signature):
            if not signature:
                raise WgtError(f"{spec.op} takes no arguments", spec.line, spec.path)
            want = ", ".join(f"{k}=<{t.__name__}>" for k, t in signature.items())
            raise WgtError(f"{spec.op} requires exactly ({want})", spec.line, spec.path)
        for k, t in signature.items():
            if t is int:
                minimum = 0 if k in ("padding", "causal") else 1
                if not isinstance(args[k], int) or args[k] < minimum:
                    raise WgtError(f"{spec.op} {k} must be a positive integer", spec.line, spec.path)
            else:
                args[k] = float(args[k])
        if spec.op == "self_attention":
            if args["dim"] % args["heads"] or args["causal"] not in (0, 1):
                raise WgtError("self_attention needs dim divisible by heads and causal=0 or 1", spec.line, spec.path)
        if spec.op == "dropout" and not 0 <= args["p"] < 1:
            raise WgtError("dropout p must be in [0, 1)", spec.line, spec.path)
        return Layer(name, spec.op, args, spec.line, spec.path)

    # Block call.
    if spec.ns is None:
        target = scope
        if spec.op not in target.blocks:
            raise WgtError(
                f"unknown layer operation or block {spec.op!r}; built-in ops: {', '.join(sorted(BUILTIN_OPS))}",
                spec.line,
                spec.path,
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
    if set(args) != set(block.params):
        raise WgtError(
            f"block {block.name} takes ({', '.join(block.params)}), got ({', '.join(args)})", spec.line, spec.path
        )
    int_args: dict[str, int] = {}
    for k, v in args.items():
        if not isinstance(v, int) or v <= 0:
            raise WgtError(f"block parameter {k} must be a positive integer", spec.line, spec.path)
        int_args[k] = v
    assert block.scope is not None
    children = [_resolve(lname, lspec, block.scope, int_args, stack + (label,)) for lname, lspec in block.body]
    return Layer(name, "seq", {}, spec.line, spec.path, children, label)


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
        elif layer.op == "layernorm":
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


def parse(source: str, path: str | None = None) -> Program:
    """Parse and statically check a model file."""

    fp = _FileParser(source, path, _Loader(), library=False)
    fp.run()
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
    for name, spec in fp.layer_specs:
        if fp.version == "0.1" and (spec.op in _V02_OPS or spec.op == "residual" or spec.ns is not None):
            raise WgtError(f"{spec.op!r} requires 'wgt 0.2'", spec.line, path)
        resolved = _resolve(name, spec, fp.scope, {}, ())

        def needs_v03(layer):
            return layer.op in _V03_OPS or any(needs_v03(c) for c in layer.children)

        if fp.version != "0.3" and needs_v03(resolved):
            raise WgtError(f"{spec.op!r} requires 'wgt 0.3'", spec.line, path)
        layers.append(resolved)

    for ds in fp.datasets.values():
        if ds.file:
            _load_csv(ds, math.prod(fp.ports["input"].shape), path)

    program = Program(
        version=fp.version or LANGUAGE_VERSION,
        model=fp.model,
        input=fp.ports["input"],
        output=fp.ports["output"],
        layers=layers,
        datasets=fp.datasets,
        seed=fp.seed,
        imports=fp.imports,
        train=fp.train,
        path=path,
        uses=fp.uses,
    )
    check(program)
    return program


def parse_library(source: str, path: str | None = None) -> Scope:
    """Parse a library file (``use`` and ``block`` only)."""

    fp = _FileParser(source, path, _Loader(), library=True)
    fp.run()
    return fp.scope


def _load_csv(ds: Dataset, input_width: int, path: str | None) -> None:
    base = Path(path).parent if path else Path(".")
    file = base / (ds.file or "")
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise WgtError(f"cannot read data file {ds.file}: {exc.strerror}", ds.line, path) from None
    actual = hashlib.sha256(raw).hexdigest()
    if actual != ds.sha256:
        raise WgtError(f"data file {ds.file} has sha256 {actual}, source pins {ds.sha256}", ds.line, path)
    lines = raw.decode("utf-8").splitlines()
    for n, row in enumerate(lines[ds.skip :], start=ds.skip + 1):
        row = row.strip()
        if not row or row.startswith("#"):
            continue
        try:
            values = [float(v) for v in row.split(",")]
        except ValueError:
            raise WgtError(f"{ds.file}:{n}: data files must contain only numbers", ds.line, path) from None
        if len(values) <= input_width:
            raise WgtError(f"{ds.file}:{n}: expected {input_width} inputs followed by targets", ds.line, path)
        ds.inputs.append(values[:input_width])
        ds.targets.append(values[input_width:])


def check(program: Program) -> None:
    """Shape and reference checks that do not need weights."""

    path = program.path
    if program.input.name == program.output.name:
        raise WgtError("input and output must have different names", program.output.line, path)
    if not any(layer.has_params for layer in program.layers):
        raise WgtError("model has no parameters (add at least one linear layer)", program.layers[0].line, path)
    if program.version == "0.3":
        dtype, shape = output_contract(program.layers, program.input.dtype, program.input.shape)
        if dtype != program.output.dtype or shape != program.output.shape:
            raise WgtError(
                f"shape/dtype mismatch: network produces {dtype}{list(shape)}, output declares "
                f"{program.output.dtype}{list(program.output.shape)}",
                program.output.line,
                path,
            )
    else:
        width = output_width(program.layers, program.input.shape[0])
        if width != program.output.shape[0]:
            raise WgtError(
                f"shape mismatch: network produces {width} features but output declares {program.output.shape[0]}",
                program.output.line,
                path,
            )
    input_size = math.prod(program.input.shape)
    output_size = math.prod(program.output.shape)

    for ds in program.datasets.values():
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

    train = program.train
    if train is None:
        frozen = [i for i in program.imports if i.freeze]
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

    if not has_trainable(program.layers, ""):
        raise WgtError("train has nothing to update: every parameter layer is frozen", train.line, path)


def _check_targets(program: Program, ds: Dataset, train: TrainStmt) -> None:
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


def parse_file(path: str | Path) -> Program:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise WgtError(f"cannot read source: {exc.strerror}", None, str(p)) from None
    return parse(text, str(p))


def is_library_source(text: str) -> bool:
    """A file without a model statement is treated as a library."""
    return not any(_strip_comment(line).split()[:1] == ["model"] for line in text.splitlines())


def output_contract(layers: list[Layer], dtype: str, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    """Static per-sample contracts: batch is always an optional leading runtime dimension."""
    for layer in layers:
        layer.input_spec = {"dtype": dtype, "shape": list(shape)}
        args, op = layer.args, layer.op

        def require(ok, message):
            if not ok:
                raise WgtError(f"{layer.name}: {message}", layer.line, layer.path)

        if op in STRUCTURAL_OPS:
            new_dtype, new_shape = output_contract(layer.children, dtype, shape)
            if op == "residual":
                require((new_dtype, new_shape) == (dtype, shape), "residual must preserve complete shape and dtype")
            dtype, shape = new_dtype, new_shape
        elif op == "embedding":
            require(dtype == "i64" and len(shape) == 1, "embedding expects i64[sequence]")
            dtype, shape = "f32", (*shape, int(args["dim"]))
        else:
            require(dtype == "f32", f"{op} expects f32; convert tokens with embedding first")
            if op == "linear":
                require(shape[-1] == args["in"], f"linear expects last dimension {args['in']}, got {shape[-1]}")
                shape = (*shape[:-1], int(args["out"]))
            elif op == "layernorm":
                require(shape[-1] == args["dim"], "layernorm dimension mismatch")
            elif op in ("self_attention", "position_embedding"):
                require(len(shape) == 2 and shape[-1] == args["dim"], f"{op} expects f32[sequence, dim]")
                if op == "position_embedding":
                    require(shape[0] <= args["max_len"], "sequence exceeds max_len")
            elif op == "meanpool":
                require(len(shape) == 2, "meanpool expects f32[sequence, dim]")
                shape = (shape[-1],)
            elif op == "flatten":
                shape = (math.prod(shape),)
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
        layer.output_spec = {"dtype": dtype, "shape": list(shape)}
    return dtype, shape
