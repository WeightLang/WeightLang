"""Parser and static checks for the WeightLang 0.1 source subset.

The format is line oriented: one statement per line, ``#`` starts a comment,
and the only multi-line construct is a ``data`` block whose rows are indented.
See ``docs/language.md`` for the full description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LANGUAGE_VERSION = "0.1"

ACTIVATIONS = {"relu", "tanh", "sigmoid"}
LAYER_OPS = {"linear"} | ACTIVATIONS
LOSSES = {"mse", "bce"}
OPTIMIZERS = {"sgd", "adam"}
DTYPES = {"f32"}

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<string>"[^"\\]*")
  | (?P<arrow>->)
  | (?P<number>-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.]))
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<word>[A-Za-z0-9_]+)
  | (?P<punct>[:=\[\](),])
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


@dataclass
class Port:
    name: str
    dtype: str
    shape: tuple[int, ...]
    line: int


@dataclass
class Layer:
    name: str
    op: str
    args: dict[str, int]
    line: int


@dataclass
class Dataset:
    name: str
    inputs: list[list[float]]
    targets: list[list[float]]
    line: int


@dataclass
class ImportStmt:
    target: str  # layer name in this model
    source: str  # artifact path as written in the source
    source_layer: str
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

    def layer(self, name: str) -> Layer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None


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
    # Keep '#' inside quoted strings (artifact paths may contain it).
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


def _parse_port(cur: _Cursor) -> tuple[str, str, tuple[int, ...]]:
    name = cur.ident("port name")
    cur.expect("punct", ":")
    dtype = cur.ident("dtype")
    if dtype not in DTYPES:
        raise cur.error(f"unsupported dtype {dtype!r}; WeightLang 0.1 supports only f32")
    cur.expect("punct", "[")
    dims = [cur.integer("dimension")]
    while cur.peek() == ("punct", ","):
        cur.next(",")
        dims.append(cur.integer("dimension"))
    cur.expect("punct", "]")
    cur.done()
    if len(dims) != 1:
        raise cur.error("WeightLang 0.1 supports only 1-D feature vectors, e.g. f32[4]")
    if dims[0] <= 0:
        raise cur.error("dimensions must be positive")
    return name, dtype, tuple(dims)


def _parse_kv(cur: _Cursor, flags: frozenset[str] = frozenset()) -> dict[str, str]:
    result: dict[str, str] = {}
    while not cur.at_end():
        key = cur.ident("key")
        if key in flags and cur.peek() != ("punct", "="):
            value = "true"
        else:
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


def parse(source: str, path: str | None = None) -> Program:
    """Parse and statically check a .wgt program."""

    version: str | None = None
    model: str | None = None
    ports: dict[str, Port] = {}
    layers: list[Layer] = []
    datasets: dict[str, Dataset] = {}
    seed: int | None = None
    imports: list[ImportStmt] = []
    train: TrainStmt | None = None
    open_data: Dataset | None = None

    for lineno, raw in enumerate(source.splitlines(), start=1):
        text = _strip_comment(raw)
        if not text.strip():
            continue
        indented = text[0] in " \t"
        if indented:
            if open_data is None:
                raise WgtError("unexpected indentation (only data rows are indented)", lineno, path)
            cur = _Cursor(_tokenize(text, lineno, path), lineno, path)
            x = _parse_vector(cur)
            cur.expect("arrow", what="'->'")
            y = _parse_vector(cur)
            cur.done()
            open_data.inputs.append(x)
            open_data.targets.append(y)
            continue
        open_data = None
        cur = _Cursor(_tokenize(text, lineno, path), lineno, path)
        keyword = cur.ident("statement keyword")

        if version is None and keyword != "wgt":
            raise cur.error("file must start with a version header: wgt 0.1")

        if keyword == "wgt":
            if version is not None:
                raise cur.error("duplicate wgt header")
            version = cur.expect("number", what="language version")
            cur.done()
            if version != LANGUAGE_VERSION:
                raise cur.error(f"unsupported language version {version}; this weightc supports {LANGUAGE_VERSION}")
        elif keyword == "model":
            if model is not None:
                raise cur.error("only one model per file is supported")
            model = cur.ident("model name")
            cur.done()
        elif keyword in ("input", "output"):
            if keyword in ports:
                raise cur.error(f"only one {keyword} is supported")
            name, dtype, shape = _parse_port(cur)
            ports[keyword] = Port(name, dtype, shape, lineno)
        elif keyword == "layer":
            name = cur.ident("layer name")
            cur.expect("punct", ":")
            op = cur.ident("layer operation")
            if op not in LAYER_OPS:
                raise cur.error(f"unknown layer operation {op!r}; supported: {', '.join(sorted(LAYER_OPS))}")
            args: dict[str, int] = {}
            if cur.peek() == ("punct", "("):
                cur.next("(")
                while cur.peek() != ("punct", ")"):
                    key = cur.ident("argument name")
                    cur.expect("punct", "=")
                    if key in args:
                        raise cur.error(f"duplicate argument {key!r}")
                    args[key] = cur.integer(key)
                    if cur.peek() == ("punct", ","):
                        cur.next(",")
                cur.expect("punct", ")")
            cur.done()
            if op == "linear":
                if set(args) != {"in", "out"}:
                    raise cur.error("linear requires exactly (in=<int>, out=<int>)")
                if args["in"] <= 0 or args["out"] <= 0:
                    raise cur.error("linear sizes must be positive")
            elif args:
                raise cur.error(f"{op} takes no arguments")
            if any(layer.name == name for layer in layers):
                raise cur.error(f"duplicate layer name {name!r}")
            layers.append(Layer(name, op, args, lineno))
        elif keyword == "data":
            name = cur.ident("dataset name")
            cur.expect("punct", ":")
            cur.done()
            if name in datasets:
                raise cur.error(f"duplicate dataset {name!r}")
            open_data = Dataset(name, [], [], lineno)
            datasets[name] = open_data
        elif keyword == "init":
            if seed is not None:
                raise cur.error("duplicate init statement")
            kv = _parse_kv(cur)
            if set(kv) != {"seed"} or not kv["seed"].isdigit():
                raise cur.error("init requires exactly seed=<non-negative int>")
            seed = int(kv["seed"])
        elif keyword == "import":
            target = cur.ident("layer name")
            cur.expect("ident", "from", what="'from'")
            src = cur.expect("string", what="quoted artifact path")[1:-1]
            kv = _parse_kv(cur, flags=frozenset({"freeze"}))
            unknown = set(kv) - {"layer", "sha256", "freeze"}
            if unknown:
                raise cur.error(f"unknown import option(s): {', '.join(sorted(unknown))}")
            sha = kv.get("sha256")
            if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise cur.error("sha256 must be 64 lowercase hex characters")
            if any(i.target == target for i in imports):
                raise cur.error(f"layer {target!r} is imported twice")
            imports.append(ImportStmt(target, src, kv.get("layer", target), sha, kv.get("freeze") == "true", lineno))
        elif keyword == "train":
            if train is not None:
                raise cur.error("only one train statement is supported")
            kv = _parse_kv(cur)
            required = {"data", "loss", "optimizer", "lr", "epochs"}
            missing = required - set(kv)
            if missing:
                raise cur.error(f"train is missing: {', '.join(sorted(missing))}")
            unknown = set(kv) - required
            if unknown:
                raise cur.error(f"unknown train option(s): {', '.join(sorted(unknown))}")
            if kv["loss"] not in LOSSES:
                raise cur.error(f"unknown loss {kv['loss']!r}; supported: {', '.join(sorted(LOSSES))}")
            if kv["optimizer"] not in OPTIMIZERS:
                raise cur.error(f"unknown optimizer {kv['optimizer']!r}; supported: {', '.join(sorted(OPTIMIZERS))}")
            try:
                lr = float(kv["lr"])
            except ValueError:
                raise cur.error("lr must be a number") from None
            if not kv["epochs"].isdigit() or int(kv["epochs"]) <= 0:
                raise cur.error("epochs must be a positive integer")
            if lr <= 0:
                raise cur.error("lr must be positive")
            train = TrainStmt(kv["data"], kv["loss"], kv["optimizer"], lr, int(kv["epochs"]), lineno)
        else:
            raise cur.error(f"unknown statement {keyword!r}")

    if version is None:
        raise WgtError("empty file; expected header: wgt 0.1", None, path)
    if model is None:
        raise WgtError("missing 'model <name>' statement", None, path)
    for kw in ("input", "output"):
        if kw not in ports:
            raise WgtError(f"missing '{kw}' statement", None, path)
    if not layers:
        raise WgtError("model has no layers", None, path)
    if seed is None:
        raise WgtError("missing 'init seed=<int>'; every parameter must have an explicit initialization", None, path)

    program = Program(
        version=version,
        model=model,
        input=ports["input"],
        output=ports["output"],
        layers=layers,
        datasets=datasets,
        seed=seed,
        imports=imports,
        train=train,
        path=path,
    )
    check(program)
    return program


def check(program: Program) -> None:
    """Shape and reference checks that do not need weights."""

    path = program.path
    if program.input.name == program.output.name:
        raise WgtError("input and output must have different names", program.output.line, path)

    width = program.input.shape[0]
    has_linear = False
    for layer in program.layers:
        if layer.op == "linear":
            has_linear = True
            if layer.args["in"] != width:
                raise WgtError(
                    f"shape mismatch: layer {layer.name!r} expects in={layer.args['in']} but receives {width}",
                    layer.line,
                    path,
                )
            width = layer.args["out"]
    if not has_linear:
        raise WgtError("model has no parameters (add at least one linear layer)", program.layers[0].line, path)
    if width != program.output.shape[0]:
        raise WgtError(
            f"shape mismatch: network produces {width} features but output declares {program.output.shape[0]}",
            program.output.line,
            path,
        )

    for ds in program.datasets.values():
        if not ds.inputs:
            raise WgtError(f"dataset {ds.name!r} has no rows", ds.line, path)
        for x, y in zip(ds.inputs, ds.targets):
            if len(x) != program.input.shape[0] or len(y) != program.output.shape[0]:
                raise WgtError(
                    f"dataset {ds.name!r} rows must be [{program.input.shape[0]} values] -> "
                    f"[{program.output.shape[0]} values]",
                    ds.line,
                    path,
                )

    for imp in program.imports:
        layer = program.layer(imp.target)
        if layer is None:
            raise WgtError(f"import targets unknown layer {imp.target!r}", imp.line, path)
        if layer.op != "linear":
            raise WgtError(f"import target {imp.target!r} has no parameters", imp.line, path)

    if program.train is not None:
        if program.train.data not in program.datasets:
            raise WgtError(f"train refers to unknown dataset {program.train.data!r}", program.train.line, path)
        trainable = [
            layer
            for layer in program.layers
            if layer.op == "linear" and not any(i.target == layer.name and i.freeze for i in program.imports)
        ]
        if not trainable:
            raise WgtError("train has nothing to update: every parameter layer is frozen", program.train.line, path)
    else:
        frozen = [i for i in program.imports if i.freeze]
        if frozen:
            raise WgtError("freeze only makes sense together with a train statement", frozen[0].line, path)


def parse_file(path: str | Path) -> Program:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise WgtError(f"cannot read source: {exc.strerror}", None, str(p)) from None
    return parse(text, str(p))
