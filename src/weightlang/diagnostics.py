"""Structured diagnostics with source labels and a plain-text compiler renderer."""

from __future__ import annotations

import difflib
import unicodedata
from pathlib import Path


def source_text(path):
    try:
        return Path(path).read_text(encoding="utf-8") if path else ""
    except (OSError, UnicodeError):
        return ""


def label(path, line, message="", *, needle=None, source=None, column=None, end_column=None):
    text = source if source is not None else source_text(path)
    lines = text.splitlines()
    raw = lines[line - 1] if line and 0 < line <= len(lines) else ""
    automatic = column is None
    if column is None:
        index = raw.find(needle) if needle else -1
        column = index + 1 if index >= 0 else len(raw) - len(raw.lstrip()) + 1
    if end_column is None:
        end_column = column + len(needle) if needle and raw.find(needle) >= 0 else max(column + 1, len(raw) + 1)
    return {
        "path": path or "<source>",
        "line": line,
        "column": column,
        "end_column": end_column,
        "message": message,
        "source_line": raw,
        **({"_needle": needle, "_automatic": automatic} if not text else {}),
    }


class WgtError(Exception):
    """Keep the legacy exception string while carrying machine-readable diagnostics."""

    def __init__(self, message, line=None, path=None, *, code="E0001", labels=None, notes=None, help=None):
        self.message, self.line, self.path = message, line, path
        self.code = code
        self.labels = list(labels or [])
        self.notes = list(notes or [])
        self.help = help
        self.source = None
        super().__init__(str(self))

    def __str__(self):
        where = self.path or "<source>"
        if self.line is not None:
            where += f":{self.line}"
        return f"{where}: {self.message}"

    def as_dict(self):
        labels = self.labels or ([label(self.path, self.line, source=self.source)] if self.line else [])
        resolved = []
        for item in labels:
            item = dict(item)
            if self.source is not None and item["path"] == (self.path or "<source>") and item["line"]:
                if item.get("_automatic"):
                    item = label(
                        self.path, item["line"], item["message"], source=self.source, needle=item.get("_needle")
                    )
                else:
                    lines = self.source.splitlines()
                    if 0 < item["line"] <= len(lines):
                        item["source_line"] = lines[item["line"] - 1]
            resolved.append({key: value for key, value in item.items() if not key.startswith("_")})
        return {
            "format": "weightlang.diagnostic",
            "format_version": 1,
            "severity": "error",
            "code": self.code,
            "message": self.message,
            "labels": resolved,
            "notes": self.notes,
            "help": self.help,
        }

    def render(self):
        return render(self.as_dict())


def suggestion(name, names):
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.65)
    return f"did you mean `{matches[0]}`?" if matches else None


def _safe(text):
    # Source text must not be able to inject terminal control sequences.
    return "".join(c if c == "\t" or unicodedata.category(c)[0] != "C" else "�" for c in text)


def _width(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def render(diagnostic):
    out = [f"{diagnostic['severity']}[{diagnostic['code']}]: {_safe(diagnostic['message'])}"]
    for item in diagnostic.get("labels", []):
        if not item.get("line"):
            continue
        number = str(item["line"])
        gutter = " " * len(number)
        raw = _safe(item.get("source_line", ""))
        column = max(1, item.get("column", 1))
        end = max(column + 1, item.get("end_column", column + 1))
        prefix = raw[: column - 1].expandtabs(4)
        marked = raw[: end - 1].expandtabs(4)
        width = max(1, _width(marked) - _width(prefix))
        out.extend(
            [
                f" {gutter}--> {_safe(item['path'])}:{number}:{column}",
                f" {gutter} |",
                f" {number} | {raw.expandtabs(4)}",
                f" {gutter} | {' ' * _width(prefix)}{'^' * width} {_safe(item.get('message', ''))}".rstrip(),
            ]
        )
    for note in diagnostic.get("notes", []):
        out.append(f"  = note: {_safe(note)}")
    if diagnostic.get("help"):
        out.append(f"  = help: {_safe(diagnostic['help'])}")
    return "\n".join(out)


def warning(code, message, layer=None, help=None):
    return {
        "format": "weightlang.diagnostic",
        "format_version": 1,
        "severity": "warning",
        "code": code,
        "message": message,
        "labels": [label(layer.path, layer.line)] if layer else [],
        "notes": [],
        "help": help,
    }


EXPLANATIONS = {
    "E0001": "A source, data or artifact contract was violated. Read the message and source labels for the specific requirement.",
    "E0100": "Invalid syntax. The highlighted token does not fit the current statement. Check punctuation, argument names and the language version.",
    "E0201": "Unknown operation or tensor name. Tensor references must point to earlier layers in the same scope. Inside a block use input; inside a scan cell state is also available.",
    "E0301": "Incompatible tensor contracts. Compare the producer labels with the consumer's dimensions. add/mul/mix/where require identical shapes; concat joins only the last dimension.",
    "E0400": "Execution failed in a layer. Its source location and expansion call sites identify the failing operation. Check sample values, especially embedding indices.",
    "E0401": "An input, output or loss became NaN or infinity. The observer stops at the first invalid output in execution order. Inspect the marked layer and earlier values.",
    "E0402": "A gradient path is missing or training produced non-finite gradients/parameters. Check output connectivity, activations, targets and learning rate.",
    "E0404": "Invalid debug configuration or sample. Provide a sample matching the declared shape, or a train statement for bounded training. Debug currently supports CPU backprop only.",
    "E0405": "The model exceeds the debug parameter-allocation limit. Reduce its size or explicitly increase --max-parameters if sufficient memory is available.",
    "W0201": "A layer is not reachable from the final output. It still executes in source order, but cannot affect the result. Connect it to the output or remove it.",
    "W0401": "A trainable parameter received no gradient in at least one inspected step. This commonly indicates a disconnected branch.",
    "W0402": "A parameter gradient was exactly zero in at least one inspected step. This can be valid, for example with inactive ReLU units or zero inputs.",
    "W0403": "At least 95% of a sigmoid output was below .01 or above .99. Saturation may be deliberate; inspect the values before changing the architecture.",
}


def explain(args):
    if args.code not in EXPLANATIONS:
        raise WgtError(f"unknown diagnostic code {args.code!r}", help="available codes: " + ", ".join(EXPLANATIONS))
    print(f"{args.code}: {EXPLANATIONS[args.code]}")
    return 0
