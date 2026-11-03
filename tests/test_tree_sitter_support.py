"""Exercise the editor parser and captures without importing the model runtime."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "editors/tree-sitter-wgt"
CLI = shutil.which("tree-sitter")
pytestmark = pytest.mark.skipif(CLI is None, reason="requires tree-sitter CLI 0.26+")


def run(*args, check=True):
    result = subprocess.run([CLI, *map(str, args)], cwd=GRAMMAR, capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    library = tmp_path_factory.mktemp("wgt-tree-sitter") / "wgt.so"
    run("build", GRAMMAR, "-o", library)
    return ["--lib-path", library, "--lang-name", "wgt"]


def test_corpus():
    run("test")


def test_repository_models(parser):
    files = []
    for directory in ("examples", "packages", "src/weightlang/std"):
        for path in (ROOT / directory).rglob("*.wgt"):
            source = path.read_text()
            # Separate language dialects: no claim of parsing their function bodies.
            if re.search(r"(?m)^\s*(backend\s+torch-dynamic|wgt\s+init\s+1)\b", source):
                continue
            files.append(path)
    assert files
    run("parse", *parser, "--quiet", *sorted(files))


def captures(parser, name, path):
    result = run("query", *parser, GRAMMAR / "queries" / name, path, "--captures")
    return re.findall(r"capture: \d+ - ([\w.]+), .*?text: `([^`]*)`", result.stdout, re.DOTALL)


def test_highlighting_outline_and_brackets(parser, tmp_path):
    source = tmp_path / "tokens.wgt"
    source.write_text('''wgt 0.3
model tiny
use "library#tag.wgt" as lib
# This is a comment, unlike the hash inside the path.
input x: f32[feature=4]
layer projection: linear(out=4) from input
layer custom: lib.linear(4)
init seed=42
''')
    tokens = captures(parser, "highlights.scm", source)
    for pair in [("keyword", "model"), ("type", "tiny"), ("string", '"library#tag.wgt"'),
                 ("type.builtin", "f32"), ("function.builtin", "linear"),
                 ("variable.parameter", "out"), ("number", "42"), ("function", "lib.linear")]:
        assert pair in tokens
    assert tokens.count(("function.builtin", "linear")) == 1
    assert [value for kind, value in tokens if kind == "comment"] == [
        "# This is a comment, unlike the hash inside the path."]
    outline = captures(parser, "outline.scm", source)
    assert [value for kind, value in outline if kind == "name"] == ["tiny", "x", "projection", "custom"]
    brackets = captures(parser, "brackets.scm", source)
    assert sum(kind == "open" for kind, _ in brackets) == 3
    assert sum(kind == "close" for kind, _ in brackets) == 3
    indents = captures(parser, "indents.scm", source)
    assert sum(kind == "indent" for kind, _ in indents) == 3


def test_crlf_eof_and_incremental_edit(parser, tmp_path):
    source = tmp_path / "edited.wgt"
    source.write_bytes(b"wgt 0.3\r\nmodel tiny\r\nlayer fc: linear(4)")
    run("parse", *parser, source, "--quiet")
    result = run("parse", *parser, source, "--edits", "2,17 0 , out=8")
    assert "parameter" in result.stdout


def test_unfinished_declaration_preserves_next_outline(parser, tmp_path):
    source = tmp_path / "unfinished.wgt"
    source.write_text("wgt 0.3\nlayer broken:\nmodel recovered\nlayer good: linear(4)\n")
    assert run("parse", *parser, source, check=False).returncode != 0
    outline = captures(parser, "outline.scm", source)
    assert ("name", "recovered") in outline
    assert ("name", "good") in outline


def test_generated_queries_are_current():
    result = subprocess.run(
        [sys.executable, ROOT / "scripts/generate_tree_sitter_queries.py", "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
