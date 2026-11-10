"""Use the native WGT frontend; the SDK does not parse the source language."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

IR_FORMAT = "weightlang.torch.v1"


class CompilationError(RuntimeError):
    """The native compiler rejected a model or returned an invalid export."""


def _compiler(executable: str | os.PathLike[str] | None) -> str:
    candidate = os.fspath(executable) if executable is not None else os.environ.get("WGT_COMPILER", "wgt")
    found = shutil.which(candidate)
    if found is None:
        raise FileNotFoundError(
            f"WeightLang compiler {candidate!r} was not found. Install the current "
            "repository's wgt binary (cargo install --locked --path crates/wgt --bin wgt), "
            "or set WGT_COMPILER to its path. Precompiled JSON and saved SDK models "
            "can be loaded without the compiler."
        )
    return found


def _compile(
    source: str,
    *,
    text: str | None,
    compiler: str | os.PathLike[str] | None,
    timeout: float,
    dimensions: dict[str, int] | None,
    directory: Path | None = None,
) -> dict[str, Any]:
    command = [_compiler(compiler)]
    if directory is not None:
        command.extend(["-C", str(directory)])
    command.extend(["compile", source, "--target", "torch"])
    for name, value in (dimensions or {}).items():
        if not isinstance(name, str) or type(value) is not int or not 1 <= value <= 16384:
            raise ValueError("dimensions must map declared names to integers in 1..=16384")
        command.extend(["--dim", f"{name}={value}"])
    try:
        result = subprocess.run(
            command, input=text, capture_output=True, text=True, encoding="utf-8",
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CompilationError(f"WGT compilation exceeded {timeout:g} seconds") from error
    if result.returncode:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise CompilationError(diagnostic or f"WGT compiler exited with status {result.returncode}")
    try:
        definition = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CompilationError("WGT compiler returned invalid JSON; use a compiler with --target torch support") from error
    if not isinstance(definition, dict) or definition.get("format") != IR_FORMAT:
        raise CompilationError(f"WGT compiler must return {IR_FORMAT}")
    return definition


def compile(
    path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | None = None,
    timeout: float = 30,
    dimensions: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compile a model.wgt file to portable JSON data without executing forward.

    Relative pure-computation imports resolve beside the source file. No model
    service, optimizer, Python callback, or training loop is started.
    """
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Expected a WGT source file: {source}")
    return _compile(str(source), text=None, compiler=compiler, timeout=timeout, dimensions=dimensions, directory=source.parent)


def compile_source(
    source: str,
    *,
    compiler: str | os.PathLike[str] | None = None,
    timeout: float = 30,
    dimensions: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compile inline WGT source through stdin. Use compile(path) for local imports."""
    if not isinstance(source, str):
        raise TypeError("source must be a WGT source string")
    return _compile("-", text=source, compiler=compiler, timeout=timeout, dimensions=dimensions)
