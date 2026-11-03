import shutil
from pathlib import Path

import pytest

from weightlang.compiler import compile_file

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A copy of examples/ next to an empty weights/ directory."""
    shutil.copytree(EXAMPLES, tmp_path / "examples")
    (tmp_path / "weights").mkdir()
    return tmp_path


@pytest.fixture
def xor_build(workspace: Path):
    return compile_file(workspace / "examples" / "xor.wgt", workspace / "weights")
