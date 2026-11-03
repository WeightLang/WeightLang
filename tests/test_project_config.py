import pytest

from weightlang import WgtError
from weightlang.packages import find_mod, get, parse_mod


CONFIG = """[project]
module = "local/demo"
edition = "0.3"
[model]
source = "main.wgt"
[runtime]
device = "cpu"
[dependencies]
"github.com/example/repo" = "v1.2.3"
"""


def test_toml_dependency_discovery(tmp_path):
    path = tmp_path / "wgt.toml"
    path.write_text(CONFIG)
    child = tmp_path / "nested"
    child.mkdir()
    assert find_mod(child) == path
    config = parse_mod(tmp_path / "wgt.mod")
    assert config.path == path
    assert config.module == "local/demo"
    assert config.requires == {"github.com/example/repo": "v1.2.3"}


def test_empty_toml_get_uses_shared_lock(tmp_path):
    (tmp_path / "wgt.toml").write_text(CONFIG.split("[dependencies]")[0])
    assert get(tmp_path) == []
    assert (tmp_path / "wgt.sum").read_text().strip() == ""


def test_conflicting_manifests_rejected(tmp_path):
    (tmp_path / "wgt.toml").write_text(CONFIG)
    (tmp_path / "wgt.mod").write_text("module local/legacy\n")
    with pytest.raises(WgtError, match="both"):
        find_mod(tmp_path)


@pytest.mark.parametrize("version", ['"latest"', "123", '"v01.2.3"'])
def test_toml_invalid_dependency(tmp_path, version):
    (tmp_path / "wgt.toml").write_text(CONFIG.replace('"v1.2.3"', version))
    with pytest.raises(WgtError):
        parse_mod(tmp_path / "wgt.toml")
