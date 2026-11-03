import pytest

from weightlang.retention import project_limit
from weightlang.syntax import WgtError


def test_defaults_and_nested_project_override(tmp_path):
    assert project_limit(tmp_path) == 5
    config = tmp_path / "wgt.toml"
    config.write_text('[project]\nmodule = "local/test"\n')
    assert project_limit(tmp_path) == 5
    nested = tmp_path / "nested"
    nested.mkdir()
    for value in (0, 1, 12):
        config.write_text(f"[revisions]\nmax_versions = {value}\n")
        assert project_limit(nested) == value


@pytest.mark.parametrize("value", ['-1', 'true', '1.5', '"5"'])
def test_invalid_limits_fail(tmp_path, value):
    (tmp_path / "wgt.toml").write_text(f"[revisions]\nmax_versions = {value}\n")
    with pytest.raises(WgtError, match="nonnegative integer"):
        project_limit(tmp_path)
