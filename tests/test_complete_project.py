"""The documented project remains portable and produces a usable trained artifact."""

import json
from pathlib import Path
import shutil

import torch

from weightlang.artifact import load_artifact
from weightlang.compiler import compile_file
from weightlang.packages import parse_mod


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "complete_project"


def test_complete_project_after_relocation(tmp_path, monkeypatch):
    project = tmp_path / "sensor_project"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("weights", ".cache"))
    monkeypatch.chdir(tmp_path)
    config = parse_mod(project / "wgt.toml")
    assert config.requires == {}
    case = json.loads((project / "tests/calibration.json").read_text())
    built = compile_file(project / case["source"], project / "weights")
    assert built.path == project / "weights" / "sensor_calibration"
    loaded = load_artifact(built.path)
    assert set(loaded.module.state_dict()) == {
        "calibration.readout.linear.weight", "calibration.readout.linear.bias"
    }
    with torch.no_grad():
        actual = loaded.module(torch.tensor(case["input"], dtype=torch.float32))
        original = built.module(torch.tensor(case["input"], dtype=torch.float32))
    torch.testing.assert_close(actual, torch.tensor(case["expected"]), atol=case["atol"], rtol=0)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
