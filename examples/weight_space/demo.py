"""Run with the source checkout installed: python examples/weight_space/demo.py."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Directly program y = 2*x + 1 without gradient training")
    parser.add_argument("-o", "--output", type=Path, default=Path("build/weight-space"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"{root} already exists; choose a new directory")
    root.mkdir(parents=True)
    here = Path(__file__).resolve().parent

    def weightc(*arguments):
        return subprocess.check_output([sys.executable, "-m", "weightlang", *map(str, arguments)], text=True)

    print(weightc("build", here / "affine.wgt", "-o", root / "baseline"), end="")
    baseline = root / "baseline" / "affine"
    patch = json.loads(weightc("patch-template", baseline, "--author", "WeightLang affine demo"))
    patch["reason"] = "Explicitly implement y = 2*x + 1; no training or automated weight search"
    patch["edits"] = [
        {"tensor": "fc.weight", "index": 0, "value": 2},
        {"tensor": "fc.bias", "index": 0, "value": 1},
    ]
    patch_file = root / "patch.json"
    patch_file.write_text(json.dumps(patch, indent=2) + "\n")
    candidate = root / "candidate"
    print(weightc("patch", baseline, "--patch", patch_file, "-o", candidate), end="")
    for name, arguments in (
        ("diff", ("diff", baseline, candidate)),
        (
            "evaluation",
            ("evaluate", candidate, "--data", here / "evaluation.json", "--loss", "mse", "--compare", baseline),
        ),
    ):
        report = weightc(*arguments)
        (root / f"{name}.json").write_text(report)
        print(report, end="")
    print(f"Preview: wgt graph {candidate} --open")


if __name__ == "__main__":
    main()
