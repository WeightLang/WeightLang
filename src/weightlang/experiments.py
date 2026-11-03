"""Named target/regression evaluations with explicit, reproducible acceptance rules."""

from __future__ import annotations
import json
import math
from pathlib import Path

from .artifact import sha256_bytes, sha256_file
from .syntax import WgtError
from .weights import diff_weights, evaluate_artifact

_RULES = {"max_loss", "min_accuracy", "max_loss_increase", "max_accuracy_drop", "min_accuracy_gain"}


def run_experiment(candidate, baseline, suite_path):
    """Read-only evaluation. A failing target or regression case fails the suite."""
    path = Path(suite_path)
    try:
        raw = path.read_bytes()
        suite = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise WgtError(f"cannot read experiment suite: {exc}") from None
    if (
        not isinstance(suite, dict)
        or set(suite) != {"format", "format_version", "cases"}
        or suite["format"] != "weightlang.experiment"
        or type(suite["format_version"]) is not int
        or suite["format_version"] != 1
        or not isinstance(suite["cases"], list)
        or not suite["cases"]
    ):
        raise WgtError("invalid experiment suite; expected weightlang.experiment v1 with cases")
    names = set()
    roles = set()
    cases = []
    for case in suite["cases"]:
        if not isinstance(case, dict) or set(case) != {"name", "role", "data", "loss", "rules"}:
            raise WgtError("each case needs exactly name, role, data, loss and rules")
        name = case["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise WgtError("case names must be unique nonempty strings")
        names.add(name)
        if case["role"] not in ("target", "regression"):
            raise WgtError("case role must be target or regression")
        roles.add(case["role"])
        if not isinstance(case["data"], str) or not case["data"]:
            raise WgtError("case data must be a file path")
        if not isinstance(case["rules"], dict) or not case["rules"] or set(case["rules"]) - _RULES:
            raise WgtError("unknown or empty acceptance rules")
        for key, value in case["rules"].items():
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid or ("accuracy" in key and value > 1):
                raise WgtError(f"invalid threshold for {key}")
        cases.append(case)
    if roles != {"target", "regression"}:
        raise WgtError("experiment needs both target and regression cases")
    diff = diff_weights(baseline, candidate)
    reports = []
    for case in cases:
        report = evaluate_artifact(candidate, path.parent / case["data"], case["loss"], compare=baseline)
        checks = {}
        metrics, delta = report["candidate"], report["delta"]
        actual = {
            "max_loss": metrics["loss"],
            "max_loss_increase": delta["loss"],
        }
        if "accuracy" in metrics:
            actual |= {
                "min_accuracy": metrics["accuracy"],
                "max_accuracy_drop": -delta["accuracy"],
                "min_accuracy_gain": delta["accuracy"],
            }
        for key, threshold in case["rules"].items():
            if key not in actual:
                raise WgtError(f"{key} needs a classification loss")
            passed = actual[key] >= threshold if key.startswith("min_") else actual[key] <= threshold
            checks[key] = {"actual": actual[key], "threshold": threshold, "passed": passed}
        reports.append(
            {
                **report,
                "name": case["name"],
                "role": case["role"],
                "checks": checks,
                "passed": all(v["passed"] for v in checks.values()),
            }
        )
    return {
        "format": "weightlang.experiment-report",
        "format_version": 1,
        "suite": {"file": str(path), "sha256": sha256_bytes(raw)},
        "baseline_manifest_sha256": sha256_file(Path(baseline) / "manifest.json"),
        "candidate_manifest_sha256": sha256_file(Path(candidate) / "manifest.json"),
        "changes": diff,
        "cases": reports,
        "passed": all(case["passed"] for case in reports),
        "note": "Explicit thresholds on these datasets only; roles and dataset independence are user-declared.",
    }
