"""Run generated WGT programs from a saved experiment report; never execute training/hooks."""

import argparse
import json
from pathlib import Path

import torch

from weightlang.artifact import build_module
from weightlang.syntax import WgtError, parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists")
    rows = json.loads(args.report.read_text())["trained_test"]["rows"]
    results = []
    torch.set_num_threads(4)
    for row in rows:
        if row["kind"] != "code":
            continue
        result = {"prompt": row["prompt"], "runs": False, "interface_matches": False}
        try:
            program = parse(row["generated"])
            expected = parse(row["answer"])
            # This benchmark only permits a tiny, bounded operator set. No compiler hooks/imports.
            def check(layers):
                for layer in layers:
                    if layer["op"] not in {"seq", "linear", "relu", "sigmoid", "gelu", "silu", "tanh"}:
                        raise ValueError("outside bounded benchmark operator set")
                    if any(v > 32 for k, v in layer.get("args", {}).items() if k in {"in", "out"}):
                        raise ValueError("outside bounded benchmark dimensions")
                    check(layer.get("layers", []))
            layers = [layer.to_dict() for layer in program.layers]
            check(layers)
            if len(program.input.shape) != 1 or not 1 <= program.input.shape[0] <= 32:
                raise ValueError("outside benchmark input bounds")
            with torch.inference_mode():
                output = build_module(layers)(torch.zeros(1, *program.input.shape))
            result["runs"] = list(output.shape) == [1, *program.output.shape] and bool(torch.isfinite(output).all())
            result["interface_matches"] = (program.input.shape == expected.input.shape and
                                            program.output.shape == expected.output.shape)
        except (WgtError, ValueError, RuntimeError) as exc:
            result["error"] = str(exc)
        results.append(result)
    report = {"scope": "forward execution and shape contracts, not trained program correctness",
              "rows": results, "run_rate": sum(r["runs"] for r in results) / len(results),
              "interface_match_rate": sum(r["interface_matches"] for r in results) / len(results)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
