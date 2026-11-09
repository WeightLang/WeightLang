"""Measure dynamic SDK forwards using the package selected by PYTHONPATH.

Example from the repository root:
    PYTHONPATH=python .venv/bin/python python/benchmarks/runtime_checks.py \
        --label current --output weights/runtime-checks-current.json

Run the same script with PYTHONPATH pointing at another revision's python/
directory to compare actual implementations. Compilation and initialization
are excluded from timings; grad_forward measures forward graph construction,
not a complete training step. No runtime functions are patched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torch
from torch.utils.benchmark import Timer

import weightlang as wgt


ROOT = Path(__file__).resolve().parents[2]


def model_source(body: str, declarations: str) -> str:
    return (
        "wgt 0.5\nmodel RuntimeChecks\ninput x:f32[1]\noutput y:f32[1]\n"
        f"{declarations}\nfn forward(x) {{ {body} }}\n"
    )


def sources() -> dict[str, str]:
    # One scalar per sample; iterations share the same untrained gain parameter.
    workloads = {}
    for count in (3, 32):
        workloads[f"loop_{count}"] = model_source(
            f"let count=[{count}]; let value=x; while true {{ "
            "value=value*gain; count=count+[-1]; "
            "if positive(count) {} else { break; } } return value;",
            "parameter gain:f32[1]=[1.001]",
        )
    # Four active iterations, with eight candidate iterations available to learn.
    workloads["learned_depth_4_of_8"] = model_source(
        "return repeat depth carrying h=x { h*gain };",
        "learn depth:int in 0..=8=4;\nparameter gain:f32[1]=[1.01]",
    )
    return workloads


def source_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree", help="Revision or implementation label")
    parser.add_argument("--compiler", type=Path, help="Native wgt binary; defaults to WGT_COMPILER or target/debug/wgt")
    parser.add_argument("--min-run-time", type=float, default=0.2, help="Minimum measured seconds per case (default: 0.2)")
    parser.add_argument("--output", type=Path, help="Also save JSON to a new file; existing files are never overwritten")
    args = parser.parse_args()
    if args.min_run_time <= 0:
        parser.error("--min-run-time must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    compiler = (args.compiler or Path(os.environ.get("WGT_COMPILER", ROOT / "target/debug/wgt"))).resolve()
    if not compiler.is_file():
        parser.error(f"wgt compiler not found: {compiler}")
    os.environ["WGT_COMPILER"] = str(compiler)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    package = Path(wgt.__file__).resolve().parent
    result = {
        "label": args.label,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "package_path": str(package),
            "package_source_sha256": source_fingerprint(package),
            "compiler": str(compiler),
        },
        "method": {
            "timer": "torch.utils.benchmark.Timer.blocked_autorange",
            "min_run_time_seconds": args.min_run_time,
            "warmup_calls": 5,
            "input_shape": [1],
            "input_value": 1.0,
            "includes_compilation_or_initialization": False,
            "grad_forward_includes_backward": False,
        },
        "measurements": [],
    }
    for name, source in sources().items():
        model = wgt.loads(source, device="cpu")
        value = torch.ones(1, dtype=torch.float32)
        for grad_enabled in (False, True):
            with torch.set_grad_enabled(grad_enabled):
                for _ in range(5):
                    model(value)
                measurement = Timer(
                    stmt="model(value)", globals={"model": model, "value": value}, num_threads=1,
                ).blocked_autorange(min_run_time=args.min_run_time)
            result["measurements"].append({
                "workload": name,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "mode": "grad_forward" if grad_enabled else "no_grad_forward",
                "median_us": round(measurement.median * 1e6, 3),
                "iqr_us": round(measurement.iqr * 1e6, 3),
                "measurements": len(measurement.raw_times),
                "calls_per_measurement": measurement.number_per_run,
            })
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
