"""Interleaved before/after benchmark for exact-byte Rust-backed weight serialization."""

import argparse
import ast
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import safetensors

from weightlang import artifact, compiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="548a0b4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("choose a new report path")
    source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:src/weightlang/artifact.py"], text=True)
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "write_safetensors")
    scope = dict(vars(artifact))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "baseline_writer", "exec"), scope)
    old = scope["write_safetensors"]
    new = artifact.write_safetensors
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tensors = {f"weight_{n}": torch.randn(128, 128) for n in range(4)}
    cache = Path('.cache/weightlang/native-perf')
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    original = compiler.write_safetensors
    try:
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            root = Path(directory)
            model = root / "model.wgt"
            model.write_text('wgt 0.3\nmodel serialization_bench\ninput x: f32[256]\noutput y: f32[256]\n'
                             'layer a: linear(in=256, out=256) -> relu\nlayer b: linear(in=256, out=256)\ninit seed=42\n')
            for workload in ("write_256KiB", "compile_two_linear_layers"):
                samples = {"before": [], "rust": []}
                hashes = {"before": [], "rust": []}
                for repeat in range(6):  # warm-up excluded, then five interleaved repetitions
                    for name, writer in (("before", old), ("rust", new)):
                        compiler.write_safetensors = writer
                        target = root / f'{workload}-{repeat}-{name}'
                        started = time.perf_counter()
                        if workload == "write_256KiB":
                            writer(target, tensors, {"benchmark": "identical bytes"})
                            file = target
                        else:
                            built = compiler.compile_file(model, target)
                            file = built.path / 'weights.safetensors'
                        elapsed = time.perf_counter() - started
                        raw = file.read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        assert safetensors.torch.load(raw)
                        if repeat:
                            samples[name].append(elapsed)
                            hashes[name].append(digest)
                assert len(set(hashes['before'] + hashes['rust'])) == 1
                results[workload] = {name: {"median_seconds": statistics.median(values),
                                           "range_seconds": [min(values), max(values)], "runs": values}
                                     for name, values in samples.items()}
                results[workload]['speedup'] = statistics.median(samples['before']) / statistics.median(samples['rust'])
                results[workload]['sha256'] = hashes['rust'][0]
    finally:
        compiler.write_safetensors = original
    report = {"baseline": args.baseline_ref, "python": platform.python_version(), "torch": torch.__version__,
              "safetensors": safetensors.__version__, "load_average": os.getloadavg(), "cpu_count": os.cpu_count(),
              "scope": "CPU serialization and compile only; warm filesystem cache, no fsync; not model training/inference throughput",
              "checks": {"identical_file_bytes": True, "all_files_load": True, "failures": 0},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
