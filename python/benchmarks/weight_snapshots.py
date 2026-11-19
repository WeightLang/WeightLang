"""Measure immutable snapshot reads and complete candidate plans on CPU.

Run from the repository with PYTHONPATH=python. Each invocation should use a
fresh process when comparing peak RSS; RSS includes imports and fixture setup.
Tracemalloc measures only allocations visible to Python, not native tensor
allocations. The fixed payload and parameter count are reported with timings.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import time
import tracemalloc

import torch
from weightlang.weights import Plan, WeightSet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("read", "plan"), required=True)
    parser.add_argument("--parameters", type=int, default=64)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if min(args.parameters, args.side, args.iterations) < 1:
        parser.error("parameters, side and iterations must be positive")
    torch.set_num_threads(1)
    values = {f"block_{i:03d}": torch.full((args.side, args.side), i / args.parameters)
              for i in range(args.parameters)}
    snapshot = WeightSet(values)
    del values
    specs = [{"name": name, "shape": [args.side, args.side], "data": [],
              "initializer": {"op": "unbound"}} for name in snapshot.names]
    definition = {"format": "weightlang.torch.v1", "kind": "forward",
                  "program": {"parameters": specs}}
    references = [snapshot.ref(name) for name in snapshot.names]
    def iteration() -> float:
        checksum = 0.
        if args.case == "read":
            for reference in references:
                tensor = reference.tensor()
                checksum += float(tensor[0, 0])
        else:
            candidate = Plan(definition, base=snapshot).build()
            assert candidate.revision == snapshot.revision
            checksum += float(candidate[references[-1].name][0, 0])
        return checksum

    gc.collect()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    checksum = sum(iteration() for _ in range(args.iterations))
    elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    # Allocation tracing changes object-heavy timings, so measure it separately.
    gc.collect()
    tracemalloc.start()
    iteration()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        rss /= 1024
    print(json.dumps({
        "case": args.case, "parameters": args.parameters, "shape": [args.side, args.side],
        "payload_values_bytes": args.parameters * args.side * args.side * 4,
        "iterations": args.iterations, "elapsed_seconds": elapsed,
        "cpu_seconds": cpu_elapsed, "seconds_per_iteration": elapsed / args.iterations,
        "peak_process_rss_kib": rss, "python_traced_peak_bytes": traced_peak,
        "checksum": checksum, "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
