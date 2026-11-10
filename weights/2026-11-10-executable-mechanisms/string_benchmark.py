"""Frozen public PBE pilot: no test examples are passed to mechanism.learn."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import sys
import time

from mechanism import Budget, execute, learn

SEEDS = [11, 23, 37, 53, 71]
SOURCE_COMMIT = "0b9a62a294897b707c86ba3c0f63561ff503cb3b"


class String(str): pass


def parse(text):
    tokens, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace(): i += 1; continue
        if c == ";":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
            continue
        if c in "()": tokens.append(c); i += 1; continue
        if c == '"':
            value, i = [], i + 1
            while i < len(text):
                if text[i] == '"':
                    if i + 1 < len(text) and text[i + 1] == '"':
                        value.append('"'); i += 2; continue
                    i += 1; break
                value.append(text[i]); i += 1
            else: raise ValueError("unclosed string")
            tokens.append(String("".join(value))); continue
        start = i
        while i < len(text) and not text[i].isspace() and text[i] not in "();": i += 1
        tokens.append(text[start:i])
    stack, root = [], []
    current = root
    for token in tokens:
        if type(token) is str and token == "(":
            child = []; current.append(child); stack.append(current); current = child
        elif type(token) is str and token == ")":
            if not stack: raise ValueError("extra closing parenthesis")
            current = stack.pop()
        else: current.append(token)
    if stack: raise ValueError("unclosed parenthesis")
    return root


def task_from_file(path):
    forms = parse(path.read_text())
    synth = next((x for x in forms if isinstance(x, list) and x and x[0] == "synth-fun"), None)
    if synth is None or len(synth[2]) != 1 or synth[2][0][1] != "String" or synth[3] != "String":
        raise ValueError("not a single-String to String task")
    fname = synth[1]
    examples = []
    for form in forms:
        if not isinstance(form, list) or not form or form[0] != "constraint": continue
        equation = form[1]
        if len(equation) != 3 or equation[0] != "=": raise ValueError("unsupported constraint")
        left, right = equation[1:]
        if isinstance(left, String): left, right = right, left
        if not (isinstance(left, list) and len(left) == 2 and left[0] == fname
                and isinstance(left[1], String) and isinstance(right, String)):
            raise ValueError("constraint is not a literal example")
        examples.append((str(left[1]), str(right)))
    examples = list(dict.fromkeys(examples))
    if len({x for x, _ in examples}) != len(examples): raise ValueError("conflicting labels")
    if len(examples) < 3: raise ValueError("fewer than three distinct inputs")
    if max(max(len(x), len(y)) for x, y in examples) > 512: raise ValueError("exceeds 512-character input/output bound")
    # We deliberately ignore any define-fun reference solutions and source grammar.
    return {"name": path.name, "family": family(path.stem), "examples": examples,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def family(stem):
    previous = None
    while previous != stem:
        previous = stem
        stem = re.sub(r"[-_](long|short|small|repeat)$", "", stem)
    return stem


def load_tasks(directory):
    tasks, exclusions, seen = [], [], {}
    for path in sorted(directory.glob("*.sl")):
        try: task = task_from_file(path)
        except (ValueError, IndexError, TypeError) as exc:
            exclusions.append({"file": path.name, "reason": str(exc)}); continue
        key = tuple(sorted(task["examples"]))
        if key in seen:
            exclusions.append({"file": path.name, "reason": "identical example set", "duplicate_of": seen[key]}); continue
        seen[key] = path.name
        tasks.append(task)
    return tasks, exclusions


def summarize(rows):
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        rr = [r for r in rows if r["mode"] == mode]
        seeds = []
        for seed in SEEDS:
            sr = [r for r in rr if r["seed"] == seed]
            seeds.append(sum(r["all_test_correct"] for r in sr) / len(sr))
        times = [r["train_s"] for r in rr]
        summary[mode] = {
            "runs": len(rr), "train_consistent": sum(r["train_consistent"] for r in rr),
            "all_test_correct": sum(r["all_test_correct"] for r in rr),
            "fit_but_wrong": sum(r["train_consistent"] and not r["all_test_correct"] for r in rr),
            "no_fit": sum(not r["train_consistent"] for r in rr),
            "task_success_mean": statistics.mean(seeds), "task_success_seed_sd": statistics.stdev(seeds),
            "heldout_examples": sum(r["test_n"] for r in rr),
            "correct_examples": sum(r["test_correct"] for r in rr),
            "train_total_s": sum(times), "train_median_ms": 1000 * statistics.median(times),
            "generated": sum(r["stats"]["generated"] for r in rr),
            "executions": sum(r["stats"]["executions"] for r in rr),
            "matching_checks": sum(r["stats"]["matching_checks"] for r in rr),
            "library_checks": sum(r["stats"]["library_checks"] for r in rr),
            "model_bytes_median_fitted": statistics.median([r["model_bytes"] for r in rr if r["train_consistent"]] or [0]),
            "inference_us_median_fitted": statistics.median([r["inference_us"] for r in rr if r["train_consistent"]] or [0]),
            "seed_success_rates": seeds,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/euphony"))
    ap.add_argument("--output", type=Path, default=Path("results/string_results.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.output.exists() and not args.force: raise SystemExit("Output exists; choose a new path or --force")
    started = time.perf_counter()
    test, excluded = load_tasks(args.data / "test")
    training, train_excluded = load_tasks(args.data / "train")
    test_families = {t["family"] for t in test}
    test_pairs = {pair for t in test for pair in t["examples"]}
    allowed = []
    for task in training:
        reason = None
        if task["family"] in test_families: reason = "test task family overlap"
        elif test_pairs.intersection(task["examples"]): reason = "exact example pair overlap with test pool"
        if reason: train_excluded.append({"file": task["name"], "reason": reason})
        else: allowed.append(task)
    budget = Budget()
    config = {"seeds": SEEDS, "budget": asdict(budget), "test_tasks": len(test),
              "test_families": len(test_families), "library_training_tasks": len(allowed),
              "source_commit": SOURCE_COMMIT, "test_exclusions": excluded, "train_exclusions": train_excluded,
              "source_test_files": len(list((args.data / "test").glob("*.sl"))),
              "source_train_files": len(list((args.data / "train").glob("*.sl"))),
              "modes": ["cold_indexed", "cold_scan", "whole_program_cache", "memorization"],
              "library_mode": "whole learned programs only; no invented subroutine abstractions",
              "parser": "SMT-LIB literal examples only; grammar and reference programs ignored"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("string_protocol.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k:v for k,v in config.items() if not k.endswith("exclusions")}), flush=True)
    library, library_rows, identities = [], [], set()
    library_start = time.perf_counter()
    for task in allowed:
        train, validation = task["examples"][:-1], task["examples"][-1:]
        p, stats = learn(train, budget)
        valid = p is not None and all(execute(p, x) == y for x,y in validation)
        if valid:
            identity = json.dumps(p)
            if identity not in identities:
                library.append(p); identities.add(identity)
        library_rows.append({"file": task["name"], "accepted": valid, "stats": asdict(stats), "program": p})
    library_time = time.perf_counter() - library_start
    print(json.dumps({"library_programs":len(library), "library_discovery_s": library_time}), flush=True)
    rows = []
    for seed in SEEDS:
        for task in test:
            pairs = list(task["examples"])
            random.Random(str(seed) + ":" + task["name"]).shuffle(pairs)
            test_n = max(1, len(pairs) // 3)
            train, heldout = pairs[test_n:], pairs[:test_n]
            for mode in config["modes"]:
                if mode == "memorization":
                    memory = dict(train)
                    predictions = [memory.get(x) for x,y in heldout]
                    p, stats = None, None
                    rawstats = {"generated":0,"executions":0,"matching_checks":0,"library_checks":0}
                    train_time, consistent, size, latency = 0.0, True, len(json.dumps(memory).encode()), 0.0
                else:
                    p, stats = learn(train, budget, library if mode == "whole_program_cache" else (), mode != "cold_scan")
                    rawstats = asdict(stats)
                    consistent = p is not None and all(execute(p,x) == y for x,y in train)
                    if p is not None and not consistent: raise AssertionError("Learner returned an inconsistent model")
                    predictions = [execute(p, x) if p is not None else None for x,y in heldout]
                    train_time = stats.elapsed_s
                    size = len(json.dumps(p, ensure_ascii=False, separators=(",", ":")).encode()) if p is not None else 0
                    latency = 0.0
                    if p is not None:
                        before = time.perf_counter()
                        for _ in range(100):
                            for x,y in heldout: execute(p,x)
                        latency = (time.perf_counter()-before) * 1e6 / (100 * len(heldout))
                correct = sum(pred == y for pred, (_,y) in zip(predictions,heldout))
                rows.append({"file":task["name"],"family":task["family"],"seed":seed,"mode":mode,
                             "train_n":len(train),"test_n":len(heldout),"train_consistent":consistent,
                             "test_correct":correct,"all_test_correct":correct==len(heldout),
                             "train_s":train_time,"model_bytes":size,"inference_us":latency,
                             "stats":rawstats,"program":p,"train":train,"heldout":heldout,"predictions":predictions})
        print(json.dumps({"completed_seed":seed,"elapsed_s":time.perf_counter()-started}), flush=True)
    summary = summarize(rows)
    cache_total = library_time + summary["whole_program_cache"]["train_total_s"]
    cold_total = summary["cold_indexed"]["train_total_s"]
    result = {"protocol":config,"summary":summary,"library":library,
              "library_rows":library_rows,"library_discovery_s":library_time,
              "cache_including_discovery_s":cache_total,"cold_total_s":cold_total,
              "cache_total_cost_ratio":cache_total/cold_total,
              "rows":rows,"environment":{"python":sys.version,"platform":platform.platform(),
              "peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
              "total_wall_s":time.perf_counter()-started,
              "learner_sha256":hashlib.sha256(Path(__file__).with_name("mechanism.py").read_bytes()).hexdigest(),
              "benchmark_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"summary":summary,"cache_total_cost_ratio":result["cache_total_cost_ratio"],
                      "wall_s":result["total_wall_s"],"rss_kib":result["environment"]["peak_process_rss_kib"]},indent=2),flush=True)


if __name__ == "__main__": main()
