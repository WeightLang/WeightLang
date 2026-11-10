"""Exploratory public-data test of learned fragment reuse; no source edits.

Protocol is frozen before this script evaluates any fragment-based prediction.
The source library is learned by the previously completed public-data pilot.
This probe uses no reference programs, neural model, or gradient optimization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

from mechanism import Budget, Stats, _compose, cost, execute, learn, literals


def identity(program):
    return json.dumps(program, ensure_ascii=False, separators=(",", ":"))


def extract_fragments(library):
    """Extract actual subtrees and nonempty prefixes; no reference solutions."""
    started = time.perf_counter()
    pool, provenance, visits = {}, {}, 0

    def add(p, where):
        key = identity(p)
        pool[key] = p
        provenance.setdefault(key, []).append(where)

    def visit(p, where):
        nonlocal visits
        visits += 1
        add(p, where)
        if p[0] == "concat":
            for i, child in enumerate(p[1]):
                visit(child, where + f"/concat:{i}")
        elif p[0] == "pipe":
            for n in range(1, len(p[1])):
                visits += 1
                add(["pipe", p[1][:n]], where + f"/prefix:{n}")

    for i, program in enumerate(library):
        visit(program, f"library:{i}")
    ordered = sorted(pool.values(), key=lambda p: (cost(p), identity(p)))
    return ordered, {"elapsed_s": time.perf_counter() - started,
                     "nodes_visited": visits, "unique_fragments": len(ordered),
                     "serialized_bytes": len(identity(ordered).encode()),
                     "provenance": provenance}


def solve(examples, budget, mode, library, extracted):
    """Use only training examples. Charge reuse against discovery budgets."""
    if mode == "cold":
        p, st = learn(examples, budget)
        return p, asdict(st), "cold/" + st.stopped
    started = time.perf_counter()
    xs, ys = tuple(zip(*examples))
    st = Stats()
    seen, fitting_fragments = {}, {}

    def finish(p, route):
        st.unique_behaviors = len(seen)
        st.elapsed_s = time.perf_counter() - started
        st.stopped = route
        return p, asdict(st), route

    candidates = library if mode == "whole_cache" else extracted
    candidates = candidates[:budget.library_checks]
    for program in candidates:
        if st.generated >= budget.candidates:
            return finish(None, mode + "/candidate_budget")
        st.generated += 1
        st.library_checks += 1
        st.executions += len(xs)
        sig = tuple(execute(program, x) for x in xs)
        if sig == ys:
            return finish(program, mode + "/direct_fit")
        old = seen.get(sig)
        if old is None or cost(program) < cost(old):
            seen[sig] = program
            if any(sig) and all(v in y for v, y in zip(sig, ys)):
                fitting_fragments[sig] = program

    if mode == "fragment_reuse":
        # Literal candidates are derived from training targets only, and charged.
        for text in literals(ys):
            if st.generated >= budget.candidates:
                break
            program, sig = ["literal", text], (text,) * len(xs)
            st.generated += 1
            st.executions += len(xs)
            if sig == ys:
                return finish(program, mode + "/literal_fit")
            if sig not in seen:
                seen[sig] = program
                if any(sig) and all(v in y for v, y in zip(sig, ys)):
                    fitting_fragments[sig] = program
        answer = _compose(fitting_fragments, ys, budget, st, True)
        if answer is not None:
            return finish(answer, "fragment_reuse/composed_fit")

    remaining = replace(budget, candidates=max(0, budget.candidates-st.generated),
                        checks=max(0, budget.checks-st.matching_checks))
    answer, fallback = learn(examples, remaining)
    for name in ("generated", "executions", "matching_checks", "index_items",
                 "states", "library_checks"):
        setattr(st, name, getattr(st, name) + getattr(fallback, name))
    # This is explicitly an aggregate, not a deduplicated union of both phases.
    aggregate_unique = len(seen) + fallback.unique_behaviors
    p, result, route = finish(answer, mode + "/fallback/" + fallback.stopped)
    result["unique_behaviors"] = aggregate_unique
    return p, result, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("results/string_results.json"))
    ap.add_argument("--protocol", type=Path, default=Path("results/fragment_protocol.json"))
    ap.add_argument("--output", type=Path, default=Path("results/fragment_results.json"))
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite existing results")
    protocol = json.loads(args.protocol.read_text())
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["source_results_sha256"]:
        raise SystemExit("Frozen source results changed")
    core_hash = hashlib.sha256(Path("mechanism.py").read_bytes()).hexdigest()
    if core_hash != protocol["learner_sha256"]:
        raise SystemExit("Frozen learner changed")
    source = json.loads(raw)
    budget = Budget(**protocol["budget"])
    library = source["library"]
    fragments, extraction = extract_fragments(library)
    cases = [r for r in source["rows"] if r["mode"] == "cold_indexed"]
    rows, started = [], time.perf_counter()
    for i, case in enumerate(cases):
        # Rotate timing order deterministically. No ordering uses test labels.
        modes = protocol["modes"]
        modes = modes[i % len(modes):] + modes[:i % len(modes)]
        for mode in modes:
            p, stats, route = solve(case["train"], budget, mode, library, fragments)
            train_ok = p is not None and all(execute(p, x) == y for x,y in case["train"])
            if p is not None and not train_ok:
                raise AssertionError("Inconsistent learned program")
            predictions = [execute(p,x) if p is not None else None for x,y in case["heldout"]]
            correct = sum(a == y for a,(_,y) in zip(predictions,case["heldout"]))
            rows.append({"mode":mode,"file":case["file"],"family":case["family"],
                         "seed":case["seed"],"train_n":case["train_n"],
                         "test_n":case["test_n"],"train_consistent":train_ok,
                         "all_test_correct":correct==case["test_n"],
                         "test_correct":correct,"route":route,"program":p,
                         "program_in_source_library":p is not None and any(p==q for q in library),
                         "stats":stats,"train_s":stats["elapsed_s"],
                         "heldout":case["heldout"],"predictions":predictions})
    discovery_s = source["library_discovery_s"]
    discovery_counts = {key:sum(r["stats"][key] for r in source["library_rows"])
                        for key in ("generated","executions","matching_checks","index_items","states")}
    by_mode = {}
    cold = {(r["file"],r["seed"]):r for r in rows if r["mode"]=="cold"}
    whole = {(r["file"],r["seed"]):r for r in rows if r["mode"]=="whole_cache"}
    for mode in protocol["modes"]:
        rr = [r for r in rows if r["mode"]==mode]
        total = sum(r["train_s"] for r in rr)
        discovery = discovery_s if mode!="cold" else 0.0
        extra = extraction["elapsed_s"] if mode=="fragment_reuse" else 0.0
        summary = {"runs":len(rr),"all_test_correct":sum(r["all_test_correct"] for r in rr),
                   "fit_but_wrong":sum(r["train_consistent"] and not r["all_test_correct"] for r in rr),
                   "no_fit":sum(not r["train_consistent"] for r in rr),
                   "correct_examples":sum(r["test_correct"] for r in rr),
                   "test_examples":sum(r["test_n"] for r in rr),
                   "test_task_learning_s":total,"source_discovery_s":discovery,
                   "source_fragment_extraction_s":extra,"lifetime_learning_s":total+discovery+extra,
                   "median_learning_ms":1000*statistics.median(r["train_s"] for r in rr),
                   "improved_vs_cold":sum(r["all_test_correct"] and not cold[r["file"],r["seed"]]["all_test_correct"] for r in rr),
                   "regressed_vs_cold":sum(not r["all_test_correct"] and cold[r["file"],r["seed"]]["all_test_correct"] for r in rr),
                   "improved_vs_whole":sum(r["all_test_correct"] and not whole[r["file"],r["seed"]]["all_test_correct"] for r in rr),
                   "regressed_vs_whole":sum(not r["all_test_correct"] and whole[r["file"],r["seed"]]["all_test_correct"] for r in rr),
                   "composed_routes":sum(r["route"]=="fragment_reuse/composed_fit" for r in rr),
                   "composed_and_heldout_correct":sum(r["route"]=="fragment_reuse/composed_fit" and r["all_test_correct"] for r in rr),
                   "composed_new_ast_and_correct":sum(r["route"]=="fragment_reuse/composed_fit" and r["all_test_correct"] and not r["program_in_source_library"] for r in rr)}
        for key in ("generated","executions","matching_checks","index_items","states","library_checks"):
            measured = sum(r["stats"][key] for r in rr)
            summary[key] = measured
            summary["lifetime_"+key] = measured + (discovery_counts.get(key,0) if mode!="cold" else 0)
        by_mode[mode] = summary
    output = {"protocol":protocol,"summary":by_mode,"fragments":fragments,
              "extraction":extraction,"source_discovery_counters":discovery_counts,
              "rows":rows,"evaluation_wall_s":time.perf_counter()-started,
              "environment":{"python":sys.version,"platform":platform.platform(),
                             "peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
              "probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256":hashlib.sha256(args.protocol.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({"summary":by_mode,"extraction":{k:v for k,v in extraction.items() if k!="provenance"},
                      "wall_s":output["evaluation_wall_s"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
