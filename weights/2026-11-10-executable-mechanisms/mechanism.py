"""A bounded, standard-library-only executable string learner.

This is a research probe of established inductive-programming ideas. Programs
are JSON ASTs, never eval'd Python. The grammar and finite budgets are explicit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import heapq
import itertools
import json
import re
import time
from pathlib import Path


@dataclass
class Budget:
    candidates: int = 25000
    checks: int = 25000
    max_parts: int = 8
    max_intermediates: int = 160
    max_output: int = 512
    library_checks: int = 128


@dataclass
class Stats:
    generated: int = 0
    executions: int = 0
    matching_checks: int = 0
    unique_behaviors: int = 0
    index_items: int = 0
    states: int = 0
    library_checks: int = 0
    elapsed_s: float = 0.0
    stopped: str = "exhausted"


def apply_op(op, s):
    name, *arg = op
    if name == "strip": return s.strip()
    if name == "lower": return s.lower()
    if name == "upper": return s.upper()
    if name == "title": return s.title()
    if name == "reverse": return s[::-1]
    if name == "space": return " ".join(s.split())
    if name == "digits": return "".join(c for c in s if c.isdigit())
    if name == "alpha": return "".join(c for c in s if c.isalpha())
    if name == "alnum": return "".join(c for c in s if c.isalnum())
    if name == "prefix": return s[:arg[0]]
    if name == "suffix": return s[-arg[0]:]
    if name == "drop_first": return s[arg[0]:]
    if name == "drop_last": return s[:-arg[0]]
    if name == "slice": return s[arg[0]:arg[1]]
    if name == "split":
        parts = s.split(arg[0]) if arg[0] else s.split()
        try: return parts[arg[1]]
        except IndexError: return ""
    if name == "before":
        i = s.find(arg[0]) if arg[1] == 0 else s.rfind(arg[0])
        return s[:i] if i >= 0 else s
    if name == "after":
        i = s.find(arg[0]) if arg[1] == 0 else s.rfind(arg[0])
        return s[i + len(arg[0]):] if i >= 0 else s
    if name == "replace": return s.replace(arg[0], arg[1])
    if name == "run":
        parts = re.findall(arg[0], s)
        try: return parts[arg[1]]
        except IndexError: return ""
    raise ValueError("Unknown primitive: " + name)


def execute(program, s):
    tag = program[0]
    if tag == "input": return s
    if tag == "literal": return program[1]
    if tag == "pipe":
        value = s
        for op in program[1]: value = apply_op(op, value)
        return value
    if tag == "concat": return "".join(execute(p, s) for p in program[1])
    raise ValueError("Unknown AST node")


def cost(program):
    """Simple structural preference, not an exact prefix-code description length."""
    if program[0] == "input": return 1
    if program[0] == "literal": return 2 + len(program[1].encode("utf-8"))
    if program[0] == "pipe":
        return 1 + sum(3 + sum(len(str(a)) for a in op[1:]) for op in program[1])
    return 1 + sum(cost(p) for p in program[1])


def operations(xs):
    # Only training inputs determine delimiter candidates. No held-out access.
    delimiters = sorted(set(" .,-_/@:;()[]\\") | {
        c for x in xs for c in x if not c.isalnum() and not c.isspace()
    })[:24]
    ops = [(n,) for n in ("strip", "lower", "upper", "title", "space",
                          "digits", "alpha", "alnum", "reverse")]
    for sep in [""] + delimiters:
        ops.extend(("split", sep, i) for i in (0, -1, 1, -2, 2))
        if sep:
            ops.extend((name, sep, side) for name in ("before", "after") for side in (0, 1))
            ops.extend(("replace", sep, new) for new in ("", " ", "-", "_") if new != sep)
    for pattern in (r"[0-9]+", r"[A-Za-z]+", r"[A-Za-z0-9]+"):
        ops.extend(("run", pattern, i) for i in (0, -1, 1, -2, 2))
    for n in range(1, 13):
        ops.extend((name, n) for name in ("prefix", "suffix", "drop_first", "drop_last"))
    for start in range(1, 7):
        for end in range(start + 1, 10): ops.append(("slice", start, end))
    return ops


def literals(ys):
    values = {"", " ", ".", ",", "-", "_", "/", ":", "@", "(", ")"}
    # Constants must be observed on all training targets; charge their byte size.
    first = ys[0]
    for start in range(len(first)):
        for end in range(start + 1, min(len(first), start + 24) + 1):
            text = first[start:end]
            if all(text in y for y in ys[1:]): values.add(text)
    if all(y == first for y in ys): values.add(first)
    return sorted(values, key=lambda x: (len(x), x))[:512]


def _compose(fragments, targets, budget, stats, indexed):
    """Find a low-cost concatenation by matching output residuals.

    A state stores how many output characters are already explained in EACH
    example. The index narrows edges by the first example; all examples verify
    an edge. A scan variant tests the same edges and differs only in indexing.
    """
    fragments = sorted(fragments.items(), key=lambda kv: (cost(kv[1]), repr(kv[1])))
    trie, empty = {}, []
    if indexed:
        for idx, (sig, _) in enumerate(fragments):
            stats.index_items += 1
            if not sig[0]: empty.append(idx); continue
            node = trie
            for char in sig[0]: node = node.setdefault(char, {})
            node.setdefault(None, []).append(idx)

    def edges(pos):
        if not indexed: return range(len(fragments))
        indices = list(empty)
        node = trie
        for char in targets[0][pos:]:
            node = node.get(char)
            if node is None: break
            indices.extend(node.get(None, ()))
        return sorted(indices)

    initial = (0,) * len(targets)
    goal = tuple(map(len, targets))
    serial = itertools.count()
    heap = [(0, next(serial), initial, ())]
    incumbent, incumbent_cost = None, float("inf")
    # Part count belongs in state: a cheap long path may not dominate a short one.
    best = {(initial, 0): 0}
    while heap:
        total, _, state, parts = heapq.heappop(heap)
        if total != best.get((state, len(parts))): continue
        stats.states += 1
        if state == goal: return ["concat", list(parts)]
        if len(parts) >= budget.max_parts: continue
        for idx in edges(state[0]):
            if stats.matching_checks >= budget.checks:
                stats.stopped = "matching_budget"
                return incumbent
            stats.matching_checks += 1
            signature, program = fragments[idx]
            if all(y.startswith(value, offset) for y, value, offset in zip(targets, signature, state)):
                nxt = tuple(offset + len(value) for offset, value in zip(state, signature))
                if nxt == state: continue
                value = total + cost(program) + 1
                if nxt == goal and value < incumbent_cost:
                    incumbent = ["concat", list(parts + (program,))]
                    incumbent_cost = value
                key = (nxt, len(parts) + 1)
                if value < best.get(key, float("inf")):
                    best[key] = value
                    heapq.heappush(heap, (value, next(serial), nxt, parts + (program,)))
    return None


def learn(examples, budget=None, library=(), indexed=True):
    """Return (program or None, counters). Examples are training data only."""
    budget = budget or Budget()
    stats = Stats()
    started = time.perf_counter()
    if not examples: raise ValueError("Need training examples")
    if any(len(x) > budget.max_output or len(y) > budget.max_output for x, y in examples):
        raise ValueError("This bounded probe supports at most 512 characters per string")
    if len({x for x, _ in examples}) != len({(x, y) for x, y in examples}):
        raise ValueError("Conflicting training examples")
    xs, ys = tuple(zip(*examples))
    seen, fragments, intermediates = {}, {}, []

    def finish(p, reason):
        stats.stopped = reason
        stats.unique_behaviors = len(seen)
        stats.elapsed_s = time.perf_counter() - started
        return p, stats

    for program in itertools.islice(library, budget.library_checks):
        stats.library_checks += 1
        stats.executions += len(xs)
        if tuple(execute(program, x) for x in xs) == ys:
            return finish(program, "library_fit")

    def register(program, signature):
        stats.generated += 1
        stats.executions += len(xs)
        old = seen.get(signature)
        if old is not None and cost(old) <= cost(program): return False
        seen[signature] = program
        if any(signature) and all(value in y for value, y in zip(signature, ys)):
            fragments[signature] = program
        return signature == ys

    if budget.candidates <= 0: return finish(None, "candidate_budget")
    if register(["input"], xs): return finish(["input"], "direct_fit")
    for literal in literals(ys):
        if stats.generated >= budget.candidates: break
        p = ["literal", literal]
        if register(p, (literal,) * len(xs)): return finish(p, "direct_fit")
    ops = operations(xs)
    for op in ops:
        if stats.generated >= budget.candidates: break
        sig = tuple(apply_op(op, x) for x in xs)
        p = ["pipe", [op]]
        novel = sig not in seen
        if register(p, sig): return finish(p, "direct_fit")
        if novel and any(sig): intermediates.append((p, sig))
    # Try simple concatenations before deepening the unary grammar.
    answer = _compose(fragments, ys, budget, stats, indexed)
    if answer is not None:
        stats.unique_behaviors = len(seen)
        return finish(answer, "concat_fit")
    for p, values in intermediates[:budget.max_intermediates]:
        for op in ops:
            if stats.generated >= budget.candidates: break
            sig = tuple(apply_op(op, x) for x in values)
            program = ["pipe", p[1] + [op]]
            if register(program, sig):
                stats.unique_behaviors = len(seen)
                return finish(program, "deep_fit")
        if stats.generated >= budget.candidates: break
    answer = _compose(fragments, ys, budget, stats, indexed)
    stats.unique_behaviors = len(seen)
    reason = "concat_fit" if answer is not None else stats.stopped
    if answer is None and stats.generated >= budget.candidates and reason != "matching_budget":
        reason = "candidate_budget"
    return finish(answer, reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("learn", help="learn a program from JSON [[input, output], ...]")
    fit.add_argument("examples", type=Path)
    fit.add_argument("--model", type=Path, required=True)
    infer = sub.add_parser("apply", help="apply an exported program to input strings")
    infer.add_argument("model", type=Path)
    infer.add_argument("text", nargs="+")
    args = parser.parse_args()
    if args.command == "learn":
        p, stats = learn(json.loads(args.examples.read_text()))
        if p is None: raise SystemExit("No consistent program within the declared budget")
        payload = {"format": "executable-mechanism-v1", "program": p,
                   "training_stats": asdict(stats), "status": "training_consistent_only"}
        with args.model.open("x") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        model = json.loads(args.model.read_text())
        if model.get("format") != "executable-mechanism-v1": raise SystemExit("Unknown model format")
        for text in args.text: print(execute(model["program"], text))


if __name__ == "__main__": main()
