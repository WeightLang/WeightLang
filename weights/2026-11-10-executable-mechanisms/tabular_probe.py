#!/usr/bin/env python3
"""A modest falsification probe for inexpensive discrete prediction.

The core CostTree below uses only the Python standard library: comparisons,
counts, sorting, and scalar logarithms. It is an established greedy decision-tree
design with a description-length-inspired split penalty, NOT a novel intelligence
theory. NumPy/scikit-learn are used only for bundled real data, reference models,
splits, and evaluation. No gradient or matrix multiplication trains CostTree.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tabular_probe.py

Protocol is fixed before this script's first evaluation: five seeds, stratified
25% outer test; a stratified 25% validation split within the remaining training
set chooses hyperparameters. The selected model is refitted on all outer train.
Test data are never passed to the learner or parameter-selection routine.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pickle
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


SEEDS = [11, 23, 37, 53, 71]
OUTER_TEST_FRACTION = 0.25
INNER_VALIDATION_FRACTION = 0.25
TREE_GRID = [
    {"penalty": p, "min_leaf": m, "max_depth": 8}
    for p, m in itertools.product([0.25, 0.5, 1.0, 2.0], [2, 5])
]
CART_GRID = [
    {"max_depth": d, "min_samples_leaf": m}
    for d, m in itertools.product([3, 5, 8, None], [1, 5])
]
FOREST_GRID = [
    {"max_depth": d, "min_samples_leaf": m, "max_features": f}
    for d, m, f in itertools.product([5, None], [1, 3], ["sqrt", None])
]


class CostTree:
    """Greedy threshold tree with an approximate conditional code-length score.

    A leaf's data score is n*H(empirical class counts). Each candidate split is
    charged penalty*(1 + log2(number of features) + log2(n-1)) bits. X is treated
    as known side information when interpreting a threshold by its sorted index.
    This is a heuristic: it omits parameter coding and does not optimize a
    complete, decodable MDL code or a global tree objective.
    """

    def __init__(self, penalty=1.0, min_leaf=2, max_depth=8):
        self.penalty = penalty
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.candidate_count = 0
        self.node_count = 0

    @staticmethod
    def _bits(counts, n):
        if not n:
            return 0.0
        return n * math.log2(n) - sum(c * math.log2(c) for c in counts if c)

    def fit(self, x, y):
        if not x or len(x) != len(y):
            raise ValueError("nonempty aligned x and y required")
        self.classes = sorted(set(y))
        self.class_index = {value: i for i, value in enumerate(self.classes)}
        self.n_features = len(x[0])
        if not self.n_features or any(len(row) != self.n_features for row in x):
            raise ValueError("rectangular nonempty feature rows required")
        if any(not math.isfinite(value) for row in x for value in row):
            raise ValueError("this probe requires finite numerical features")
        self.candidate_count = self.node_count = 0
        encoded = [self.class_index[v] for v in y]

        def build(indices, depth):
            self.node_count += 1
            counts = [0] * len(self.classes)
            for i in indices:
                counts[encoded[i]] += 1
            winner = max(range(len(counts)), key=lambda k: counts[k])
            leaf = {"class": self.classes[winner], "n": len(indices)}
            n = len(indices)
            if depth >= self.max_depth or n < 2 * self.min_leaf or max(counts) == n:
                return leaf
            best_score = self._bits(counts, n)
            best = None
            charge = self.penalty * (1.0 + math.log2(self.n_features) + math.log2(n - 1))
            for feature in range(self.n_features):
                ordered = sorted(indices, key=lambda i: x[i][feature])
                left = [0] * len(counts)
                right = counts[:]
                for pos in range(n - 1):
                    idx = ordered[pos]
                    label = encoded[idx]
                    left[label] += 1
                    right[label] -= 1
                    nl = pos + 1
                    nr = n - nl
                    lo = x[idx][feature]
                    hi = x[ordered[pos + 1]][feature]
                    if nl < self.min_leaf or nr < self.min_leaf or lo == hi:
                        continue
                    self.candidate_count += 1
                    score = self._bits(left, nl) + self._bits(right, nr) + charge
                    if score < best_score - 1e-12:
                        best_score = score
                        best = (feature, lo + (hi - lo) / 2.0, ordered[:nl], ordered[nl:])
            if best is None:
                return leaf
            feature, threshold, left_idx, right_idx = best
            return {"feature": feature, "threshold": threshold,
                    "left": build(left_idx, depth + 1), "right": build(right_idx, depth + 1)}

        self.root = build(list(range(len(x))), 0)
        return self

    def predict(self, x):
        result = []
        for row in x:
            node = self.root
            while "class" not in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            result.append(node["class"])
        return result


def checksum(x, y):
    """Hash a stable JSON representation of the exact numerical dataset."""
    data = json.dumps({"x": x.tolist(), "y": y.tolist()}, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def node_count(model):
    if isinstance(model, CostTree):
        return model.node_count
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    if hasattr(model, "estimators_"):
        return sum(int(tree.tree_.node_count) for tree in model.estimators_)
    return 1


def run(output_dir):
    import numpy as np
    import sklearn
    from sklearn.datasets import load_breast_cancer, load_wine
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    loaders = {
        "wine": (load_wine, "https://archive.ics.uci.edu/dataset/109/wine"),
        "breast_cancer_wisconsin_diagnostic": (
            load_breast_cancer, "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")}
    protocol = {
        "seeds": SEEDS, "outer_test_fraction": OUTER_TEST_FRACTION,
        "inner_validation_fraction_of_outer_train": INNER_VALIDATION_FRACTION,
        "stratified": True, "selection_metric": "validation_macro_f1",
        "tie_break": "fewer nodes, then fixed grid order",
        "grids": {"cost_tree": TREE_GRID, "cart": CART_GRID, "random_forest": FOREST_GRID},
        "forest_trees": 100, "forest_jobs": 1,
        "inference_timing": "median of 31 whole test-batch predictions / test size, including call overhead",
        "serialized_size": "pickle protocol 5; implementation-dependent, not fair compression ratio",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Written before any test model is evaluated.
    (output_dir / "tabular_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    report = {
        "protocol": protocol,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "sklearn": sklearn.__version__, "numpy": np.__version__,
                        "logical_cpus": os.cpu_count(), "gpu_used": False,
                        "threads_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}},
        "datasets": {}, "runs": [], "summary": [],
        "limitations": [
            "CostTree is an established greedy decision-tree family, not a novel intelligence theory.",
            "Its split cost is MDL-inspired, not a complete decodable MDL code or a global optimum.",
            "Tiny tabular datasets do not establish general intelligence, causality, language ability, or continual learning.",
            "Python and compiled sklearn implementations differ; timings measure these implementations, not fundamental algorithm limits.",
            "Repeated holdouts overlap and are not independent trials; dispersion is descriptive, not a confidence interval.",
            "Hyperparameter search cost is included separately from final fit; no test-set tuning is performed.",
            "Medical dataset is an algorithm benchmark only, with no clinical validation or recommendation.",
        ]}
    total_start = time.perf_counter()
    for dataset_name, (loader, source_url) in loaders.items():
        dataset = loader()
        x, y = dataset.data, dataset.target
        report["datasets"][dataset_name] = {
            "rows": len(y), "features": x.shape[1], "classes": len(set(y.tolist())),
            "class_counts": dict(Counter(map(int, y))), "source_url": source_url,
            "sha256_canonical_numeric_json": checksum(x, y),
            "feature_names": list(dataset.feature_names),
            "loader": f"sklearn.datasets.{loader.__name__}",
        }
        for seed in SEEDS:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=OUTER_TEST_FRACTION, random_state=seed, stratify=y)
            xi, xv, yi, yv = train_test_split(
                x_train, y_train, test_size=INNER_VALIDATION_FRACTION,
                random_state=seed + 1000, stratify=y_train)
            for model_name, grid in [
                ("dummy", [{}]), ("cost_tree", TREE_GRID),
                ("cart", CART_GRID), ("random_forest", FOREST_GRID)]:
                def create(params):
                    if model_name == "cost_tree":
                        return CostTree(**params)
                    if model_name == "cart":
                        return DecisionTreeClassifier(random_state=seed, **params)
                    if model_name == "random_forest":
                        return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed, **params)
                    return DummyClassifier(strategy="most_frequent", random_state=seed)

                if model_name == "cost_tree":
                    a, b, c, d = xi.tolist(), xv.tolist(), x_train.tolist(), x_test.tolist()
                    ia, ic = yi.tolist(), y_train.tolist()
                else:
                    a, b, c, d, ia, ic = xi, xv, x_train, x_test, yi, y_train
                search_start = time.perf_counter()
                best = None
                for rank, params in enumerate(grid):
                    model = create(params).fit(a, ia)
                    score = float(f1_score(yv, model.predict(b), average="macro", zero_division=0))
                    key = (score, -node_count(model), -rank)
                    if best is None or key > best[0]:
                        best = (key, params)
                tuning_seconds = time.perf_counter() - search_start
                selected_params = best[1]
                model = create(selected_params)
                fit_start = time.perf_counter()
                model.fit(c, ic)
                fit_seconds = time.perf_counter() - fit_start
                pred = model.predict(d)
                times = []
                for _ in range(31):
                    start = time.perf_counter_ns()
                    model.predict(d)
                    times.append(time.perf_counter_ns() - start)
                run_record = {
                    "dataset": dataset_name, "seed": seed, "model": model_name,
                    "train_n": len(y_train), "test_n": len(y_test),
                    "selected_parameters": selected_params,
                    "validation_macro_f1": best[0][0],
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                    "tuning_seconds": tuning_seconds, "final_fit_seconds": fit_seconds,
                    "total_training_seconds": tuning_seconds + fit_seconds,
                    "predict_us_per_sample": statistics.median(times) / 1000 / len(y_test),
                    "node_count": node_count(model),
                    "serialized_pickle_bytes": len(pickle.dumps(model, protocol=5)),
                    "threshold_candidates_final_fit": getattr(model, "candidate_count", None),
                }
                report["runs"].append(run_record)
                print(f"{dataset_name} seed={seed} {model_name}: acc={run_record['accuracy']:.4f} "
                      f"F1={run_record['macro_f1']:.4f} train={run_record['total_training_seconds']:.4f}s "
                      f"nodes={run_record['node_count']}", flush=True)
    for dataset_name in loaders:
        for model_name in ["dummy", "cost_tree", "cart", "random_forest"]:
            group = [r for r in report["runs"] if r["dataset"] == dataset_name and r["model"] == model_name]
            summary = {"dataset": dataset_name, "model": model_name, "n_runs": len(group)}
            for key in ["accuracy", "macro_f1", "total_training_seconds", "final_fit_seconds",
                        "tuning_seconds", "predict_us_per_sample", "node_count", "serialized_pickle_bytes"]:
                values = [r[key] for r in group]
                summary[key + "_mean"] = statistics.mean(values)
                summary[key + "_sd"] = statistics.stdev(values)
            report["summary"].append(summary)
    report["total_wall_seconds"] = time.perf_counter() - total_start
    output = output_dir / "tabular_results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("RESULT " + str(output), flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)


def self_check():
    # Meaningful invariants: labels at pure leaves, no splitting constant data,
    # generalization of a threshold, deterministic tie handling, input rejection.
    pure = CostTree().fit([[0.0], [1.0]], [7, 7])
    assert pure.predict([[-100.0], [100.0]]) == [7, 7]
    constant = CostTree().fit([[1.0]] * 6, [0, 0, 0, 1, 1, 1])
    assert constant.node_count == 1 and constant.predict([[1.0]]) == [0]
    threshold = CostTree(penalty=0.25, min_leaf=1).fit(
        [[float(i)] for i in range(20)], [0] * 10 + [1] * 10)
    assert threshold.predict([[2.5], [14.5]]) == [0, 1]
    assert threshold.node_count == 3
    try:
        CostTree().fit([[float("nan")]], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("NaN should be rejected")


if __name__ == "__main__":
    self_check()
    run(Path(__file__).resolve().parent / "results")
