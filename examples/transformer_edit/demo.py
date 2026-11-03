"""Deterministic synthetic copy-task experiment; explicit output-row edits, no LLM."""

import argparse
import itertools
import json
import random
from pathlib import Path

from weightlang.artifact import load_artifact
from weightlang.compiler import compile_file
from weightlang.experiments import run_experiment
from weightlang.patches import apply_patch, patch_template


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=Path("weights/transformer-edit"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"{root} already exists; choose a new output directory")
    root.mkdir(parents=True)
    target = list(itertools.product((0, 1), repeat=4))
    regression = list(itertools.product((2, 3), repeat=4))
    held_out = set(target + regression)
    mixed = [x for x in itertools.product(range(4), repeat=4) if x not in held_out]
    random.Random(42).shuffle(mixed)
    train, development = mixed[:128], mixed[128:160]
    assert not (set(train) & set(development) or set(train + development) & held_out)
    source = """wgt 0.3
use std/transformer
use std/heads
model token_copy
input tokens: i64[4]
output logits: f32[4, 4]
layer embed: transformer.tokens(vocab=4, dim=8, max_len=4)
layer body: transformer.decoder(dim=8, heads=2, hidden=16)
layer norm: layernorm(dim=8)
layer head: heads.classifier(in=8, classes=4)
init seed=42
"""
    for name, rows in (("train", train), ("development", development)):
        source += "data " + name + ":\n"
        source += "".join(f"  {list(row)} -> {list(row)}\n" for row in rows)
    source += "train data=train validate=development loss=cross_entropy optimizer=adamw lr=0.02 epochs=80 batch=16\n"
    file = root / "copy.wgt"
    file.write_text(source)
    result = compile_file(file, root / "baseline")
    baseline = result.path
    artifact = load_artifact(baseline, warn_untrained=False)
    state = artifact.module.state_dict()
    dump(root / "target.json", {"inputs": target, "targets": [[1 - v for v in row] for row in target]})
    dump(root / "regression.json", {"inputs": regression, "targets": regression})
    dump(
        root / "splits.json",
        {
            "seed": 42,
            "train": train,
            "development": development,
            "target": target,
            "regression": regression,
            "note": "Held-out sequences, same synthetic task/vocabulary; no real-language claim.",
        },
    )
    suite = {
        "format": "weightlang.experiment",
        "format_version": 1,
        "cases": [
            {
                "name": "swap token labels 0 and 1",
                "role": "target",
                "data": "target.json",
                "loss": "cross_entropy",
                "rules": {"min_accuracy": 1, "min_accuracy_gain": 0.9},
            },
            {
                "name": "retain token labels 2 and 3",
                "role": "regression",
                "data": "regression.json",
                "loss": "cross_entropy",
                "rules": {"min_accuracy": 1, "max_loss_increase": 0.000001},
            },
        ],
    }
    dump(root / "suite.json", suite)
    reports = {}
    for name, pair in (("candidate", (0, 1)), ("rejected", (0, 2))):
        patch = patch_template(baseline, "WeightLang synthetic edit experiment")
        patch["reason"] = f"Explicitly swap output-label rows {pair}; no training during edit"
        for tensor_name in ("head.logits.weight", "head.logits.bias"):
            tensor = state[tensor_name]
            width = tensor.numel() // tensor.shape[0]
            for dest, source_row in (pair, pair[::-1]):
                for column in range(width):
                    index, source_index = dest * width + column, source_row * width + column
                    patch["edits"].append(
                        {
                            "tensor": tensor_name,
                            "index": index,
                            "expected": tensor.flatten()[index].item(),
                            "value": tensor.flatten()[source_index].item(),
                        }
                    )
        patch_file = root / (name + ".patch.json")
        dump(patch_file, patch)
        apply_patch(baseline, patch_file, root / name)
        report = run_experiment(root / name, baseline, root / "suite.json")
        dump(root / (name + ".report.json"), report)
        reports[name] = {
            "passed": report["passed"],
            "changed_percent": report["changes"]["changed_percent"],
            "cases": [
                {"name": c["name"], "accuracy": c["candidate"].get("accuracy"), "passed": c["passed"]}
                for c in report["cases"]
            ],
        }
    assert reports["candidate"]["passed"] and not reports["rejected"]["passed"], reports
    print(json.dumps(reports, indent=2))
    print(f"wgt graph {root / 'candidate'} --compare {baseline} --open")


if __name__ == "__main__":
    main()
