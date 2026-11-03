"""Run with this checkout installed: python examples/weight_iterations/demo.py."""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from weightlang.artifact import load_artifact
from weightlang.compiler import compile_file
from weightlang.revisions import RevisionStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("weights/weight_iterations"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    store = RevisionStore(args.output / "store")
    skeleton = (
        "wgt 0.3\nmodel {name}\ninput x: f32[2]\noutput y: f32[1]\n"
        "layer body: linear(in=2, out=512)\nlayer act: tanh\n"
        "layer head: linear(in=512, out=1)\ninit seed=42\n{rules}"
        "data task:\n  [0, 0] -> [0]\n  [0, 1] -> [1]\n  [1, 0] -> [1]\n  [1, 1] -> [2]\n"
        "train data=task loss=mse optimizer=adamw lr=0.001 epochs=20 weight_decay=0.01\n"
    )
    with tempfile.TemporaryDirectory(prefix=".iteration-", dir=args.output) as tmp:
        scratch = Path(tmp)
        base_source = scratch / "base.wgt"
        base_source.write_text(skeleton.format(name="base", rules=""))
        base = compile_file(base_source, scratch / "weights")
        parent = store.save(base.path)
        rules = (
            f'inherit from "{base.path.resolve()}" sha256={base.manifest["weights"]["sha256"]}\n'
            "freeze all\nunfreeze head.bias\n"
        )
        next_source = scratch / "next.wgt"
        next_source.write_text(skeleton.format(name="next", rules=rules))
        child = compile_file(next_source, scratch / "weights")
        for name, tensor in base.module.state_dict().items():
            if name != "head.bias":
                assert torch.equal(tensor, child.module.state_dict()[name]), name
        revision = store.save(child.path, parent=parent["revision"])
        exported = scratch / "exported"
        store.checkout(revision["revision"], exported)
        for name in ("manifest.json", "weights.safetensors", "source.wgt"):
            assert (exported / name).read_bytes() == (child.path / name).read_bytes()
        loaded = load_artifact(exported, warn_untrained=False)
        assert torch.equal(loaded.predict([[0, 1]]), child.module(torch.tensor([[0.0, 1.0]])))
        report = {
            "parent": parent,
            "child": revision,
            "frozen_bitwise_equal": True,
            "export_byte_identical": True,
            "trainable": child.manifest["training_policy"]["trainable"],
            "note": "Synthetic training example; payload byte counts, not a large-model quality benchmark.",
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
