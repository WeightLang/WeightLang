"""Explicit initialization, selected-layer mixing and architecture migration.

Run from the repository root with the SDK installed and wgt on PATH:
    python python/examples/weight_lifecycle/build.py
Every output uses a new snapshot directory; repeated writes are rejected.
"""

from pathlib import Path

import torch
import weightlang as wgt


HERE = Path(__file__).resolve().parent
OUTPUT = Path("weights/sdk-weight-lifecycle")


def scaled_identity(shape):
    """A deterministic custom initializer; no global or implicit random stream."""
    return torch.eye(shape[0], shape[1], dtype=torch.float32) * 2


def main():
    definition = wgt.compile(HERE / "model.wgt")
    original = (wgt.Plan(definition)
                .set("first", scaled_identity)
                .set("second", lambda shape: torch.eye(*shape))
                .set("bias", lambda shape: torch.zeros(shape))
                .build())
    original.save(OUTPUT / "original")

    # A second version can be produced by training; these explicit edits make
    # the example reproducible without running a training loop.
    alternate = (wgt.Plan(definition, base=original)
                 .replace("first", original["first"] + .5)
                 .replace("second", original["second"] * 3)
                 .build())
    alternate.save(OUTPUT / "alternate")
    plan = wgt.Plan(definition, base=original)
    for name in ("first", "second"):
        plan.replace(name, .75 * original[name] + .25 * alternate[name],
                     sources=(original.ref(name), alternate.ref(name)))
    print(plan.preview())
    mixed = plan.build()
    mixed.save(OUTPUT / "mixed")
    pinned = wgt.WeightSet.load(OUTPUT / "mixed", revision=mixed.revision)

    wider = wgt.compile(HERE / "wider/model.wgt")
    migrated = (wgt.Plan(wider)
                .set("first", torch.cat((pinned["first"], torch.zeros(1, 2)), dim=0),
                     sources=[pinned.ref("first")])
                .copy("second", pinned.ref("second"))
                .copy("offset", pinned.ref("bias"))
                .set("gain", lambda shape: torch.ones(shape))
                .build())
    migrated.save(OUTPUT / "migrated")
    model = wgt.restore(wider, migrated, device="cpu").eval()
    print("revision:", migrated.revision)
    print("prediction:", model(torch.tensor([1., 2., 0.])))


if __name__ == "__main__":
    main()
