# Declarative model derivation (wgt 0.3)

A derived model describes a final architecture and where its weights come from.
It does not execute a sequence of mutation commands. Undeclared layers and their
weights inherit the baseline. Same-name layer declarations replace definitions;
new definitions require explicit weight sources even when their shapes match.

## Expand a classifier from 10 to 12 classes

```wgt
wgt 0.3
model expanded extends "weights/base" sha256="<64 lowercase hex digits>":
  output logits: f32[12]
  layer head: linear(in=768, out=12)
  weights head:
    default: init(seed=42)
    weight[0:10, :] = base.head.weight
    bias[0:10] = base.head.bias
```

`sha256` pins **the exact bytes of the baseline's manifest.json**, not just the
weights file. That manifest pins the weights and architecture; loading also
verifies its source and tensor contracts. Obtain it with
`sha256sum weights/base/manifest.json`. The baseline path is relative to the
`.wgt` source. The placeholder above must be replaced with the real digest.

Run the small, executable example from the repository root:

```sh
python examples/derivation/demo.py
```

It creates `weights/derivation_demo/`, retains both artifacts and their sources,
and refuses an existing directory. This example has randomly initialized
weights; it demonstrates exact migration, not a trained classifier.

Check/build derived sources with the ordinary commands:

```sh
weightc check expanded.wgt
weightc build expanded.wgt --dry-run
weightc build expanded.wgt --no-store -o weights
# Or, in a wgt project:
wgt build expanded.wgt
```

`--dry-run` computes the derived state in memory and reports its architecture,
weight sources and changed-element counts without training or publishing files.
Omit `--no-store` to use normal automatic version recording. A derived artifact
is an ordinary runnable artifact and can itself be the next derivation's base.
Loading/inference does not require the ancestor to remain available; rebuilding
requires the pinned ancestor and the source's original relative path layout.

## Architecture declarations

- Same-name `layer head: ...` replaces a baseline definition in its existing
  position. Dotted targets such as `layer encoder.ff: ...` replace nested layers.
- `remove layer activation` explicitly removes a baseline layer. Omitting a
  layer never deletes it.
- `layer adapter after trunk: ...` or `layer adapter before head: ...` inserts
  a new top-level layer relative to a retained baseline layer. New names must
  be unique. Two insertions at the same side of the same anchor are rejected.
- Parent/child replacements or removals that overlap are rejected.
- Input/output contracts inherit unless explicitly declared; all connections
  and shapes are checked again. Blocks, constants, libraries, datasets, freezing
  and training declarations use the existing syntax inside the model body.

Architecture positions come from the baseline and explicit anchors, not the
order in which the change declarations appear. Ordinary sequential layer order
inside a newly declared `block` still defines that block's computation. This is
not a new arbitrary graph-patching language.

## Weight declarations

```wgt
  weights head:
    bias[3] = base.head.bias[3] + 0.1
    weight[0:2, :] = base.head.weight[0:2, :] * 0.9
```

Every right-hand reference reads the immutable baseline. Assignments cannot read
other assignments, so reordering disjoint assignments preserves the weight
result. Each weights group names an existing parameter-bearing layer/subtree.
The left-hand side names a tensor relative to that group; right-hand `base.*`
references use full baseline tensor names. `weights encoder:` can address
`ff.weight`, for example.

Supported selections are non-negative indices and half-open `start:stop` slices,
including `:`. Specify exactly one selector per dimension, or omit brackets to
select the entire tensor. Negative indices, strides and empty/out-of-range
slices fail. Tensor-to-tensor copies require identical selected shapes: no
implicit broadcasting, transposition, truncation or padding. A numeric literal
explicitly fills its selected region. Baseline references may optionally apply
one scalar `+`, `-`, `*` or `/` operation. Values must remain finite float32.

A changed/new parameter-bearing layer must have complete weight coverage:

```wgt
  weights head:
    default: base.head       # explicit same-shape reuse
```

or `default: init(seed=N)`, or disjoint assignments covering every element.
Defaults supply values only where assignments do not; placing `default` last
has the same meaning as placing it first. Seeds are non-negative integers below
2**64. Initializers use the existing module defaults in a separate seeded RNG
stream for each group. No initializer silently fills an unspecified remainder.
Overlapping assignments and overlapping group defaults are errors.

Unchanged layers automatically retain baseline weights at unspecified positions.
Explicit `import`/`inherit` cannot be combined with `extends`. The explicit
baseline disables automatic HEAD weight reuse; reuse mappings, overlap policy
and strict automatic-reuse mode are incompatible. Any subsequent training
executes after derivation and respects normal freeze rules.

## Provenance and reproducibility

The compiler records `manifest.derivation`: pinned ancestor identity, replaced,
removed and inserted layers, weight rules, per-tensor source/action, changed
scalar counts where shapes are comparable, and dropped tensor names. These
counts describe the derived state **before subsequent training**, as identified
by `stage: "before_training"`; final training changes require a separate diff. It applies
slices in bulk; it does not expand a large slice into millions of JSON edits.

Inherited tensors retain upstream provenance. Initialized tensors remain
`random-init`. Explicitly assigned tensors are `edited` unless subsequently
trained. The ancestor's training scores are not claimed for the child; evaluation
must be run again. Equality of migrated weights does not guarantee equality of
behavior after arbitrary architecture changes.

The baseline is checked at parsing and again at building. Even `--force` cannot
overwrite the baseline directory or place a child inside it. Normal artifact
overwrite protections and weight version storage remain in effect. Rebuilding
the same declarations against the same baseline/runtime produces identical
weight bytes. Different source order changes the source hash even when weights
are identical; the model name is also part of the saved weight-file metadata.

First iteration limits: local WeightLang artifacts, fixed shapes, no dynamic
selection expressions or arbitrary executable weight expressions. The result is
materialized as a complete artifact; this is not yet a standalone portable,
compressed patch archive. The parent identity plus declarative source describe
the transformation, and the version store can deduplicate retained data.
