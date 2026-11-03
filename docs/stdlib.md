# Standard library design (proposal)

Status: **design only, nothing here is implemented.** This document proposes
what the WeightLang standard library contains, how it is addressed and
versioned, what it promises, and the order in which to build it. The decisions
marked **Decision needed** should be settled before implementation starts.

## Where WeightLang stands (0.1)

- Built-in layers: `linear`, `relu`, `tanh`, `sigmoid`; sequential networks only.
- Losses `mse`, `bce`; optimisers `sgd`, `adam`; full-batch training on inline data.
- `import` reuses **weights** of a `linear` layer; there is no way to reuse
  **structure** (a group of layers) between programs.
- Packages: GitHub paths, `wgt.mod`/`wgt.sum`, release assets (`packages/gates`).

A useful standard library therefore cannot just be "some packages": most of what
people need first is missing from the compiler itself.

## Three tiers

WeightLang code has three kinds of reusable things, which differ in who
implements them, how big they are and how they change. The standard library is
split the same way.

| Tier | What | Implemented as | Ships with | Versioned by |
|---|---|---|---|---|
| **1. Built-ins** | layer ops, losses, optimisers, data loading, metrics | Python in `weightc`, mapped to PyTorch | `weightc` | language version (`wgt 0.x`) |
| **2. `std` blocks** | reusable structures: `mlp`, `classifier`, `residual`, … | `.wgt` source files, no weights | `weightc` (bundled in the wheel) | language version |
| **3. Standard weights** | small trained artifacts with documented data and metrics | packages with release assets | GitHub releases, fetched with `weightc get` | own semver tags + `wgt.sum` |

Rationale:

- Tiers 1 and 2 must match the compiler exactly (a block can only use ops the
  compiler has), are small, and need no network. Shipping them with `weightc`
  avoids a dependency fetch for basic programs, as Go does with its std.
- Tier 3 is large, changes on its own schedule and needs provenance review, so
  it stays an ordinary package. Bundling weights in the compiler would make
  every `weightc` upgrade a weights upgrade.

## Tier 1: built-ins

### Admission rules

A built-in op is added only if it:

1. maps onto a stable PyTorch module or function (no custom CUDA, no
   third-party library);
2. has all state either as parameters (saved, origin-tracked) or as declared
   buffers (saved, origin `statistics`), with nothing hidden in Python objects;
3. is deterministic on CPU for a fixed seed, or is clearly documented when not;
4. gets a static shape rule, so `weightc check` can reject mismatches before
   running anything;
5. ships with tests: shape rule, comparison against the PyTorch reference
   output, save → reload bit-equality, and gradient flow.

### Proposed first batch (wgt 0.2)

| Area | Additions | Notes |
|---|---|---|
| Activations | `gelu`, `silu`, `softmax(dim=-1)`, `identity` | parameter-free |
| Normalisation | `layernorm(dim=n)` | weight + bias, origin-tracked |
| Regularisation | `dropout(p=0.1)` | active only in training; seeded |
| Embedding | `embedding(vocab=n, dim=m)` | needs integer inputs, see dtypes |
| Losses | `cross_entropy` (class indices), `mse`, `bce` | `cross_entropy` takes logits, no final softmax |
| Optimisers | `sgd`, `adam`, `adamw(weight_decay=…)` | |
| Training | `batch=n`, `shuffle seed`, `epochs`, `validate=<dataset>` | mini-batches with seeded shuffling, fully reproducible |
| Data | `data <name>: file="x.csv" sha256=<hex> inputs=0..3 target=4` | relative path + mandatory hash; inline rows still allowed |
| Metrics | `accuracy`, `loss` on train and validation sets | the manifest separates train and validation numbers |
| Dtypes | `i64` inputs for class indices / tokens | `f32` stays the only parameter dtype |

Deliberately **not** in the first batch: `batchnorm` (running statistics make
train/eval behaviour subtle), convolutions and attention (they need
multi-dimensional shapes and a non-sequential graph, see Tier 2), GPU training
flags, learning-rate schedules.

### Format impact

Artifact `format_version` 2: `layers[].args` may contain floats; tensors may
carry origin `statistics` (buffers); `interface` may declare `i64` inputs.
Loaders for version 1 keep working on version 1 artifacts.

## Tier 2: `std` blocks

### Language addition: `block` and `use`

A block is a named, parameterised group of layers. It has no weights; it
expands at compile time into ordinary layers.

```wgt
# std/nn.wgt — bundled with weightc
wgt 0.3

block mlp(in, hidden, out):
  layer fc1: linear(in=in, out=hidden)
  layer act: gelu
  layer fc2: linear(in=hidden, out=out)
```

```wgt
wgt 0.3
use std/nn
model classifier

input  x: f32[4]
output logits: f32[3]

layer body: nn.mlp(in=4, hidden=32, out=3)

init  seed=1
train data=iris loss=cross_entropy optimizer=adam lr=0.01 epochs=200 batch=16
```

Semantics:

- **Expansion, not runtime calls.** `body` becomes layers `body.fc1`,
  `body.act`, `body.fc2`. Tensors are named `body.fc1.weight` etc. The manifest
  stores the expanded layer list plus a `block` field (`std/nn.mlp@wgt0.3`)
  for provenance, so the loader still only needs built-in ops.
- **Parameters are integers only**, used in layer arguments. No conditionals,
  loops or recursion inside blocks; blocks may use other blocks (no cycles).
- **Shapes are checked after expansion**, with errors pointing at both the
  use site and the line inside the block.
- **Weight import uses dotted names**: `import body.fc1 from …`, and
  `import body.* from …` to import every parameter layer of a block whose
  expanded structure matches.

`use std/<name>` resolves to files bundled with `weightc`. `use
github.com/<owner>/<repo>/<path>` resolves through `wgt.mod`/`wgt.sum` like
other packages; third-party blocks ship as source only.

### Non-sequential structure

`residual` cannot be written as a sequential list. Proposal: a small set of
structural built-ins taking a block, rather than a general graph language:

```wgt
layer res1: residual(nn.mlp(in=64, hidden=256, out=64))   # x + f(x)
```

**Decision needed:** whether structural built-ins (`residual`, later
`parallel`/`concat`) are enough for the std roadmap, or whether WeightLang
should get named-wire graph syntax (`h = res1(x)`). The structural form keeps
programs easy to generate and check; a graph syntax is more general.

### Proposed initial `std` modules

| Module | Contents |
|---|---|
| `std/nn` | `mlp`, `mlp_deep(in, hidden, out, depth)` only if loops are added; otherwise fixed `mlp2`, `mlp3` |
| `std/heads` | `classifier(in, classes)` (linear → logits), `regressor(in, out)` |
| `std/blocks` | `residual_mlp(dim, hidden)`, `prenorm_mlp(dim, hidden)` (layernorm → mlp → residual) |

Every std block gets an expansion snapshot test (exact layer names, ops and
shapes) and an end-to-end training test on a tiny dataset.

### Stability

- Before `wgt 1.0`: no promise; std blocks change with the language version.
- From `wgt 1.0`: within a language version, a std block's expansion (layer
  names, ops, shapes, tensor names) never changes, so weights trained against
  it stay importable. A different structure gets a new block name.

## Tier 3: standard weights

### Admission rules

A standard weight artifact must have:

1. training data that is redistributable, hashed in the manifest and either
   bundled or fetchable by hash;
2. a recorded, rerunnable build (`source.wgt`, seed, PyTorch version) — a
   rebuild on the same PyTorch version and CPU must give identical bytes;
3. metrics on **held-out** data, not just training rows, with a written
   statement of what the model is and is not useful for;
4. a license that allows redistribution of the weights, with any attribution
   requirements (e.g. CC BY) carried in the manifest;
5. a size limit (initially 50 MB per artifact) — larger models belong in
   separate packages, not the standard set.

Weights that were imported or converted from elsewhere (e.g. existing PyTorch
checkpoints) are welcome as ordinary packages, but only enter the standard set
if their origin and license can be documented.

### Location

**Decision needed:** where the standard weights live:

- **A.** In this repository under `weights/<name>` with tags
  `weights/<name>/vX.Y.Z` (the current `packages/gates` pattern); simple and
  reviewed together with the compiler.
- **B.** A separate repository `github.com/WeightLang/std-weights`, so weight
  releases do not clutter compiler releases and can have different
  maintainers.

Recommendation: start with **A** for the first few artifacts and move to **B**
once there are more than a handful; module paths change with the move, so
move before announcing any stability.

### Initial candidates

Small, honest, useful as examples and as components:

- `gates`: the existing XOR package, extended with all two-input gates and
  truth-table spec tests;
- `iris_classifier`: classic tabular classification with a held-out split
  (UCI Iris, CC BY 4.0 — attribution required; license to be rechecked before inclusion);
- `digits_mlp`: an MLP on 8×8 handwritten digits (UCI Optical Recognition of
  Handwritten Digits, CC BY 4.0 — to be rechecked before inclusion), reporting test accuracy; usable as a feature extractor for reuse
  experiments where reuse can actually be measured.

## Roadmap

| Milestone | Scope | Exit criteria |
|---|---|---|
| **S1 — wgt 0.2 built-ins** | Tier 1 first batch, external hashed data, mini-batches, validation metrics, artifact format v2 | all admission tests; `iris` example reaches a documented held-out accuracy |
| **S2 — wgt 0.3 blocks** | `block`, `use`, dotted imports, `residual`; `std/nn`, `std/heads`, `std/blocks` | expansion snapshots; a reuse example that freezes a std block's weights |
| **S3 — standard weights** | admission checklist as a local command (`weightc audit`), first 2–3 artifacts | each artifact passes the checklist and rebuilds byte-identically on the reference PyTorch |

Each milestone is released like 0.1: local tests, GitHub release, PyPI through
`publish-pypi`.

## Decisions needed

1. **Three-tier split**, with built-ins and std blocks shipped inside `weightc`
   and standard weights as packages — agree?
2. **Structure:** structural built-ins (`residual(…)`) vs a named-wire graph syntax.
3. **Standard weights location:** this repository (A) or `WeightLang/std-weights` (B).
4. **S1 scope:** is the first batch above the right size, or should
   `embedding`/`i64` wait so S1 stays purely `f32`?
