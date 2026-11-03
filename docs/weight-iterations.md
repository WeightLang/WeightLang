# Weight iterations and selective training

Implemented in this checkout; not a claim about the published 0.2 wheel.
See [weight version control](weight-version-control.md) for automatic parents and branches.
The normative contracts are [object storage v1](specs/revision-v1.md) and
[repository references and revision v2](specs/repository-v1.md).

## Freeze independently of imports (wgt 0.3)

```wgt
freeze all
unfreeze head
freeze head.weight
```

The final rule set above trains only `head.bias`. Selectors match an exact
parameter name or a dotted module subtree. `all` is reserved for every parameter.
Rules run in source order **after all imports**, irrespective of where imports
appear in the file. The last matching rule wins. Unknown/empty selectors,
no training statement, and a final empty trainable set are errors.
Existing `import ... freeze` establishes defaults that explicit rules can override.

The compiler sets `requires_grad` and passes only trainable parameters to the
optimizer, including AdamW. It does not detach the frozen forward computation:
gradients may still pass through a frozen layer to earlier trainable layers.
Freezing controls parameters, not module eval/train mode (e.g. dropout).
The artifact records the resolved names in `training_policy`, format version 1;
this is a record of the build, not an instruction to resume optimization when
loading an inference artifact. Frozen random initialization remains `random-init`.
Slices, wildcard selectors, LoRA, reduced precision, optimizer-state checkpoints
and exact training resumption are not implemented.

## Inherit an entire model (wgt 0.3)

```wgt
inherit from "../weights/base" sha256=<64-character weights SHA256>
freeze all
unfreeze head
train data=task loss=mse optimizer=adamw lr=0.001 epochs=20
```

Replace the placeholder with the hash reported by `weightc inspect --json`.
Declare the architecture, interfaces, data and `init` as usual. Inherit requires
the same expanded architecture and input/output shapes/dtypes; names of ports
may differ. It imports every parameter-bearing top-level layer, retains origins,
and requires a weights hash pin. The path is a local artifact directory relative
to the source file (absolute paths also work). Do not combine inherit and import;
use explicit imports for partial or remote package reuse.

## Durable version store

```sh
weightc revision commit weights/base -m "baseline"
weightc build next.wgt -m "next iteration"
weightc revision log
weightc revision switch -c experiment
weightc revision inspect HEAD
weightc revision verify HEAD
weightc revision checkout HEAD~1 -o weights/previous
weightc run weights/previous --input '[[1, 2]]'
```

The default repository is `weights/.revisions`. `--store PATH` selects another.
The first commit creates `main`; subsequent commits automatically parent to HEAD
and advance the selected branch. There is no need to copy IDs. Identical artifacts
are no-ops. `save` is an alias for `commit`. Branch switching selects where future
commits go; it does not replace existing artifact directories. Use checkout to
materialize a branch or revision in a new directory.

The automatically recorded parent represents ancestry; it does **not** initialize the model. Use `inherit`
or `import` in the source to reuse weights. To inherit a stored revision, first
checkout the selected revision and use that artifact's weights hash. A revision
ID hashes the revision record; a weights hash hashes the safetensors file. These
are deliberately different identities.

Default `build` writes a temporary full artifact **inside the disk-backed store**,
saves the revision, and removes that temporary directory on success or failure.
Use `build --no-store` for standalone output with `-o` and explicit replacement
via `--force`; these options are not available for versioned builds. Saving an existing artifact
never deletes it. Both paths still need memory for the model and temporary disk
space for a full artifact; this feature saves retained iteration storage, not
peak training memory or scratch space. `new_bytes` counts newly published object
and revision payload bytes, excluding mutable refs, filesystem allocation and peak space.
`revision stats` reports total stored payload including refs.

Checkout refuses any existing destination, reconstructs and verifies all three
original files, and loads the result with the normal artifact validator before
publishing it. `verify` also checks model structure/loadability using disposable
scratch inside the store. It does not evaluate task quality.

The version store is a separate format. Python and Rust artifact readers, graph,
package and publish commands consume the **checked-out standard artifact**;
there is no new artifact v4 and no silent partial support in old readers. Copy
the entire store to transfer history, or checkout a standalone artifact for
existing publication tools. Preserve the entire store; no automatic GC exists.

## Runnable example

```sh
python examples/weight_iterations/demo.py --output weights/weight_iterations
```

This trains a small synthetic regressor, changes only its final bias, checks all
frozen parameters bitwise, saves parent/child revisions and verifies exact file
restoration. JSON output records actual payload byte counts, not a large-model
benchmark. Retained store and report stay in the output directory; build scratch
is removed. The example refuses to overwrite an existing output directory.
