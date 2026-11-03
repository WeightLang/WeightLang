# Weight iterations and selective training

Implemented in this checkout; not a claim about the published 0.2 wheel.
The normative storage contract is [revision v1](specs/revision-v1.md).

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
weightc revision save weights/base --store weights/history
# Copy the returned revision ID into PARENT.
weightc build next.wgt --store weights/history --parent "$PARENT"
# Copy the new revision ID into CHILD.
weightc revision inspect "$CHILD" --store weights/history
weightc revision verify "$CHILD" --store weights/history
weightc revision checkout "$CHILD" --store weights/history -o weights/next
weightc run weights/next --input '[[1, 2]]'
```

`parent` records ancestry; it does **not** initialize the model. Use `inherit`
or `import` in the source to reuse weights. To inherit a stored revision, first
checkout the selected revision and use that artifact's weights hash. A revision
ID hashes the revision record; a weights hash hashes the safetensors file. These
are deliberately different identities.

`build --store` writes a temporary full artifact **inside the disk-backed store**,
saves the revision, and removes that temporary directory on success or failure.
It cannot be combined with `--force` or a custom `-o`. Saving an existing artifact
never deletes it. Both paths still need memory for the model and temporary disk
space for a full artifact; this feature saves retained iteration storage, not
peak training memory or scratch space. `new_bytes` counts newly published object
and revision payload bytes, not filesystem allocation, peak space, or compression.

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
