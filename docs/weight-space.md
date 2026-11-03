# Weight-space experiments

## Acknowledgments

Thank you to **Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab** for
[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/).
We explicitly cite their article as inspiration for WeightLang's weight
inspection and comparison workflow. Their research explores direct LLM weight
reading and writing without gradient optimization.

WeightLang is an independent project. Its current gates and Iris examples use
gradient training. We have not reproduced the article's Astra/Qwen experiments,
and their results are not WeightLang benchmarks.

## Inspect, compare, evaluate

These commands are **on main, not in the published weightc 0.2.0 wheel**.
Install this checkout with `pip install -e .`.

```sh
weightc build examples/xor.wgt -o build
weightc weights build/xor_mlp --limit 0 > xor.weights.json
weightc weights build/xor_mlp --layer hidden --limit 16
weightc diff build/baseline build/candidate
weightc evaluate build/candidate --data evaluation.json --loss bce --compare build/baseline
```

All three commands verify artifact hashes before reading weights. They output
versioned JSON and preserve the artifact and its training metadata.

- `weights` includes the input/output contract, expanded architecture, tensor
  shapes, provenance and flattened row-major float32 values. The default budget
  is **256 scalar values across all selected tensors**. Each tensor states how
  many values were omitted. `--limit 0` deliberately exports every value;
  `--layer body.fc1` selects that exact dotted subtree, not similarly named layers.
- `diff BEFORE AFTER` requires matching architecture and input/output contracts.
  It reports exact scalar inequality, changed percentage, per-tensor maximum
  absolute delta and L2 delta. It does not count changed tensors as changed
  parameters, and it does not infer whether an edit helps.
- `evaluate` runs inference in eval mode, with no training, on a fixed JSON file
  whose SHA-256 is recorded. `--compare` evaluates the baseline on those same
  bytes. Deltas are **candidate minus baseline**: lower loss and higher accuracy
  are better. Untrained origins remain visible even if a toy test scores well.

BCE and MSE data use target matrices:

```json
{
  "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
  "targets": [[0], [1], [1], [0]]
}
```

Cross-entropy data use a flat class-index array, e.g. `"targets": [0, 1, 2]`,
with one integer per input row. Sequence logits [N,L,V] use targets [N,L];
inputs and other targets use nested arrays matching their sample shapes. The model must emit logits. BCE requires binary
targets and probability outputs. Invalid shapes, non-finite values, out-of-range
labels and incompatible comparison models are errors.

Evaluating the four XOR truth-table rows demonstrates that finite truth table;
they are also its training data. Calling a file “evaluation” does not make it an
independent holdout set. Keep training, development and held-out tests separate
when assessing generalization.

## Direct edits (main, unreleased)

The patch-template and patch commands complete the read → edit → compare →
evaluate loop. Neither command trains the model or contacts an LLM.

~~~sh
weightc patch-template build/baseline --author "Your name" > patch.json
# Fill reason and edits in patch.json, retaining the two baseline hashes.
weightc patch build/baseline --patch patch.json -o build/candidate
weightc diff build/baseline build/candidate
weightc evaluate build/candidate --data evaluation.json --loss mse --compare build/baseline
wgt graph build/candidate --open
~~~

The template has format "weightlang.patch", format_version 1,
baseline_weights_sha256, baseline_manifest_sha256, author, reason, and
edits. All fields are required; unknown fields and duplicate JSON keys fail.
Fill the initially empty edits array, for example:

~~~json
[
  {"tensor": "fc.weight", "index": 0, "expected": 0.25, "value": 2.0},
  {"tensor": "fc.bias", "index": 0, "value": 1.0}
]
~~~

Indices are zero-based into each tensor's flattened row-major values.
The optional expected field checks the prior float32 value. Every edit sets a
single scalar; there are no executable expressions. The baseline hashes bind
both weights and metadata, rejecting stale patches even if only the baseline
manifest changed. Names, indices, duplicate positions, booleans, NaN/Infinity,
float32 overflow, and patches that change no float32 values are rejected.

Output must be a **new directory outside the baseline**. The baseline is
unchanged; failed writes remove the incomplete candidate. Applying the same
patch to the same baseline under the same runtime produces byte-identical
candidate files.

Changed tensors receive origin "edited"; untouched tensors retain provenance.
The v3 manifest embeds the complete patch, its hash, actual before/after
values, prior tensor metadata, and prior training metrics in edit_history.
Current training is cleared: baseline loss/accuracy does not describe the
edited candidate. Evaluate the candidate again. Two-hop imports preserve edited
provenance, and audit/package checks treat directly edited values as lacking
training provenance, even when imported from a previously trained model.
The standard trained-package admission rules therefore still reject them by
default. This does not imply the edit is useless: report behavior independently.

The original source.wgt remains the architecture/baseline source, not a complete
recipe for the edited weights. Rebuilding it alone will not replay edits.
Reproduction requires the exact baseline artifact and recorded patches.
Authorship and history are **self-reported, not signed or authenticated**.

Run the [affine demo](../examples/weight_space/README.md) for a complete example:
it sets a weight to 2 and a bias to 1, implementing y = 2x + 1 without training,
then measures baseline/candidate MSE. This proves the direct-edit workflow,
not general model-editing ability or LLM-generated weight programming.

Transformer tensor names are also editable, for example
body.attention.attention.qkv.weight in the bundled tiny Transformer.
Use weightc weights with --layer to inspect the actual names and shapes.
Changing a Transformer parameter is mechanically supported; useful behavioral
editing remains an experimental task requiring target and regression tests.

There is no automatic LLM optimizer, rollback-to-history command, or signed edit
history yet. Evaluation is read-only and does not automatically grant admission.
