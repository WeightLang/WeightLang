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
with one integer per input row. The model must emit logits. BCE requires binary
targets and probability outputs. Invalid shapes, non-finite values, out-of-range
labels and incompatible comparison models are errors.

Evaluating the four XOR truth-table rows demonstrates that finite truth table;
they are also its training data. Calling a file “evaluation” does not make it an
independent holdout set. Keep training, development and held-out tests separate
when assessing generalization.

## Next boundary

These tools enable readable, measurable weight experiments. They do not yet
provide a direct-edit statement, automatic LLM optimizer, or signed edit history.
A future patch format must bind edits to the baseline hash and tensor indices,
record authorship and edit provenance separately from training, reject stale or
non-finite patches, and retain independent before/after evaluations.
