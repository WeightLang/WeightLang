# Reusing weights after architecture changes

`weightc build` now uses the HEAD captured at build start as its weight baseline.
No parent, store path or exported baseline is needed. First builds initialize
normally. This describes the source implementation, not an existing published wheel.

```sh
weightc build model.wgt -m baseline
# Edit model.wgt: add/remove/resize/rename layers.
weightc reuse-plan model.wgt
weightc build model.wgt -m "new architecture"

# Explicitly permit copying leading rows/columns when resizing supported layers.
weightc reuse-plan model.wgt --reuse overlap
weightc build model.wgt --reuse overlap --reuse-report weights/reuse-result.json

# Rename body -> encoder without losing its weights.
weightc build model.wgt --reuse-map encoder=body
# Reject any automatic tensor that cannot be reused exactly.
weightc build model.wgt --reuse-strict
# Deliberately start fresh.
weightc build model.wgt --reuse none
```

## Selection and safety

| Situation | Default `exact` | Opt-in `overlap` |
| --- | --- | --- |
| Same name, operator, configuration, dtype and shape | Copy entire tensor | Same |
| New layer or missing source tensor | Keep new initialization, report reason | Same |
| Removed layer | List unused source tensors | Same |
| Renamed layer | Supply `--reuse-map TARGET=SOURCE` | Same |
| Resized linear/embedding/position_embedding | Keep new initialization | Copy leading intersection of axes; retain initializer outside it |
| Changed attention configuration, including head count | Reject reuse even for equal shapes | Same; no packed-attention cropping |
| Changed other operator or unsupported shape | Keep new initialization, report reason | Same |

Linear bias can remain compatible when only the number of input features changes:
its coordinates are output features. Mappings accept tensors or layer prefixes;
the longest matching prefix wins. Bias cannot be reinterpreted as a weight.
Explicit `.wgt import` and `inherit` take precedence, and conflicting mappings
are errors. `inherit` remains a strict architecture contract: use automatic reuse
for architecture migration rather than loosening that contract.

Overlap is an explicit assumption that leading coordinates retain their meaning.
Token IDs, class IDs and feature order must remain stable. The tool cannot detect
semantic changes or permute vocabularies automatically. Renaming and preserving
shapes likewise do not prove functional equivalence. Shrinking drops source tails;
`source_unused_values` records this. Convolutions, normalization and packed attention
are only reused exactly; their resizing is not supported by overlap v1.

`--reuse-from REF` (alias `--from`) selects an older revision/branch as the baseline.
The new version's parent is still the captured HEAD. `--reuse-artifact DIR` instead
uses an exported artifact. `--no-store` has no automatic HEAD baseline; combine it
with `--reuse-artifact` if reuse is wanted. Sources are integrity-checked before
copying. A corrupted or unresolved baseline is a build error, never a silent fresh start.

## Visible results and failure

`reuse-plan` prints JSON without training, writing model weights or advancing HEAD.
It builds target shapes on the meta device, but loads the baseline for validation.
`build` includes the report in its output and in `manifest.json` under `reuse`.
`weightc inspect ARTIFACT` shows it. `--reuse-report NEW_FILE` saves a separate JSON
report, including strict/baseline failures; existing files are never overwritten.

The [reuse v1 JSON Schema](specs/reuse-v1.schema.json) describes the report:

- `status`: `complete`, `partial`, `no_reuse`, `no_baseline`, `disabled`,
  `explicit_only`, or `failed`.
- `execution`: `planned` for previews/rejected plans, `applied` after copying.
  Preview `copied_values` are planned counts, not evidence of executed copies.
- `summary`: scalar counts copied, freshly initialized and delegated to explicit
  imports, plus copied fraction and tensor counts. Counts include tensor buffers.
- `tensors`: target/source names, shapes, status, reason, counts, source provenance
  and (for overlap) copied extents and unused source values.
- `source`: revision and hashes identify the baseline even when it differs from
  the version's history parent. Architecture and interface changes are flagged.
- `quality: not_evaluated`: transfer is an initialization operation, not an
  accuracy claim. Evaluate the changed model on representative data separately.

Strict mode checks the whole plan before copying/training. Any automatically
initialized or partially compatible tensor fails the build, even if cropping
would fill every target value. It emits `strict_requires_exact_reuse`, exits
nonzero, creates no output model and leaves HEAD unchanged. Explicit imports are
validated by the existing import compiler. Baseline failures emit
`baseline_validation_failed`. Syntax/option/training failures remain ordinary
compiler errors; they are not claimed as transfer failures or saved reuse reports.

For a two-layer MLP widened from 4 to 6 hidden units, the regression fixture has
25 target scalars: default exact reuse preserves 1 output bias; overlap preserves
17 old scalars and initializes the remaining 8. This demonstrates copying, not a
trained model quality benchmark.

Copying happens before `freeze`/`unfreeze` and optimization. Freeze compatible
layers explicitly to keep them unchanged during training; reuse does not freeze
them automatically. Fully populated tensors preserve upstream trained/untrained
provenance. Partially populated tensors retain initialization provenance until
training; imported random values are never presented as trained knowledge.
Optimizer moments and training step state are not resumed.

## Storage and Python API

Reused unchanged chunks are naturally shared by the existing lossless object
store. Reshaping can shift byte boundaries and training can change every chunk;
reuse percentage does not equal disk savings. Use `revision stats` for actual
storage counts. The optional manifest report does not change artifact format v3.

```python
from weightlang.compiler import compile_file

result = compile_file(
    "next.wgt", "weights", reuse_source="weights/baseline",
    reuse_policy="overlap", reuse_mapping={"encoder": "body"},
)
print(result.manifest["reuse"])
```

Direct Python compilation keeps its existing fresh-init behavior unless reuse
options are supplied. It does not infer a repository implicitly.
