# Ablation experiments

Source builds provide `weightc ablate`: measure inference-time interventions
against an unchanged model on a fixed evaluation dataset. No training occurs.
Variants run independently; interventions within one variant combine.

```sh
weightc ablate weights/my_model --suite ablation.json
# Retain a report alongside experiment artifacts:
mkdir -p weights/my_experiment
weightc ablate weights/my_model --suite ablation.json > weights/my_experiment/ablation-report.json
```

For a model with a layer named `fc`, `ablation.json` can contain:

```json
{
  "format": "weightlang.ablation",
  "format_version": 1,
  "data": "evaluation.json",
  "loss": "mse",
  "variants": [
    {"name": "without-fc-output", "interventions": [{"op": "zero_output", "target": "fc"}]},
    {"name": "without-bias", "interventions": [{"op": "zero_tensor", "target": "fc.bias"}]},
    {"name": "bypass-fc", "interventions": [{"op": "skip", "target": "fc"}]}
  ]
}
```

The data path is relative to the suite file. Data has `inputs` and `targets`
arrays, using the same format as `weightc evaluate`. Supported losses:
`mse`, `bce` (probabilities), `cross_entropy` (raw logits).

| Operation | Effect |
| --- | --- |
| `zero_output` | Replace the named module's output with zeros of the same shape. |
| `zero_tensor` | Zero a complete tensor, such as `fc.weight` or `fc.bias`. |
| `skip` | Replace a module with identity: pass its input to the next module. |

Targets use exact dotted PyTorch module or tensor names, including nested
modules. `weightc weights MODEL --limit 1` shows tensor names. A skipped module
must leave valid dimensions and dtype for its consumers; incompatible variants
fail. Clearing weights is not equivalent to clearing layer output when bias,
normalization or activation remains. Clearing a residual branch's output differs
from clearing the entire residual block.

Each variant needs a unique name and at least one intervention. Duplicate or
ancestor/descendant targets within a variant are rejected to avoid hidden
interventions. Unknown targets and invalid outputs fail the command.

The JSON report records baseline weight and manifest hashes, suite and dataset
hashes, baseline metrics, each intervention and its metrics. Deltas are
**variant minus baseline**: positive loss delta or negative accuracy delta means
degradation. Classification reports include accuracy. Exit status 0 means the
experiment ran successfully, not that its variants improved the model.

Model files remain unchanged. Only one variant copy is retained in memory at a
time. These are inference-only results on supplied data, not retrained
architecture comparisons, confidence intervals or evidence of generalization.
The command does not export modified models or infer causal importance beyond
the declared interventions and dataset.
