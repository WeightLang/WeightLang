# Python interoperability

WeightLang's compiler and runtime use PyTorch. The public Python entry points are
`compile_file`, `compile_program`, `parse`, `parse_file`, `load_artifact`,
`save_artifact`, `Artifact`, `BuildResult`, `Trainer`, and `TrainingContext`.

## Build and load

```python
from weightlang import compile_file, load_artifact

result = compile_file("examples/xor.wgt", "weights")
artifact = load_artifact(result.path)
prediction = artifact.predict([[0, 1], [1, 1]])
```

`compile_file(source, out_dir="weights", *, force=False, trainers=None)`
writes `out_dir/<model>/`. `compile_program`
accepts a program returned by `parse_file` and an output root. Existing builds
are protected unless explicitly replaced with `force=True`.

`load_artifact(path, *, warn_untrained=True)`
verifies hashes, shapes and dtypes, restores declared frozen parameters, and
returns an artifact whose `.module` is an ordinary PyTorch module in evaluation
mode. `.predict()` checks input shape and runs without gradients. For training,
use `.module`, call `.train()`, and then construct your optimizer.

## Train in Python and save

```python
import torch
from weightlang import load_artifact

artifact = load_artifact("weights/xor_mlp")
model = artifact.module
model.train()
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=0.001
)
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])
for _ in range(20):
    optimizer.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

updated = artifact.save(
    "weights/xor_finetuned",
    training={"method": "Adam", "lr": 0.001, "steps": 20, "dataset": "xor"},
)
```

`save_artifact(artifact, path, *, training=None)` is equivalent to `artifact.save`.
Unlike the compiler's output root, **path is the exact new artifact directory**.
It never overwrites an existing directory, including the baseline, and refuses
output inside the baseline. Keep each experiment in a separate directory.
The returned artifact is verified, loaded on CPU, and in evaluation mode;
saving does not change the original module's mode or device.

Saving checks that the declared architecture, tensor names, shapes and dtypes
remain unchanged. Non-finite weights, changes to frozen parameters, custom
Python forward implementations and forward hooks are rejected. This API saves
weights, not arbitrary Python code, optimizer state or an exact training-resume
checkpoint. Concurrent model mutation during saving is unsupported.

Without `training`, changed tensors become `edited`, not `trained`. With a JSON
training record containing a nonempty `method`, changed parameters become
`trained` (or `imported+trained` for imported parameters). Unchanged tensors
retain their origins; buffers changed outside the parameter set remain `edited`.
Training records are **caller-reported**, not proof of model quality or that a
particular algorithm ran. Include dataset/code identifiers, seeds and metrics
when available. Records cannot contain NaN, infinity or non-JSON objects.

The new manifest records baseline hashes, changed tensors, previous origins,
previous training metadata and the Python/PyTorch versions in `python_history`.
Source and weight hashes are regenerated. The original `.wgt` remains the base
model description: rebuilding it alone does not reproduce external Python
updates. Retain the training code and data separately. The saved artifact uses
format version 3 and still loads with the ordinary WeightLang loader.

## Supply a training algorithm during compilation

```python
from weightlang import compile_file, TrainingContext

def fit(context: TrainingContext):
    # Train context.module using context.inputs / context.targets.
    # context.parameters contains only parameters allowed by the freeze policy.
    # Return a dictionary of finite numeric metrics, or None.
    ...

result = compile_file("model.wgt", "weights", trainers={"custom": fit})
```

The source must select `trainer=custom` in its training directive. The callback
runs inside the existing compiler workflow, which checks its changes and records
training provenance. A complete executable example is in
[`examples/custom_trainer`](../examples/custom_trainer/).

## Export standalone Python source

`weightc export-python model.wgt -o weights/python_model/model.py` exports architecture only.
Pass a built artifact directory instead to copy its weights and generate `load_model()`.
The output runs without WeightLang; see [Python export](python-export.md) for dependencies,
examples, batch conventions and limitations. The Python API is `export_python(source, output)`.
