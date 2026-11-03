# Custom training algorithms (wgt 0.3)

> 当前 `build` 只生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例改用 `weightc legacy-build`；参见[分发包与迁移规范](bundles.md)。


Training is replaceable. The default `backprop` trainer retains SGD/Adam/AdamW
and the existing full-batch/minibatch behavior. A custom trainer can update
weights without calling autograd or constructing an optimizer.

## Run the forward-only example

From the repository root, with weightc installed:

```sh
weightc check examples/custom_trainer/model.wgt
weightc legacy-build examples/custom_trainer/model.wgt --no-store \
  --trainer search=examples/custom_trainer/search.py:train
weightc run weights/forward_search --input '[[0], [1]]'
```

`wgt export` accepts the same repeatable `--trainer NAME=FILE.py:FUNCTION`
argument and forwards it to weightc. Paths are relative to the compiler's working
directory (the project directory for `wgt -C PROJECT build`). Normal overwrite
rules still apply: existing model directories require explicit `--force`.

The example uses random perturbations and accepts only candidates that improve
full-dataset loss. It is a small demonstration, not a scalable training algorithm.
It never calls `backward()`, `autograd.grad()` or a PyTorch optimizer.

```wgt
train data=samples loss=mse trainer=search lr=0.1 epochs=400
```

`trainer` is an identifier, not an import path. Parsing/checking does not load
Python. The CLI binding supplies trusted local Python code, which
runs with the build process's permissions; this is not a sandbox. Loading an
already-built artifact never needs the plugin.

## Python API

```python
import torch

from weightlang.compiler import compile_file
from weightlang.trainers import TrainingContext


def my_algorithm(context: TrainingContext):
    with torch.no_grad():
        # Implement local learning, search, a closed-form solve, etc.
        # Update context.parameters in place.
        ...
    return {"updates": 10}  # optional, finite numeric metrics


result = compile_file("model.wgt", "weights", trainers={"search": my_algorithm})
```

`compile_program` also accepts
`trainers=`. The mapping is per build; no global registration state is required.
A callable receives:

| Field | Meaning |
|---|---|
| `module` | Initialized network, after imports, freezing and device placement |
| `parameters` | Tuple of parameters that may be updated; excludes frozen parameters |
| `inputs`, `targets` | Training tensors, with cross-entropy targets converted to integers |
| `loss_fn(prediction, target)` | Selected loss, moving targets to the prediction device |
| `config` | Train declaration: epochs, lr, batch, data, loss, validation name |
| `seed` | Model initialization seed |
| `batches()` | Iterator over all configured epochs with deterministic minibatch shuffling |

The callback owns its loop; using `batches()` is optional. The framework does not
create an optimizer, compute gradients or perform hidden updates for custom
trainers. It calls the callback once. It does not assume that one epoch means one
update for every algorithm. Extra algorithm settings can be bound in a Python
closure/callable or implemented by the plugin; arbitrary `.wgt` plugin options
are not supported yet.

Custom declarations require `data`, `loss`, `epochs`; optional settings are
`lr` (default 0.01), `batch` and `validate`. `lr` is a generic positive step size
that the algorithm may use. `optimizer` and `weight_decay` are rejected for a
custom trainer rather than silently ignored. Existing declarations without
`trainer` are unchanged. Explicit `trainer=backprop` requires the usual optimizer
and learning rate. Selecting a trainer requires `wgt 0.3`.

The configured loss and datasets retain the existing language checks, including
BCE output/target and cross-entropy shape requirements. This is a training plugin
interface, not general tensor-programming syntax or a custom-loss registry.

## Checks and provenance

After the callback, the compiler rejects replaced parameter/module objects,
changed tensor shapes/dtypes, modified freeze flags or frozen parameters,
non-finite weights and invalid metrics. It independently evaluates training and
validation loss and saves an ordinary artifact. Errors occur before publishing
a new artifact or replacing an existing one. Frozen-parameter verification uses
one temporary clone of each frozen parameter.

The artifact's `training.trainer` records the trainer name, API version, callable
identity and any reported metrics. CLI-loaded files additionally record their
path and the SHA-256 of the exact source bytes executed. Python callables have
an identity but no inferred source digest. Plugin metrics stay separate from
compiler-computed loss/accuracy. Custom training records do not claim to have
used SGD/Adam or weight decay. Default trainer metadata remains compatible with
existing artifacts.

PyTorch CPU and configured CUDA random streams are isolated and seeded by the
compiler. Plugins using Python `random`, NumPy, external state or additional
CUDA devices must manage those themselves. A source hash identifies the plugin
file, not all dependencies or a complete reproducible environment. The compiler
does not bundle plugin source into artifacts.

## Source-declared Python and installed entry points

With wgt 0.3, replace `trainer=NAME` with `python="hooks.py:train"` (the function
name defaults to `train`) or `entrypoint="NAME"` in the train statement. The
source declaration supplies the callback without a separate CLI binding.
Paths are relative to the model file. Installed entry points use the
`weightlang.trainers` group, must resolve uniquely, and follow the same
TrainingContext and result contracts. These options are mutually exclusive
with `trainer=`. Data callbacks use `weightlang.data_loaders` and a separate
DataContext. See [data and Python loading](datasets.md) for signatures,
metadata registration, limits and runnable examples. Check and --no-train
never import these callbacks; trained artifacts remain independent of them.
