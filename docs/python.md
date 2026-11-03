# Python interoperability

> 当前命令已拆分：`build` 生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例使用 `weightc legacy-build`；详见[分发包、元数据与迁移规范](bundles.md)。


WeightLang's compiler and runtime use PyTorch. The public Python entry points are
`load`, `compile_file`, `compile_program`, `parse`, `parse_file`, `load_artifact`,
`save_artifact`, `Artifact`, `BuildResult`, `Trainer`, and `TrainingContext`.

## Recommended: load a model into Python

Python owns application execution; WGT defines model computation. This entry point
is available in the current source, not yet a published release:

```python
import torch
from weightlang import load

model = load("examples/python_first/model.wgt")
model.train()
x = torch.randn(8, 4)
loss = torch.nn.functional.cross_entropy(model(x), torch.zeros(8, dtype=torch.long))
loss.backward()
```

`load(source: str | Path, *, device="cpu", placements=None) -> torch.nn.Module`
accepts a WGT file or a complete artifact directory. It returns an eval-mode
module. `device` and the optional layer/parameter-to-device `placements` mapping
follow [device placement](devices.md). Construct the optimizer after loading.
Source loading shares initialization, explicit imports, derivations and freezing
with the compiler, preserves the caller's RNG state and writes no artifact.
It skips training data and trainer callbacks even when the source declares them.
Explicit imported dependencies may still use the package resolver/cache.
Directory loading verifies the artifact and restores its saved weights/freezing;
untrained artifacts retain the usual warning. A source-only architecture is
initialized, not trained, and does not automatically discover old checkpoints.

Call the module with a batched tensor; unlike `Artifact.predict`, this does not
coerce lists, add a batch dimension or disable gradients. The current interface
has one tensor input. Multiple independent masks/conditions/caches require new
input contracts and runtime support, not just standard-library functions.
External I/O stays in Python; existing Python modules can wrap the loaded model.

For a Python-only checkpoint use `model.state_dict()` and separately save the
optimizer/scheduler/step/RNG state needed by your training workflow. Restore the
same architecture and load its state dict; freezing is supplied by the architecture,
not a state dict. A state dict alone is not a complete WeightLang artifact.
For complete artifact publication, the existing `Artifact.save` API below remains
the supported route. The [Python-first example](../examples/python_first/train.py)
trains an in-memory model, constructs an initialization baseline only at publication,
transfers the state dict and verifies the saved artifact's predictions exactly.
This publication approach assumes unchanged source and pinned dependencies.
Arbitrary Python wrappers/hooks are not serializable by `Artifact.save`.

## Inspect intermediate values with Python hooks

WGT does not need a `print` statement. Loaded modules already support PyTorch's
`register_forward_hook`, `register_forward_pre_hook` and tensor gradient hooks.
Use `model.named_modules()` to discover paths and `model.get_submodule(path)` to
attach a native hook directly. Remove its returned handle when finished.

The current source also provides a scoped convenience API (not yet released):

```python
import torch
from weightlang import load, forward_hooks

model = load("examples/python_first/model.wgt")

def show(module, inputs, output):
    print(output.detach().flatten()[:6].cpu().tolist())
    # Return None: leave the output unchanged.

with forward_hooks(model, {"hidden.step0": show, "head": show}):
    logits = model(torch.ones(2, 4))
    logits.sum().backward()
# Only hooks installed by this context are now removed.
```

`forward_hooks(model: nn.Module, hooks: Mapping[str, Callable])` is a context
manager. The mapping is required; there are no implicit targets or wildcard
matches. Keys are exact module paths; `""` selects the entire model. All targets
and callbacks are validated before installation. Hooks are installed on context
entry and removed on normal exit or exceptions, including callback exceptions.
Nested contexts and pre-existing native hooks are preserved.

Callbacks receive `(module, inputs, output)` with PyTorch semantics: inputs is a
tuple of positional arguments; merge operators can contain a list of tensors
inside that tuple. Return `None` to observe. Any other return replaces the output
and can change values, shapes and gradients. In-place mutation also changes model
behavior; this is not a read-only sandbox. Callbacks run after successful forward
calls and do not disable autograd. A shared recur/scan cell invokes its hook on
every iteration; hooking the enclosing loop observes the final result instead.
Hooks see runtime module boundaries, not every internal tensor operation.

Use `detach()` for printing and `detach().clone()` for retained snapshots; do not
keep live outputs indefinitely during training. Copying GPU values to CPU for
printing synchronizes execution. Hooks are attached to the module instance, so
concurrent calls on that same instance also trigger them; they are not thread-local.
For gradients, use PyTorch's `tensor.register_hook` when the tensor requires grad.

Hooks are Python debugging/customization code, never serialized in WGT or weights.
`Artifact.save()` rejects active forward hooks; save after leaving the context.
If output replacement defines essential model behavior, removing it changes the
model: represent that behavior in WGT or retain the Python application separately.
See [the runnable debug example](../examples/python_first/debug.py).

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

## Construct or migrate without training

Pass `no_train=True` to `compile_file` / `compile_program` to initialize and reuse
weights without executing the source's training declaration. `compile_file`
also skips loading and validating training data; source syntax and architecture
contracts remain checked. It does not load or invoke a trainer. For example:

```python
result = compile_file(
    "next.wgt", "weights/candidate", no_train=True,
    reuse_source="weights/baseline", reuse_policy="exact",
)
```

For a pre-parsed Program without training-data access, use
`parse_file("next.wgt", check_training=False)` and compile with `no_train=True`.
The compiler does not mutate the Program's training declaration. Imported origins
and freeze flags are retained; new parameters are not labeled as trained.
Direct Python calls still need an explicit reuse_source to select old weights.


## 可选动态 Torch 模型前端（0.3，本地未发布）

`backend torch-dynamic` 提供 `module/setup/fn` 的 Python 兼容可执行模型源码，
支持动态张量、多输入／输出和显式跨调用状态；它与静态 layer/block 图分开，
`check --shapes` 只报告运行时契约，不提供静态形状证明。构造参数沿用对应 Torch 类，
GRU 为 PyTorch reset-after。制品使用独立格式，加载须显式信任源码；Rust、静态导出、
revision/自动复用不支持该格式。完整语法、参数、默认值、保存与验证边界见
[动态张量程序](tensor-programs.md)。

### Explicit decoder execution (local 0.3)

`run_sequence(model, tokens, attention_mask=mask, past_key_values=cache, use_cache=True)`
returns output, per-attention KV pairs, summed training auxiliary loss and explicitly
requested intermediate tensors (`capture=("norm",)`). It supports variable time length
for sequence-polymorphic decoder graphs; it is not a general dynamic-shape or multi-input
WGT syntax. Use one device for this interface. Masks/caches are caller-owned tensors;
concurrent calls do not overwrite module state. See the Chinese reference's explicit
sequence section for shapes, YaRN, sparse MoE and tied embedding projection contracts.
Uniform-device modules preserve parameter aliases and follow subsequent `Module.to(...)`;
explicit mixed placements retain their device routing and cannot split tied parameters.

`compile_program(program, out_dir, no_train=True, initial_state=state_dict)` can
materialize a complete caller-supplied float32 snapshot, including already-frozen
parameters. It requires exact tensor names/shapes, finite values and consistent
alias values; training, imports, inheritance, derivation and reuse cannot be mixed
with this option. This initializes a new artifact and does not modify a frozen
baseline. Tensor provenance is `edited` with `caller-supplied` metadata, never
silently certified as trained. Existing artifact updates still use `Artifact.save`
and retain its frozen-weight checks. MiniMind uses this for frozen snapshots.


具名多输入/输出、bool 条件和显式 scan 初态已加入本地 0.3 静态图。
返回字典与固定形状检查、制品 v4 及限制见 [端口契约](dataflow.md#named-tensor-boundaries-and-explicit-scan-state-03-localunreleased)。
符号维度、KV cache 图端口和多状态循环仍未完成。


Custom source initialization accepts `init file="init.wgt"` or
`init python="generate.py:initialize"` (optional seed, default 0).
The Python callback receives immutable tensor specifications and returns a complete,
validated CPU state dictionary; `load` executes it once during construction,
without writing artifacts or invoking training. See [initialization](initialization.md)
for the full callback contract, rule dialect and executable examples.


本地 0.3 已新增 `dim NAME min=N max=M` 和 `require LEFT OP RIGHT`，用于同一具名模型
接受有限范围内的运行时维度。每个符号需直接绑定某个输入轴，编译器检查所有合法组合，
运行时先验证约束。制品 v5 保留表达式，详见 [有界维度及限制](dimensions.md#bounded-symbolic-dimensions-03-localunreleased)。
这不代表通用动态形状、零长度 cache 或 KV/MoE 图端口已完成。

## Structured configuration and training state

The public package exposes `model_config`, `package_model`, `parameter_inventory`,
`migrate_parameters`, `save_checkpoint`, `checkpoint_report`, `restore_checkpoint`,
and `merge_adapter`/`unmerge_adapter`/`save_adapter`/`load_adapter`. See
[composition contracts](composition.md) for complete signatures and compatibility
limits. `load(source, config={...})` builds the configured model in memory without
training or artifact output. Configuration overrides are not accepted for fixed artifacts.
