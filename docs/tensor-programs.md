# Dynamic Torch tensor programs (0.3, unreleased)

The [unification work](migrations/unified-language.zh-CN.md) targets the declarative
WGT graph as the single model language while retaining dynamic models in WGT.
This legacy frontend remains operational until those models and artifact paths
have been migrated and validated. It has **not** yet been replaced; the native
math increment alone is not a completed migration.

`backend torch-dynamic` is an opt-in, Python-compatible model frontend. It is
separate from the static `layer`/`block` graph: it compiles trusted module and
function declarations to Python/PyTorch, preserving dynamic control flow and
Torch operator semantics. It does not offer static shape inference, a sandbox,
or a Rust-native execution path. Existing static programs are unchanged.

```wgt
wgt 0.3
backend torch-dynamic
host weightlang.tensor_program
extern

# x: [batch,time,3], state: [2,batch,4] or None.
# Both GRU layers share their respective parameters across time. State is
# explicit and can continue a stream; callers decide when to reset it.
module Memory(Module):
    setup(self, width=4):
        super().__init__()
        self.memory = gru(3, width, 2, batch_first=True)
        self.head = linear(width, 2)
    fn forward(self, x, state=None):
        output, state = self.memory(x, state)
        return self.head(output), state
```

This source only initializes parameters. It has no training directive or data.
The example's host is the backend itself; application programs name their
installed host adapter module. `extern` with no names is valid.

## Source contract

The first two lines are exactly `wgt 0.3` and `backend torch-dynamic`.
`host dotted.module` gives the Python module that owns the exported classes.
`extern name, other` explicitly names bindings obtained from that installed
module. `bind_tensor_program(path, bindings)` instead receives those bindings
from a host adapter during normal import. Bindings are not automatically model
weights; preprocessing, dataclasses and deterministic runtime services can stay
in the host. Host adapters must not implement the migrated network forward.

`module Name(Base):` declares a class; `Module` is the Torch module base.
`setup(self, ...)` is its constructor. `fn name(...)` and `async fn name(...)`
declare methods or top-level helpers. Python-compatible signatures, expressions,
slices, annotations, decorators, comprehensions, loops, exceptions and return
values are lowered with the Python AST compiler. Tensor attributes and calls use
Torch semantics, including broadcasting and masks. Arbitrary host bindings make
this trusted executable code, even though import/global/nonlocal statements and
direct eval/exec/open/compile calls are rejected. This rejection is a diagnostic
restriction, not a security boundary. Do not load untrusted programs.

`torch`, `F` (torch.nn.functional) and `Module` are backend-provided names.
The following constructors map exactly to the installed `torch.nn` classes;
all positional/keyword arguments, defaults, initializers, train/eval behavior
and weight layouts are those of that class. There is no approximate lowering:

| Constructor | Torch class |
| --- | --- |
| `linear` | Linear |
| `embedding` | Embedding |
| `layernorm` | LayerNorm |
| `gru` | GRU (reset-after, unlike static std/recurrent) |
| `gru_cell` | GRUCell (reset-after) |
| `conv1d` | Conv1d |
| `gelu`, `relu`, `tanh` | GELU, ReLU, Tanh |
| `sequential` | Sequential |
| `parameter` | Parameter |
| `transformer_encoder` | TransformerEncoder |
| `transformer_layer` | TransformerEncoderLayer |
| `multihead_attention` | MultiheadAttention |

Input shapes include batch; there is no implicit batch convention imposed by
this frontend. Module guards and Torch validate runtime shapes/dtypes. Repeated
calls to one stored layer share parameters. Storing two different layers creates
independent parameters. Freezing uses `requires_grad` and persists in artifacts.

## Checking, building and loading

`weightc check file.wgt --shapes` parses only, without executing class bodies,
constructors or host imports. It reports dynamic contracts, **not** a shape proof.
Execution tests must cover representative dynamic branches and state.

```sh
weightc build memory.wgt --no-train --no-store --tensor-model Memory \
  --tensor-args '{"width":4}' -o weights/new
weightc verify weights/new/Memory --trust-tensor-source
```

`--tensor-model` is optional only for a source containing one module.
`--tensor-args` is a JSON object of constructor keyword arguments. Build requires
`--no-train --no-store`; it never trains or uses revision HEAD. Force, reuse,
placement maps, static exporters/trace/debug and the Rust artifact reader do not
support this format. A whole-module device can be selected; saving reloads CPU/eval.
Python `compile_file(..., no_train=True, tensor_model=..., tensor_args={...})`
dispatches to this frontend. `parse_file`/`compile_program` remain static APIs;
use `parse_tensor_program` for this frontend.

`save_tensor_module(module, exact_new_path, args=(), kwargs={}, training=None,
origin="random-init", metadata=None)` writes a complete artifact. `kwargs`/`args`
must recreate the model: JSON values, tuples and dataclasses are supported.
Arbitrary Python objects and optimizer state are not serialized. It checks tensor
keys/shapes/dtypes, module structure, finite values, hooks and replaced methods.
Source snapshots, host file hashes, constructor values, tensor contracts and
SHA256 hashes accompany safetensors. Instantiation can execute trusted host code.

`load_artifact(path, trusted_tensor_source=True)` or
`load_tensor_artifact(path, trusted=True)` verifies and loads this format.
Without the explicit trust option it is rejected. The installed host adapters
must match recorded hashes. Saved WGT definitions are compiled from the bundle,
not from current editable WGT. `.save(new_path, training=...)` / `save_artifact`
preserve constructor/source and reject overwrites and changed frozen parameters.
Training records are caller-reported, not proof of model quality. Use `.module`
for gradients and optimizer steps; `.predict(*args, **kwargs)` disables gradients.
The manifest format is `weightlang.tensor-program`, version 1, deliberately
separate from `weightlang.artifact` static graphs so old consumers fail closed.

Verified locally: source changes affect the normal forward path, streaming/full
sequence equivalence, parameter updates, artifact roundtrip, source snapshots,
freeze behavior, overwrite/hash/hook/nonfinite rejection, and existing static
frontend regressions. The LightNest migration separately compares 21 cases with
its original CPU/f32 networks, gradients and one optimizer step. No release or
model-quality claim follows from these checks.
