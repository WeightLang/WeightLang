# Artifact format (version 2)

`weightc build --no-store model.wgt -o weights` writes `weights/<model>/`:

```
weights/xor_mlp/
  manifest.json         structure, I/O contract, provenance, hashes, versions
  weights.safetensors   float32 tensors, named <layer>.weight / <layer>.bias
  source.wgt            exact copy of the source that produced the weights
```

Loading never unpickles: tensors are read with `safetensors`, and the network is
rebuilt from the declarative `layers` list. No Python code is stored in or
executed from an artifact.

## manifest.json

| Field | Meaning |
|---|---|
| `format`, `format_version` | `"weightlang.artifact"`, `2`. This loader also accepts v1. |
| `language_version` | `.wgt` version of the source (`"0.1"` or `"0.2"`). |
| `model` | Model name. |
| `compiler` | `{"name": "weightc", "version": ...}` |
| `built_with` | PyTorch and Python versions used for the build. |
| `source` | `{"file": "source.wgt", "sha256": ...}` |
| `interface` | Input/output names, dtype (`f32`) and feature shape. Inference takes `[features]` or `[N, features]`. |
| `layers` | Ordered layer objects; seq/residual contain nested `layers` and an optional `block` identity (e.g. std/nn.mlp). |
| `libraries` | Top-level use alias → name, sha256 and bundled flag. |
| `init` | Seed and initialisation method. |
| `imports` | For each imported layer: source path, source model and layer, source weights and manifest sha256, whether it was pinned and frozen. |
| `training` | `null` if untrained; otherwise dataset name and sha256, rows, loss, optimizer, lr, epochs, initial/final loss and (for bce/cross_entropy) training-set accuracy. Includes batch, weight_decay, `dataset` metadata and optional `validation` with loss, accuracy and dataset. |
| `weights` | File name, sha256 of the file, and per tensor: shape, dtype, `origin` and (for imports) `upstream_origin`. |

## Verification on load

`load_artifact()` / `weightc verify` check, in order: manifest format version;
sha256 of `weights.safetensors` and `source.wgt`; that the tensor set exactly
matches the architecture, with matching shapes and `float32` dtype; that every
origin is known. Any failure is an error, not a warning. Untrained tensors
produce an `UntrainedWeightsWarning`.

## Byte stability

`weightc` writes the safetensors file itself with tensors and header keys in
sorted order (the format is standard and is read back with the official
loader). The reference `safetensors.torch.save_file` does not keep header order
stable, which made the same weights hash differently between runs. Byte-stable
output is what makes `sha256` pins in `import` and the future lock file usable.

Same source + same PyTorch version + same CPU produces identical bytes (tested).
Across PyTorch versions or hardware, floating-point training results may differ;
reproducibility is then checked by the recorded hashes, not assumed.

## Using an artifact

With WeightLang installed:

```python
from weightlang import load_artifact

model = load_artifact("weights/xor_mlp")
model.predict([[0, 1], [1, 1]])   # tensor of shape [2, 1]
model.module                      # the torch.nn.Sequential
model.manifest                    # parsed manifest.json
```

Without WeightLang (only `torch` and `safetensors`) — this does not verify hashes:

```python
import json
from collections import OrderedDict
import torch
from safetensors.torch import load_file

root = "weights/xor_mlp"
manifest = json.load(open(f"{root}/manifest.json"))
class Residual(torch.nn.Sequential):
    def forward(self, x):
        return x + super().forward(x)

acts = {
    "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid,
    "gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "identity": torch.nn.Identity,
}

def layer(l):
    op, a = l["op"], l.get("args", {})
    if op in ("seq", "residual"):
        cls = Residual if op == "residual" else torch.nn.Sequential
        return cls(OrderedDict((c["name"], layer(c)) for c in l["layers"]))
    if op == "linear":
        return torch.nn.Linear(a["in"], a["out"])
    if op == "layernorm":
        return torch.nn.LayerNorm(a["dim"])
    if op == "dropout":
        return torch.nn.Dropout(a["p"])
    if op == "softmax":
        return torch.nn.Softmax(dim=-1)
    return acts[op]()

net = torch.nn.Sequential(OrderedDict((l["name"], layer(l)) for l in manifest["layers"]))
net.load_state_dict(load_file(f"{root}/weights.safetensors"))
net.eval()
```


## Compatibility and dataset metadata

New builds emit v2, including builds of wgt 0.1 sources. Existing v1 artifacts
remain readable; no migration or reserialization is required. The original
gates v0.1.0 consumer and lock file remain unchanged.

Nested tensor names use dots (e.g. body.fc1.weight). A block records its source
identity as `"block": "std/nn.mlp"`; residual layers retain the same nested
children and add their result to the input.

`training.dataset` and `training.validation.dataset` record name, rows and
sha256 of canonical numeric data. File datasets additionally record file and
file_sha256 (the original bytes); license and source are recorded when supplied.
`training.validation` contains loss and, for classification, accuracy, measured
after training in eval mode. train_accuracy and final_loss refer to training
rows only. Dataset files and library sources are not copied into artifacts;
rerunning a build requires the original repository data/libraries.


## Format v3 (main, unreleased)

Sources declaring wgt 0.3 emit format_version 3. The current loader and native
Rust verifier read versions 1, 2 and 3; released weightc 0.2.0 cannot load v3.
Existing 0.1/0.2 sources still emit v2 with unchanged tensor initialization.

V3 interfaces allow i64 token inputs and multidimensional sample shapes.
Layers additionally record `input` and `output` objects with dtype and shape;
loaders check these against recursively inferred contracts. Parameter tensors
remain F32 safetensors. Embedding owns `weight[V,D]`, learned positions own
`weight[M,D]`, and self-attention owns `qkv.weight[3D,D]`, `qkv.bias[3D]`,
`proj.weight[D,D]`, `proj.bias[D]`. Conv2d owns `weight[O,I,K,K]` and `bias[O]`.

The standalone PyTorch loading recipe above covers v1/v2. V3 also needs the
small modules in [modern.py](../src/weightlang/modern.py), which depend only on
PyTorch; copy their definitions alongside that recipe and route the new op
names through `build_modern`. No Python objects are serialized in artifacts.

## Directly edited artifacts (main)

A weight patch writes artifact v3 and marks changed tensors with origin "edited".
The edit_history field embeds patches bound to baseline weight and manifest SHA-256,
the hash of each supplied patch, numeric changes and previous metadata/metrics.
Current training metrics are cleared. Untouched tensors retain origins; imported
edited tensors retain upstream_origin "edited" and require v3 even with a
vector-only wgt 0.2 source. History is not a signature. See
[the patch workflow](weight-space.md#direct-edits-main-unreleased).

## Lossless revision storage

Artifacts can be retained in a separate content-addressed
[revision store v1](specs/revision-v1.md). Each revision references complete files
through shared byte objects and optionally records a parent. This does not change
artifact versions 1–3: checkout restores all three original files byte-for-byte.
Use checkout before passing a stored revision to existing loaders, Rust tools or
package publishers. [Selective training and examples](weight-iterations.md).

## Native serialization

Weight serialization uses the existing safetensors Rust extension while preserving
WeightLang's canonical file bytes and hashes. Python handles metadata and orchestration.
See [local benchmarks and compatibility checks](native-performance.md).
