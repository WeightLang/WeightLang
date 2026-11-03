# Artifact format (version 1)

`weightc build model.wgt -o build` writes `build/<model>/`:

```
build/xor_mlp/
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
| `format`, `format_version` | `"weightlang.artifact"`, `1`. Loaders reject other versions. |
| `language_version` | `.wgt` version of the source (`"0.1"`). |
| `model` | Model name. |
| `compiler` | `{"name": "weightc", "version": ...}` |
| `built_with` | PyTorch and Python versions used for the build. |
| `source` | `{"file": "source.wgt", "sha256": ...}` |
| `interface` | Input/output names, dtype (`f32`) and feature shape. Inference takes `[features]` or `[N, features]`. |
| `layers` | Ordered list of `{"name", "op", "args"}`; the whole architecture. |
| `init` | Seed and initialisation method. |
| `imports` | For each imported layer: source path, source model and layer, source weights and manifest sha256, whether it was pinned and frozen. |
| `training` | `null` if untrained; otherwise dataset name and sha256, rows, loss, optimizer, lr, epochs, initial/final loss and (for `bce`) training-set accuracy. |
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

model = load_artifact("build/xor_mlp")
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

root = "build/xor_mlp"
manifest = json.load(open(f"{root}/manifest.json"))
acts = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}
net = torch.nn.Sequential(OrderedDict(
    (l["name"], torch.nn.Linear(l["args"]["in"], l["args"]["out"]) if l["op"] == "linear" else acts[l["op"]]())
    for l in manifest["layers"]
))
net.load_state_dict(load_file(f"{root}/weights.safetensors"))
net.eval()
```
