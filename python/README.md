# WeightLang Python SDK

Describe the architecture and forward computation in `model.wgt`, then load a
regular PyTorch `torch.fx.GraphModule`. Python owns data loading, training,
inference, checkpoints, and visualization.

Install from a source checkout:

```sh
cargo install --locked --force --path crates/wgt --bin wgt
uv venv
uv pip install -e './python[visualize]'
```

The compiler must come from the same source checkout with `wgt compile --target
torch` support. Set `WGT_COMPILER=/path/to/wgt` to choose an executable. Compiled
JSON and saved SDK models load without the compiler.

```python
import torch
import weightlang as wgt

model = wgt.load('model.wgt')
device = next(model.parameters()).device
x = torch.randn(16, model.input_shape[0], device=device)
y = model(x)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(wgt.summary(model))
print(wgt.visualize(model))  # Mermaid text, with real computation edges
wgt.save(model, 'weights/demo/model.pt')
restored = wgt.load('weights/demo/model.pt').eval()
```

`device='auto'` selects available CUDA, XPU, MPS, then CPU. Use `device='cpu'` or
another explicit PyTorch device to override it. Source definitions initialize
weights; they do not train them. Call `eval()` and `torch.inference_mode()` for
inference. An SDK saved model contains architecture and `state_dict`; host
optimizer state can be saved separately using PyTorch.

The [SDK guide](https://github.com/WeightLang/WeightLang/blob/main/docs/python-sdk.md)
documents installation, complete training, supported operations, and current
limits. The local [MLP example](examples/mlp/train.py) runs a complete training and
save/reload flow. This SDK is developed in the repository; it has not been
published to PyPI by this change.
