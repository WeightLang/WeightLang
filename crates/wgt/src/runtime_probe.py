"""Run with the managed interpreter in isolated mode, never with system Python."""

import importlib.metadata
import json
import platform
import sys

sys.path.insert(0, sys.argv[1])
import torch
import safetensors  # noqa: F401 -- verify this runtime dependency imports successfully
import weightlang.cli

backend = sys.argv[2]
if backend == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable: check the NVIDIA driver and PyTorch build")
    device = "cuda:0"
elif backend == "mps":
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable on this macOS/PyTorch combination")
    device = "mps"
else:
    device = "cpu"
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
y = (x @ x.T).sum()
y.backward()
assert y.item() == 52.0 and torch.isfinite(x.grad).all().item()
packages = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
print(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "packages": dict(sorted(packages.items())),
            "tensor_test": "forward+backward passed",
            "compiler": weightlang.__version__,
        }
    )
)
