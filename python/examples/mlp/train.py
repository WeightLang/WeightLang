"""Train a WGT architecture with a standard PyTorch loop; no data downloads."""

from pathlib import Path

import torch
from torch import nn

import weightlang as wgt


def main() -> None:
    model = wgt.load(Path(__file__).with_name("model.wgt"))
    device = next(model.parameters()).device
    inputs = torch.linspace(-1, 1, 64, device=device).unsqueeze(-1)
    targets = 2 * inputs + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()

    model.train()
    initial = criterion(model(inputs), targets).item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        final = criterion(model(inputs), targets).item()
    print(f"device={device}, MSE {initial:.6f} -> {final:.6f}")
    print(wgt.summary(model))

    destination = wgt.save(model, "weights/sdk-mlp/model.pt")
    restored = wgt.load(destination, device=device).eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), model(inputs))
    wgt.visualize(model, "weights/sdk-mlp/architecture.mmd")
    print(f"Saved model: {destination}")


if __name__ == "__main__":
    main()
