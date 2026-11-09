"""Learn an iteration count with PyTorch, then save and inspect the WGT model."""

import argparse
from pathlib import Path

import torch

import weightlang as wgt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/sdk-adaptive"),
                        help="New output directory; existing model artifacts are preserved")
    output = parser.parse_args().output
    if output.exists():
        parser.error(f"{output} already exists; choose a new directory with --output")

    model = wgt.load(Path(__file__).with_name("model.wgt"))
    depth = model.get_parameter(model.parameter_names["depth"])
    inputs = torch.tensor([1.0], device=depth.device)
    targets = torch.tensor([4.0], device=depth.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    wgt.bind_optimizer(model, optimizer)

    model.train()
    initial_depth = depth.item()
    initial_loss = model.loss(model(inputs), targets).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        prediction = model(inputs)
        final_loss = model.loss(prediction, targets).item()
    torch.testing.assert_close(prediction, targets)
    assert final_loss < initial_loss, "training should reduce the declared WGT loss"
    print(f"device={depth.device}, loss {initial_loss:.6f} -> {final_loss:.6f}")
    print(f"depth shadow: {initial_depth:.6f} -> {depth.item():.6f}; input=1, prediction={prediction.item():.6f}")

    destination = wgt.save(model, output / "model.pt")
    restored = wgt.load(destination, device=depth.device).eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), prediction)
    diagram = wgt.visualize(restored, output / "architecture.mmd")
    print(f"Saved model: {destination}")
    print(f"IR graph: {diagram}")


if __name__ == "__main__":
    main()
