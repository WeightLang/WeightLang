"""A WGT updater proposes weights; Python explicitly binds or commits them.

This small prescribed-feedback example is not a trained controller or benchmark.
Run with --native to also apply its output to the independent Rust runtime.
"""
from pathlib import Path
import argparse

import torch
import weightlang as wgt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model = wgt.load(root / "model.wgt")
    updater = wgt.load(root / "updater.wgt")
    wgt.write_parameters(model, {"projection.weight": torch.zeros(1, 2)})
    wgt.write_parameters(updater, {"direction.weight": torch.eye(2) * 0.25})
    memory = torch.zeros(1, 2)
    desired = torch.tensor([[2., -1.]])
    x = torch.tensor([3., 1.])
    with torch.no_grad():
        for _ in range(12):
            weights = wgt.read_parameters(model)["projection.weight"]
            # A fully observable toy feedback signal; no loss gradient is computed.
            proposal = updater(weights=weights, feedback=desired - weights, memory=memory)
            bindings = {"projection.weight": proposal["next_weights"]}
            preview = wgt.call_with_parameters(model, bindings, x)
            wgt.write_parameters(model, bindings)
            torch.testing.assert_close(model(x), preview)
            memory = proposal["next_memory"]
    print("updated prediction:", model(x).detach().tolist(), "target: 5.0")
    if args.native:
        from weightlang.native import NativeSession
        with NativeSession.from_source(root / "model.wgt", {"projection.weight": [0., 0.]},
                                       budget_bytes=128, device="cpu") as session:
            weights = wgt.read_parameters(model)["projection.weight"].flatten().tolist()
            session.write_parameters([dict(name="projection.weight", values=weights)])
            torch.testing.assert_close(torch.tensor(session(x.tolist())), model(x))
            print("native prediction:", session(x.tolist()))


if __name__ == "__main__":
    main()
