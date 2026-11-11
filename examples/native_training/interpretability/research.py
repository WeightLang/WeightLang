"""Run capture, attribution, circuit discovery and learned-feature interventions."""
import json
from pathlib import Path

import torch
import weightlang as wgt


def main():
    directory = Path(__file__).resolve().parent
    model = wgt.load(directory / "model.wgt")
    value = torch.tensor([2., 3.])
    seed = torch.ones(2)
    output, events = wgt.inspect(model, value)
    attribution = wgt.attribute(model, value, seed)
    circuit = wgt.circuit(model, value, seed)

    # Host code supplies the activation dataset and optimizer. WGT owns the
    # dictionary architecture and differentiable reconstruction/sparsity loss.
    dictionary_model = wgt.load(directory / "sparse_autoencoder/model.wgt")
    dataset = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0.5, 0.25]])
    optimizer = torch.optim.Adam(dictionary_model.parameters(), lr=0.01)
    initial_loss = dictionary_model.loss(dictionary_model(dataset), dataset).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = dictionary_model.loss(dictionary_model(dataset), dataset)
        loss.backward()
        optimizer.step()
    dictionary = wgt.SparseAutoencoder.from_wgt(dictionary_model)
    effects = wgt.feature_effects(model, value, dictionary, events[0].instruction, seed)

    print(json.dumps({
        "output": output.tolist(),
        "activation": events[0].before.tolist(),
        "input_gradient": attribution.input_gradient.tolist(),
        "activation_gradient": attribution.events[0].gradient.tolist(),
        "circuit": circuit.to_dict(),
        "diagram": circuit.mermaid(),
        "dictionary_initial_loss": initial_loss,
        "dictionary_final_loss": loss.item(),
        "feature_effects": effects["effects"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
