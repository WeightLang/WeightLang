"""Causal experiments through the public SDK and shared native compiler."""
from pathlib import Path
import pytest
import torch
from weightlang import torch_api as wgt

SOURCE = Path(__file__).resolve().parents[2] / "examples/native_training/interpretability/model.wgt"

def test_capture_ablate_patch_and_preserve():
    model = wgt.load(SOURCE)
    value = torch.tensor([2., 3.])
    output, events = wgt.inspect(model, value)
    torch.testing.assert_close(output, torch.tensor([6., 12.]))
    torch.testing.assert_close(events[0].before, torch.tensor([4., 9.]))
    ablated, _ = wgt.inspect(model, value, replacements={"hidden": torch.zeros(2)})
    torch.testing.assert_close(ablated, value)
    patched, _ = wgt.inspect(model, torch.ones(2), replacements={"hidden": events[0].before})
    torch.testing.assert_close(patched, torch.tensor([5., 10.]))
    with pytest.raises(RuntimeError, match="budget"):
        wgt.inspect(model, value, max_bytes=0)
    with pytest.raises(ValueError, match="unknown"):
        wgt.inspect(model, value, replacements={"missing": torch.zeros(2)})
    torch.testing.assert_close(model(value), output)

def test_multiaxis_and_gradient_identity():
    model = wgt.loads("""wgt 0.5
model Matrix
input x:f32[4]
output y:f32[4]
fn forward(x) {
    let matrix = x as [2,2];
    let hidden = observe("matrix", matrix.T);
    return hidden as [4];
}
""")
    value = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    model(value).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones(4))
    output, events = wgt.inspect(model, value)
    assert events[0].shape == (2, 2)
    torch.testing.assert_close(output, torch.tensor([1., 3., 2., 4.]))
    patched, _ = wgt.inspect(model, value, replacements={"matrix": torch.zeros(2, 2)})
    torch.testing.assert_close(patched, torch.zeros(4))


CIRCUIT = """wgt 0.5
model Circuit
input x:f32[1]
output y:f32[1]
fn helper(x) { return observe("a",x*x); }
fn forward(x) {
    let a = helper(x);
    let b = observe("b",a+a);
    return b+x;
}
"""


def test_attribution_and_circuit_match_native_oracle():
    import json
    import os
    import subprocess
    oracle = os.environ.get("WGT_REFERENCE",
                            str(SOURCE.parents[3] / "target/debug/examples/torch_reference"))
    native = json.loads(subprocess.check_output(
        [oracle], input=json.dumps({"source": CIRCUIT, "input": [2.],
                                    "mode": "research", "gradient": [1.]}).encode()))
    model = wgt.loads(CIRCUIT)
    value, seed = torch.tensor([2.]), torch.tensor([1.])
    report = wgt.attribute(model, value, seed)
    torch.testing.assert_close(report.output, torch.tensor(native["attribution"]["output"]))
    torch.testing.assert_close(report.input_gradient, torch.tensor(native["attribution"]["input_gradient"]))
    assert len(report.events) == len(native["attribution"]["events"])
    for event, expected in zip(report.events, native["attribution"]["events"]):
        assert event.site == expected["site"]
        torch.testing.assert_close(event.gradient, torch.tensor(expected["gradient"]))
        torch.testing.assert_close(event.contribution, torch.tensor(expected["contribution"]))
    graph = wgt.circuit(model, value, seed, epsilon=1e-6)
    assert graph.edges == native["circuit"]["edges"]
    assert graph.work == native["circuit"]["work"]
    assert report.context == native["attribution"]["context"]
    assert graph.context == native["circuit"]["context"]
    assert "Output target" in graph.mermaid()
    json.dumps(graph.to_dict())


def test_targeted_occurrences_detach_and_failed_experiments():
    model = wgt.loads("""wgt 0.5
model Loop
input x:f32[1]
output y:f32[1]
parameter count:f32[1]=[0]
fn forward(x) {
    let remaining = [2];
    let value = x;
    while positive(remaining) {
        value = observe("iteration",value*value);
        count <- count+[1];
        remaining = remaining-[1];
    }
    return value;
}
""")
    value, seed = torch.tensor([2.]), torch.tensor([1.])
    report = wgt.attribute(model, value, seed)
    torch.testing.assert_close(report.input_gradient, torch.tensor([32.]))
    assert [e.occurrence for e in report.events] == [0, 1]
    torch.testing.assert_close(report.events[0].gradient, torch.tensor([8.]))
    _, events = wgt.inspect(model, value)
    output, patched = wgt.inspect(model, value, interventions=[
        wgt.Intervention(events[0].instruction, 1, torch.tensor([3.]))])
    torch.testing.assert_close(output, torch.tensor([3.]))
    torch.testing.assert_close(patched[0].after, torch.tensor([4.]))
    with pytest.raises(ValueError, match="not executed"):
        wgt.inspect(model, value, interventions=[
            wgt.Intervention(events[0].instruction, 5, torch.tensor([3.]))])
    with pytest.raises((ValueError, RuntimeError), match="budget"):
        wgt.circuit(model, value, seed, max_work=1)
    torch.testing.assert_close(model(value), torch.tensor([16.]))
    counter = model.get_parameter(model.parameter_names["count"])
    torch.testing.assert_close(counter, torch.tensor([0.]))

    detached = wgt.loads("""wgt 0.5
model Detached
input x:f32[1]
output y:f32[1]
fn forward(x) { return observe("detached",detach(x)); }
""")
    result = wgt.attribute(detached, value, seed)
    torch.testing.assert_close(result.input_gradient, torch.zeros(1))
    torch.testing.assert_close(result.events[0].gradient, torch.ones(1))


def test_sparse_features_retain_residual_and_measure_actual_effects():
    dictionary = wgt.SparseAutoencoder(2, 2)
    with torch.no_grad():
        dictionary.encoder.weight.copy_(torch.eye(2))
        dictionary.decoder.weight.copy_(torch.eye(2))
        dictionary.encoder.bias.zero_()
        dictionary.decoder.bias.zero_()
    activation = torch.tensor([4., 9.])
    torch.testing.assert_close(dictionary.intervene(activation, 0, 0.), torch.tensor([0., 9.]))
    model = wgt.load(SOURCE)
    value = torch.tensor([2., 3.])
    _, events = wgt.inspect(model, value)
    report = wgt.feature_effects(model, value, dictionary, events[0].instruction, torch.ones(2))
    assert report["effects"] == [4., 9.]
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = dictionary.loss(torch.tensor([[1., 2.], [2., 1.]]))
    loss.backward()
    assert dictionary.encoder.weight.grad is not None
    optimizer.step()


def test_wgt_sparse_autoencoder_trains_and_exports_usable_dictionary():
    model = wgt.load(SOURCE.parent / "sparse_autoencoder/model.wgt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    values = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0.5, 0.25]])
    initial = model.loss(model(values), values).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = model.loss(model(values), values)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.3
    dictionary = wgt.SparseAutoencoder.from_wgt(model)
    output = model(values)
    torch.testing.assert_close(dictionary(values), output[:, :2])
    torch.testing.assert_close(dictionary.encode(values), output[:, 2:])
    exported = dictionary.export()
    assert exported["width"] == 2 and exported["features"] == 4
    assert len(exported["encoder"]) == 8


def test_feature_dictionary_matches_native_oracle_and_records_snapshot():
    import json
    import os
    import subprocess
    model = wgt.load(SOURCE)
    dictionary = wgt.SparseAutoencoder(2, 2)
    with torch.no_grad():
        dictionary.encoder.weight.copy_(torch.eye(2))
        dictionary.encoder.bias.zero_()
        dictionary.decoder.weight.copy_(0.5 * torch.eye(2))
    value, seed = torch.tensor([2., 3.]), torch.ones(2)
    _, events = wgt.inspect(model, value)
    instruction = events[0].instruction
    report = wgt.feature_effects(model, value, dictionary, instruction, seed)
    oracle = os.environ.get("WGT_REFERENCE",
                            str(SOURCE.parents[3] / "target/debug/examples/torch_reference"))
    native = json.loads(subprocess.check_output([oracle], input=json.dumps({
        "source": SOURCE.read_text(), "input": [2., 3.], "gradient": [1., 1.],
        "mode": "features", "dictionary": dictionary.export(), "instruction": instruction,
    }).encode()))
    assert report["effects"] == native["effects"]
    assert report["context"] == native["context"]
    assert report["dictionary_fingerprint"] == native["dictionary_fingerprint"]

    trainable = wgt.load(SOURCE.parent / "sparse_autoencoder/model.wgt")
    before = wgt.attribute(trainable, value, torch.ones(6)).context
    native_dictionary = json.loads(subprocess.check_output([oracle], input=json.dumps({
        "source": (SOURCE.parent / "sparse_autoencoder/model.wgt").read_text(),
        "input": [2., 3.], "gradient": [1.] * 6, "mode": "research",
    }).encode()))
    assert before == native_dictionary["attribution"]["context"]
    with torch.no_grad():
        trainable.get_parameter(trainable.parameter_names["encoder"]).add_(0.01)
    after = wgt.attribute(trainable, value, torch.ones(6)).context
    assert before["state_fingerprint"] != after["state_fingerprint"]
    assert before["model_fingerprint"] == after["model_fingerprint"]


def test_transformer_source_fingerprints_and_attribution_match_native():
    import json
    import os
    import subprocess
    root = SOURCE.parents[3]
    source = (root / "crates/wgt-model/tests/fixtures/transformer_encoder/model.wgt").read_text()
    source = source.replace('let probabilities = softmax(scores,"key_sequence");',
                            'let probabilities = observe("attention",softmax(scores,"key_sequence"));')
    value = torch.tensor([1., 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8])
    seed = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    model = wgt.loads(source)
    report = wgt.attribute(model, value, seed)
    oracle = os.environ.get("WGT_REFERENCE", str(root / "target/debug/examples/torch_reference"))
    native = json.loads(subprocess.check_output([oracle], input=json.dumps({
        "source": source, "input": value.tolist(), "gradient": seed.tolist(), "mode": "research",
    }).encode()))
    assert report.context == native["attribution"]["context"]
    torch.testing.assert_close(report.input_gradient,
                               torch.tensor(native["attribution"]["input_gradient"]), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(report.events[0].gradient.reshape(-1),
                               torch.tensor(native["attribution"]["events"][0]["gradient"]), rtol=1e-4, atol=1e-6)
