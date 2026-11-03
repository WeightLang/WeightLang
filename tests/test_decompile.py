import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.decompile import decompile, parse_inventory, tensor_inventory
from weightlang.syntax import WgtError, parse_file


def checkpoint(path, tensors=None):
    save_file(tensors if tensors is not None else {"linear.weight": torch.arange(6).reshape(2, 3)}, str(path))
    return path


def test_safetensors_references_actual_bytes(tmp_path):
    source = checkpoint(tmp_path / "model.safetensors")
    out = tmp_path / "source.wgt"
    report = decompile(str(source), out, references=True)
    assert report["parameters"] == 6 and not report["executable"]
    doc = parse_inventory(out.read_text())
    tensor = doc["tensors"]["linear.weight"]
    assert tensor["dtype"] == "I64" and tensor["shape"] == [2, 3]
    metadata = doc["sources"][tensor["source"]]
    with Path(metadata["uri"]).open("rb") as stream:
        stream.seek(metadata["data_start"] + tensor["data_offsets"][0])
        assert struct.unpack("<6q", stream.read(48)) == tuple(range(6))
    assert main(["check", str(out)]) == 0
    assert main(["legacy-build", str(out), "-o", str(tmp_path / "build")]) == 1
    assert not (tmp_path / "build").exists()
    with pytest.raises(WgtError, match="already exists"):
        decompile(str(source), out, references=True)


def test_shards_and_index(tmp_path):
    checkpoint(tmp_path / "a.safetensors", {"a": torch.ones(2)})
    checkpoint(tmp_path / "b.safetensors", {"b": torch.zeros(3, dtype=torch.bfloat16)})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
    (tmp_path / "config.json").write_text('{"model_type": "test"}')
    doc = tensor_inventory(str(tmp_path))
    assert len(doc["sources"]) == 2 and doc["config"]["model_type"] == "test"
    assert doc["tensors"]["b"]["dtype"] == "BF16"
    index.write_text(json.dumps({"weight_map": {"wrong": "a.safetensors"}}))
    with pytest.raises(WgtError, match="does not match"):
        tensor_inventory(str(index))
    index.write_text('{"weight_map": {"a": "../escape.safetensors"}}')
    with pytest.raises(WgtError, match="invalid shard"):
        tensor_inventory(str(index))


@pytest.mark.parametrize(
    "header,payload",
    [
        ({"a": {"shape": [2], "dtype": "F32", "data_offsets": [0, 4]}}, b"1234"),
        ({"a": {"shape": [1], "dtype": "F32", "data_offsets": [1, 5]}}, b"12345"),
        ({"a": {"shape": [-1], "dtype": "F32", "data_offsets": [0, 0]}}, b""),
        ({"a": {"shape": [1], "dtype": "F32", "data_offsets": [0, 4]}}, b"12345"),
    ],
)
def test_invalid_layout_rejected(tmp_path, header, payload):
    raw = json.dumps(header).encode()
    source = tmp_path / "bad.safetensors"
    source.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    with pytest.raises(WgtError):
        decompile(str(source), tmp_path / "out.wgt")
    assert not (tmp_path / "out.wgt").exists()


def test_scalar_and_empty_tensor(tmp_path):
    source = checkpoint(tmp_path / "model.safetensors", {"empty": torch.empty(0), "scalar": torch.tensor(3.0)})
    report = decompile(str(source), tmp_path / "out.wgt")
    assert report["parameters"] == 1


def test_python_sequential_numerical_equivalence(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        "from torch import nn\nmodel = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Sigmoid())\n"
    )
    out = tmp_path / "source.wgt"
    report = decompile(str(source), out, references=True)
    assert report["kind"] == "architecture" and report["trained_weights"] is False
    built = compile_file(out, tmp_path / "weights")
    reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2), torch.nn.Sigmoid())
    # Architecture equivalence with identical tensors; Python initialization is not recovered.
    for target, original in zip(reference.parameters(), built.module.parameters()):
        target.data.copy_(original.data)
    data = torch.tensor([[1.0, 2.0, -1.0], [0.0, 1.0, 3.0]])
    torch.testing.assert_close(reference(data), built.module(data), rtol=0, atol=0)


@pytest.mark.parametrize(
    "source",
    [
        'from torch import nn\nmodel = nn.Sequential(nn.GELU(approximate="tanh"))',
        "from torch import nn\nclass Model(nn.Module):\n def forward(self, x): return x + 1",
        'from torch import nn\nmodel = nn.Sequential(nn.Linear(2, 1))\nraise RuntimeError("executed")',
    ],
)
def test_python_unsupported_never_executed(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(source)
    with pytest.raises(WgtError):
        decompile(str(path), tmp_path / "source.wgt")
    assert not (tmp_path / "source.wgt").exists()


@pytest.mark.parametrize("example", ["untrained.wgt", "residual.wgt", "transformer.wgt", "cnn.wgt"])
def test_native_checkpoint_round_trip(tmp_path, example):
    source = Path(__file__).resolve().parents[1] / "examples" / example
    built = compile_file(source, tmp_path / "original")
    recovered = tmp_path / "source.wgt"
    report = decompile(str(built.path), recovered, references=True)
    assert report["kind"] == "checkpoint"
    assert parse_file(recovered).train is None
    rebuilt = compile_file(recovered, tmp_path / "recovered")
    assert built.module.state_dict().keys() == rebuilt.module.state_dict().keys()
    for key, value in built.module.state_dict().items():
        torch.testing.assert_close(value, rebuilt.module.state_dict()[key], rtol=0, atol=0)


def test_cli_failure_no_traceback(tmp_path, capsys):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"bad")
    assert main(["decompile", str(path), "-o", str(tmp_path / "out.wgt")]) == 1
    assert "truncated" in capsys.readouterr().err


def test_remote_inventory_pins_revision_and_checks_shards(tmp_path, monkeypatch):
    import weightlang.decompile as module

    source = checkpoint(tmp_path / "model.safetensors")
    raw = source.read_bytes()
    revision = "a" * 40
    requests = []

    def get(url, limit=0):
        return json.dumps({"sha": revision, "siblings": [{"rfilename": "model.safetensors"}]}).encode()

    def byte_range(url, start, end):
        requests.append(url)
        return raw[start : end + 1], len(raw)

    monkeypatch.setattr(module, "_get", get)
    monkeypatch.setattr(module, "_range", byte_range)
    doc = tensor_inventory("hf://test/model")
    assert doc["revision"] == revision
    assert all(f"/resolve/{revision}/" in url for url in requests)
    assert doc["sources"][0]["payload_verified"] is False


def test_range_refuses_full_download(monkeypatch):
    import weightlang.decompile as module

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            pytest.fail("must reject full response before reading payload")

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(WgtError, match="refusing full download"):
        module._range("https://example.test/model.safetensors", 0, 7)


def test_inventory_tampering_is_rejected(tmp_path):
    from weightlang.decompile import render_inventory

    source = checkpoint(tmp_path / "model.safetensors")
    doc = tensor_inventory(str(source))
    doc["tensors"]["linear.weight"]["data_offsets"] = [0, 1]
    with pytest.raises(WgtError, match="offsets"):
        parse_inventory(render_inventory(doc))


def test_invalid_pytorch_rejected(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"not a safe tensor format")
    with pytest.raises(WgtError, match="restricted PyTorch"):
        decompile(str(source), tmp_path / "source.wgt")


def test_default_structure_groups_layers_without_values_or_locations(tmp_path):
    from weightlang.decompile import parse_architecture

    tensors = {f"layers.{i}.weight": torch.full((2, 3), 123.456) for i in range(12)}
    tensors["layers.12.weight"] = torch.ones(3, 3)
    source = checkpoint(tmp_path / "model.safetensors", tensors)
    out = tmp_path / "structure.wgt"
    report = decompile(str(source), out)
    text = out.read_text()
    doc = parse_architecture(text)
    assert report["kind"] == "architecture" and report["tensors"] == 13
    assert len(doc["modules"]) == 2
    assert doc["modules"][0]["indices"] == [str(i) for i in range(12)]
    for excluded in ("123.456", "data_offsets", "sha256", str(source), "data_start"):
        assert excluded not in text
    assert main(["check", str(out)]) == 0
    assert main(["legacy-build", str(out), "-o", str(tmp_path / "built")]) == 1
    with pytest.raises(WgtError, match="already exists"):
        decompile(str(source), out)


def test_native_default_architecture_needs_no_payload(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/untrained.wgt"
    built = compile_file(source, tmp_path / "original")
    (built.path / "weights.safetensors").unlink()
    out = tmp_path / "architecture.wgt"
    report = decompile(str(built.path), out)
    assert report["trained_weights"] is False
    assert not parse_file(out).imports
    assert "sha256=" not in out.read_text()
    rebuilt = compile_file(out, tmp_path / "rebuilt")
    assert rebuilt.module.state_dict().keys() == built.module.state_dict().keys()


def test_existing_inventory_converts_without_source_files(tmp_path):
    from weightlang.decompile import parse_architecture

    source = checkpoint(tmp_path / "model.safetensors")
    inventory = tmp_path / "inventory.wgt"
    decompile(str(source), inventory, references=True)
    source.unlink()
    out = tmp_path / "architecture.wgt"
    assert main(["decompile", str(inventory), "-o", str(out)]) == 0
    assert parse_architecture(out.read_text())["tensors"]["linear.weight"]["shape"] == [2, 3]
