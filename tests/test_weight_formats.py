import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from weightlang.cli import main
from weightlang.decompile import decompile, parse_inventory, render_inventory, tensor_inventory
from weightlang.syntax import WgtError
from weightlang.weight_metadata import read_metadata, write_metadata


def convert(path, tmp_path):
    output = tmp_path / (path.name + ".wgt")
    report = decompile(str(path), output, references=True)
    doc = parse_inventory(output.read_text())
    assert doc["version"] == 2 and not report["executable"]
    assert main(["check", str(output)]) == 0
    assert main(["legacy-build", str(output), "-o", str(tmp_path / "not-built")]) == 1
    assert not (tmp_path / "not-built").exists()
    return doc


@pytest.mark.parametrize("extension", ["pt", "pth", "bin", "ckpt"])
@pytest.mark.parametrize("wrapper", [None, "state_dict", "model_state_dict", "model"])
def test_pytorch_state_and_metadata(tmp_path, extension, wrapper):
    path = tmp_path / f"model.{extension}"
    state = {"linear.weight": torch.arange(6).reshape(2, 3).t(), "bias": torch.ones(2, dtype=torch.bfloat16)}
    checkpoint = {wrapper: state, "epoch": 4, "config": {"author": "测试", "lr": 0.01}} if wrapper else state
    torch.save(checkpoint, path)
    doc = convert(path, tmp_path)
    tensor = doc["tensors"]["linear.weight"]
    assert tensor["shape"] == [3, 2] and tensor["dtype"] == "torch.int64"
    assert tensor["locator"]["stride"] == [1, 3]
    assert tensor["locator"]["path"] == ([wrapper, "linear.weight"] if wrapper else ["linear.weight"])
    if wrapper:
        assert doc["sources"][0]["metadata"]["epoch"] == 4
        assert doc["sources"][0]["metadata"]["config"]["author"] == "测试"


def test_pytorch_legacy_and_shared_views(tmp_path):
    path = tmp_path / "legacy.pt"
    tensor = torch.arange(20)
    torch.save({"a": tensor[1:7], "b": tensor[2:8]}, path, _use_new_zipfile_serialization=False)
    doc = convert(path, tmp_path)
    assert doc["tensors"]["a"]["locator"]["storage_offset"] == 1
    assert doc["tensors"]["b"]["locator"]["storage_offset"] == 2


def _unpickle_marker(path):
    Path(path).touch()
    return {}


class Untrusted:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return _unpickle_marker, (str(self.marker),)


def test_restricted_load_does_not_execute_objects(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "unsafe.pt"
    torch.save(Untrusted(marker), path)
    with pytest.raises(WgtError, match="restricted PyTorch"):
        decompile(str(path), tmp_path / "out.wgt")
    assert not marker.exists() and not (tmp_path / "out.wgt").exists()


def test_torchscript_rejected_before_load(tmp_path, monkeypatch):
    path = tmp_path / "script.pt"
    torch.jit.trace(torch.nn.Linear(2, 1), torch.zeros(1, 2)).save(str(path))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("must not call torch.load on TorchScript"))
    with pytest.raises(WgtError, match="TorchScript"):
        decompile(str(path), tmp_path / "out.wgt")


def test_pytorch_omitted_optimizer_metadata_is_explicit(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"a": torch.ones(2)}, "optimizer": {"moment": torch.ones(2)}, "epoch": 9}, path)
    doc = convert(path, tmp_path)
    assert doc["sources"][0]["metadata_omitted"] == ["optimizer"]
    assert doc["sources"][0]["metadata"]["epoch"] == 9


def test_pytorch_shards_and_mismatch(tmp_path):
    torch.save({"a": torch.ones(2)}, tmp_path / "one.bin")
    torch.save({"b": torch.ones(3)}, tmp_path / "two.bin")
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "one.bin", "b": "two.bin"}}))
    doc = convert(index, tmp_path)
    assert len(doc["sources"]) == 2 and doc["tensors"]["b"]["source"] == 1
    assert len(tensor_inventory(str(tmp_path))["tensors"]) == 2
    index.write_text(json.dumps({"weight_map": {"wrong": "one.bin"}}))
    with pytest.raises(WgtError, match="index does not match"):
        tensor_inventory(str(index))


def test_gguf_f32_quantized_and_metadata(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(path, "test")
    writer.add_name("Tiny model")
    writer.add_string("general.license", "MIT")
    writer.add_array("test.labels", ["a", "b"])
    values = np.arange(64, dtype=np.float32).reshape(2, 32)
    quantized = gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("plain", values)
    writer.add_tensor("quant", quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    doc = convert(path, tmp_path)
    assert doc["sources"][0]["metadata"]["general.license"] == "MIT"
    assert doc["sources"][0]["metadata"]["test.labels"] == ["a", "b"]
    assert doc["tensors"]["quant"]["dtype"] == "gguf.Q8_0"
    assert doc["tensors"]["quant"]["shape"] == [2, 32]
    for name, expected in [("plain", values.tobytes()), ("quant", quantized.tobytes())]:
        loc = doc["tensors"][name]["locator"]
        with path.open("rb") as stream:
            stream.seek(loc["offset"])
            assert stream.read(loc["length"]) == expected


def make_onnx(path, external=False):
    onnx = pytest.importorskip("onnx")
    h = onnx.helper
    w = onnx.numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2), "weight")
    graph = h.make_graph(
        [h.make_node("MatMul", ["x", "weight"], ["y"])],
        "linear",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])],
        [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])],
        [w],
    )
    model = h.make_model(graph, producer_name="test")
    h.set_model_props(model, {"author": "测试", "license": "MIT"})
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    return model


@pytest.mark.parametrize("external", [False, True])
def test_onnx_embedded_and_external(tmp_path, external):
    path = tmp_path / "model.onnx"
    make_onnx(path, external)
    doc = convert(path, tmp_path)
    assert doc["sources"][0]["metadata"]["metadata_props"]["author"] == "测试"
    assert doc["config"]["graph_summary"]["nodes"][0]["op_type"] == "MatMul"
    tensor = doc["tensors"]["weight"]
    assert tensor["shape"] == [3, 2] and tensor["dtype"] == "onnx.FLOAT"
    if external:
        assert tensor["source"] == 1
        loc = tensor["locator"]
        with Path(doc["sources"][1]["uri"]).open("rb") as stream:
            stream.seek(loc["offset"])
            assert stream.read(loc["length"]) == np.arange(6, dtype=np.float32).tobytes()
    else:
        assert tensor["locator"]["path"] == ["graph", "initializer", 0]


def test_onnx_missing_external_payload(tmp_path):
    path = tmp_path / "model.onnx"
    make_onnx(path, True)
    (tmp_path / "weights.data").unlink()
    with pytest.raises(WgtError, match="cannot read"):
        decompile(str(path), tmp_path / "out.wgt")


@pytest.mark.parametrize("extension", ["npy", "npz"])
def test_numpy_arrays(tmp_path, extension):
    path = tmp_path / f"weights.{extension}"
    data = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    if extension == "npy":
        np.save(path, data)
    else:
        np.savez_compressed(path, weights=data, scalar=np.array(42))
    doc = convert(path, tmp_path)
    assert doc["tensors"]["weights"]["shape"] == [3, 4]
    assert doc["tensors"]["weights"]["locator"]["order"] == "F"
    if extension == "npz":
        assert doc["tensors"]["scalar"]["shape"] == []


def test_numpy_object_pickle_rejected(tmp_path):
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"python": "object"}], dtype=object))
    with pytest.raises(WgtError, match="object"):
        decompile(str(path), tmp_path / "out.wgt")


def test_missing_optional_dependency(tmp_path, monkeypatch):
    import weightlang.weight_formats as formats

    monkeypatch.setattr(formats.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError()))
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    with pytest.raises(WgtError, match=r"weightlang\[formats\]"):
        decompile(str(path), tmp_path / "out.wgt")


def test_metadata_write_preserves_payload_and_original(tmp_path):
    path, output = tmp_path / "source.safetensors", tmp_path / "annotated.safetensors"
    data = {"weight": torch.arange(6).reshape(2, 3), "scalar": torch.tensor(0.5)}
    save_file(data, str(path), metadata={"author": "original", "license": "MIT"})
    original = path.read_bytes()
    result = write_metadata(path, output, {"author": "测试", "config": '{"hidden_size":3}'})
    assert path.read_bytes() == original and result["source_modified"] is False
    with safe_open(output, framework="pt") as reader:
        assert reader.metadata() == {"author": "测试", "license": "MIT", "config": '{"hidden_size":3}'}
        for key, tensor in data.items():
            torch.testing.assert_close(reader.get_tensor(key), tensor, rtol=0, atol=0)
    doc = tensor_inventory(str(output))
    assert doc["sources"][0]["metadata"]["author"] == "测试"
    parse_inventory(render_inventory(doc))
    assert read_metadata(str(output))["verified"] is False
    with pytest.raises(WgtError, match="already exists"):
        write_metadata(path, output, {"author": "overwrite"})
    assert main(["metadata", str(output)]) == 0


def test_metadata_cli_roundtrip_and_validation(tmp_path, capsys):
    path, output = tmp_path / "source.safetensors", tmp_path / "annotated.safetensors"
    save_file({"a": torch.ones(1)}, str(path))
    assert main(["metadata", str(path), "--set", "author=me", "--set", "license=MIT", "-o", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["metadata"]["author"] == "me"
    assert main(["metadata", str(path), "--set", "author=me"]) == 1
    assert main(["metadata", str(path), "--set", "invalid", "-o", str(tmp_path / "bad.safetensors")]) == 1
    assert main(["metadata", str(tmp_path / "missing.safetensors")]) == 1
    assert not (tmp_path / "bad.safetensors").exists()


def test_v2_tamper_rejected(tmp_path):
    path = tmp_path / "model.npy"
    np.save(path, np.ones(2))
    doc = tensor_inventory(str(path))
    doc["tensors"]["model"]["locator"]["length"] = 10**9
    with pytest.raises(WgtError, match="byte range"):
        parse_inventory(render_inventory(doc))
    doc["version"] = 1
    with pytest.raises(WgtError):
        parse_inventory(render_inventory(doc))


@pytest.mark.parametrize("mode", ["traversal", "short_range", "short_raw"])
def test_onnx_invalid_payload_rejected(tmp_path, mode):
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "model.onnx"
    make_onnx(path, mode != "short_raw")
    model = onnx.load_model(str(path), load_external_data=False)
    tensor = model.graph.initializer[0]
    if mode == "short_raw":
        tensor.raw_data = b"x"
    else:
        for prop in tensor.external_data:
            if prop.key == ("location" if mode == "traversal" else "length"):
                prop.value = "../outside.data" if mode == "traversal" else "1"
    path.write_bytes(model.SerializeToString())
    with pytest.raises(WgtError):
        decompile(str(path), tmp_path / "out.wgt")
    assert not (tmp_path / "out.wgt").exists()


def test_metadata_failed_copy_leaves_no_partial_output(tmp_path, monkeypatch):
    import weightlang.weight_metadata as module

    path, output = tmp_path / "source.safetensors", tmp_path / "new.safetensors"
    save_file({"a": torch.ones(2)}, str(path))
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(module.shutil, "copyfileobj", fail)
    with pytest.raises(WgtError, match="disk full"):
        write_metadata(path, output, {"author": "me"})
    assert not output.exists() and path.read_bytes() == original


def test_pytorch_state_metadata_keeps_original_namespace(tmp_path):
    path = tmp_path / "model.pt"
    state = torch.nn.Linear(2, 1).state_dict()
    torch.save({"state_dict": state, "state_dict_metadata": "user field"}, path)
    doc = convert(path, tmp_path)
    assert doc["sources"][0]["metadata"]["state_dict_metadata"] == "user field"
    assert doc["sources"][0]["state_dict_metadata"] == {"": {"version": 1}}
