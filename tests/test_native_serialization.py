"""The Rust encoder must preserve WeightLang's content-addressed file format."""

import json
import struct

import pytest
import torch
from safetensors.torch import load_file

from weightlang.artifact import write_safetensors
from weightlang.syntax import WgtError


def test_canonical_bytes_views_aliases_and_special_values(tmp_path):
    # Explicit bit patterns include negative zero, NaN payload and infinity.
    bits = torch.tensor([0, -2147483648, 0x7FC00001, 0x7F800000], dtype=torch.int32)
    storage = torch.arange(20, dtype=torch.float32)
    tensors = {"特殊": bits.view(torch.float32), "offset": storage[3:7],
               "alias": storage[3:7], "transpose": torch.arange(6, dtype=torch.float32).reshape(2, 3).T,
               "scalar": torch.tensor(1.25), "empty": torch.empty(0, 3)}
    metadata = {"z": "中文", "a": "first"}
    header = {"__metadata__": {"a": "first", "z": "中文"}}
    payload = bytearray()
    for name, tensor in sorted(tensors.items()):
        raw = bytes(tensor.contiguous().clone().untyped_storage())
        header[name] = {"dtype": "F32", "shape": list(tensor.shape),
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    expected = struct.pack("<Q", len(encoded)) + encoded + payload
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    write_safetensors(a, tensors, metadata)
    write_safetensors(b, dict(reversed(list(tensors.items()))), dict(reversed(list(metadata.items()))))
    assert a.read_bytes() == b.read_bytes() == expected
    loaded = load_file(a)
    for name, tensor in tensors.items():
        assert torch.equal(loaded[name].reshape(-1).view(torch.int32), tensor.contiguous().reshape(-1).view(torch.int32))


def test_invalid_dtype_does_not_touch_destination(tmp_path):
    destination = tmp_path / "existing"
    destination.write_bytes(b"keep")
    with pytest.raises(WgtError, match="float32"):
        write_safetensors(destination, {"a": torch.ones(3), "z": torch.ones(3, dtype=torch.int64)}, {})
    assert destination.read_bytes() == b"keep"
