"""Offline checks for the dataset boundaries; run directly with Python."""

import gzip
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data import FORMAT, SPLIT_SEED, load_data, pixels, split_indices
from prepare_data import fetch, read_idx


class DataTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(tempfile.gettempdir()) / "weightlang"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="mnist-test-", dir=scratch)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_split_is_disjoint_complete_and_repeatable(self):
        train, validation = split_indices()
        self.assertEqual(len(train), 55000)
        self.assertEqual(len(validation), 5000)
        self.assertEqual(len(np.intersect1d(train, validation)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate((train, validation))), np.arange(60000))
        np.testing.assert_array_equal(train, split_indices()[0])

    def test_offline_missing_does_not_use_network(self):
        with patch("prepare_data.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "offline"):
                fetch(self.root, "missing.gz", "unused", offline=True)
            network.assert_not_called()

    def test_corrupt_download_never_replaces_existing_archive(self):
        old = self.root / "data.gz"
        old.write_bytes(b"old data")
        expected = hashlib.md5(b"expected data").hexdigest()
        with patch("prepare_data.urlopen", return_value=io.BytesIO(b"bad download")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch(self.root, "data.gz", expected)
        self.assertEqual(old.read_bytes(), b"old data")
        self.assertEqual(list(self.root.iterdir()), [old])

    def test_valid_cache_never_uses_network(self):
        old = self.root / "data.gz"
        old.write_bytes(b"valid data")
        with patch("prepare_data.urlopen") as network:
            self.assertEqual(fetch(self.root, "data.gz", hashlib.md5(b"valid data").hexdigest()), old)
            network.assert_not_called()

    def test_idx_rejects_truncated_payload(self):
        path = self.root / "images.gz"
        with gzip.open(path, "wb") as stream:
            stream.write(struct.pack(">IIII", 2051, 2, 28, 28) + bytes(784))
        with self.assertRaisesRegex(ValueError, "length"):
            read_idx(path, 2, True)

    def test_preprocessing_preserves_pixel_order_and_scale(self):
        image = np.zeros((28, 28), dtype=np.uint8)
        image[1, 0] = 255
        result = pixels(image)
        self.assertEqual(result.shape, (1, 784))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result[0, 28], 1.0)
        self.assertEqual(result.sum(), 1.0)

    def test_missing_prepared_data_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "prepare_data.py"):
            load_data(self.root)

    def test_modified_prepared_cache_is_rejected_before_loading(self):
        (self.root / "mnist.npz").write_bytes(b"tampered archive")
        (self.root / "dataset.json").write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "split_seed": SPLIT_SEED,
                    "npz_sha256": "0" * 64,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_data(self.root)


if __name__ == "__main__":
    unittest.main()
