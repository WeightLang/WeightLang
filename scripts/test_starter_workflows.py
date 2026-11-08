"""Real CPU acceptance for the generated recipes; run in the tiny-lm uv environment."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TEMPLATES = Path(__file__).resolve().parents[1] / "crates/wgt/templates"


class StarterWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import transformers
            import safetensors.torch
        except ImportError as error:
            raise RuntimeError("Run these acceptance tests with the tiny-lm project's uv-managed Python after uv sync") from error
        cls.torch = torch
        torch.set_num_threads(1)

    def setUp(self):
        scratch = Path(tempfile.gettempdir()) / "weightlang"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="starter-tests-", dir=scratch)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def project(self, kind, replacements=()):
        root = self.root / kind
        root.mkdir()
        text = (TEMPLATES / f"{kind}.toml").read_text(encoding="utf-8")
        text = text.replace("PROJECT_NAME", "starter-test").replace("PROJECT_MODULE", "local/starter_test")
        for old, new in replacements:
            self.assertIn(old, text)
            text = text.replace(old, new)
        (root / "pyproject.toml").write_text(text, encoding="utf-8")
        shutil.copyfile(TEMPLATES / "workflow.py", root / "workflow.py")
        data = ("tiny-lm.txt", "data.txt") if kind == "tiny-lm" else ("mlp.csv", "data.csv")
        shutil.copyfile(TEMPLATES / data[0], root / data[1])
        return root

    def run_workflow(self, root, *args, success=True, stdlib=False):
        command = [sys.executable, *(["-S"] if stdlib else []), str(root / "workflow.py"), *args]
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        result = subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=90, env=environment)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def load_host(self, root):
        spec = importlib.util.spec_from_file_location("starter_host", root / "workflow.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def metadata(self, root):
        return json.loads((root / "weights/model/weightlang.json").read_text(encoding="utf-8"))

    def test_check_validates_without_site_packages_or_artifacts(self):
        for kind in ("tiny-lm", "mlp"):
            root = self.project(kind)
            result = self.run_workflow(root, "check", stdlib=True)
            self.assertIn("no initialization", result.stdout)
            self.assertFalse((root / "weights").exists())
            self.assertFalse((root / ".venv").exists())
        project = self.root / "tiny-lm"
        path = project / "pyproject.toml"
        original = path.read_text(encoding="utf-8")
        for old, new, message in (
            ("heads = 4", "heads = 3", "divisible"),
            ("steps = 100", "steps = 0", "steps"),
            ("dropout = 0.0", "dropout = nan", "finite"),
            ('data = "data.txt"', 'data = "missing.txt"', "data not found"),
            ('output = "weights/model"', 'output = "../outside"', "relative project path"),
            ("layers = 2", "layers = 2\nlayres = 2", "unknown model"),
        ):
            with self.subTest(message=message):
                path.write_text(original.replace(old, new), encoding="utf-8")
                result = self.run_workflow(project, "check", success=False, stdlib=True)
                self.assertIn(message, result.stderr)
                self.assertNotIn("PyTorch is missing", result.stderr)
                self.assertFalse((project / "weights").exists())

    def test_mlp_learns_and_saved_weights_reproduce_evaluation(self):
        from safetensors.torch import load_file

        torch = self.torch
        root = self.project("mlp")
        self.run_workflow(root, "train", "--steps", "80")
        metadata = self.metadata(root)
        self.assertEqual(metadata["train"]["steps"], 80)
        self.assertLess(metadata["metrics"]["final_loss"], metadata["metrics"]["initial_loss"] / 4)
        host = self.load_host(root)
        torch.manual_seed(metadata["train"]["seed"])
        model = host.make_model(torch, metadata["model"])
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        saved = load_file(root / "weights/model/model.safetensors")
        self.assertTrue(any(not torch.equal(initial[name], value) for name, value in saved.items()))
        model.load_state_dict(saved)
        model.eval()
        rows = host.load_data(metadata["train"], metadata["model"])
        batch = torch.tensor(rows[:4])
        with torch.inference_mode():
            loss = torch.nn.functional.mse_loss(model(batch[:, :1]), batch[:, 1:]).item()
            expected = model(torch.tensor([[0.5]])).item()
        self.assertAlmostEqual(loss, metadata["metrics"]["final_loss"], places=6)
        actual = json.loads(self.run_workflow(root, "infer").stdout)[0]
        self.assertAlmostEqual(expected, actual, places=6)
        path = root / "pyproject.toml"
        path.write_text(path.read_text().replace("hidden_size = 16", "hidden_size = 32"))
        self.assertEqual(self.run_workflow(root, "infer").stdout, json.dumps([actual]) + "\n")

    def test_language_model_updates_is_causal_and_generates_locally(self):
        from transformers import GPT2LMHeadModel

        torch = self.torch
        root = self.project("tiny-lm", (("hidden_size = 64", "hidden_size = 32"),
                                        ("layers = 2", "layers = 1"),
                                        ("context_length = 128", "context_length = 48")))
        self.run_workflow(root, "train", "--steps", "6")
        metadata = self.metadata(root)
        self.assertTrue(all(math.isfinite(value) for value in metadata["metrics"].values()))
        self.assertLess(metadata["metrics"]["final_loss"], metadata["metrics"]["initial_loss"])
        model = GPT2LMHeadModel.from_pretrained(root / "weights/model", local_files_only=True).eval()
        host = self.load_host(root)
        torch.manual_seed(metadata["train"]["seed"])
        initial = host.make_model(torch, metadata["model"])
        self.assertFalse(torch.equal(initial.transformer.wte.weight, model.transformer.wte.weight))
        values = torch.tensor(host.load_data(metadata["train"], metadata["model"]))
        batch = values[torch.arange(4)[:, None] + torch.arange(48)]
        with torch.inference_mode():
            loss = model(input_ids=batch, attention_mask=torch.ones_like(batch), labels=batch).loss.item()
            original = torch.arange(12).unsqueeze(0)
            changed = original.clone()
            changed[:, 8:] += 100
            first = model(original, attention_mask=torch.ones_like(original)).logits[:, :8]
            second = model(changed, attention_mask=torch.ones_like(changed)).logits[:, :8]
        self.assertAlmostEqual(loss, metadata["metrics"]["final_loss"], places=5)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
        generated = self.run_workflow(root, "infer", "--max-new-tokens", "8").stdout
        self.assertTrue(generated.startswith("WeightLang"))
        self.assertGreater(len(generated.rstrip("\n")), len("WeightLang"))
        path = root / "pyproject.toml"
        path.write_text(path.read_text().replace("hidden_size = 32", "hidden_size = 64"))
        self.assertEqual(generated, self.run_workflow(root, "infer", "--max-new-tokens", "8").stdout)
        rejected = self.run_workflow(root, "infer", "--max-new-tokens", "48", success=False)
        self.assertIn("exceed", rejected.stderr)

    def test_training_never_overwrites_without_force(self):
        root = self.project("mlp")
        self.run_workflow(root, "train", "--steps", "2")
        checkpoint = root / "weights/model/model.safetensors"
        before = hashlib.sha256(checkpoint.read_bytes()).digest()
        result = self.run_workflow(root, "train", "--steps", "1", success=False)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).digest())
        self.run_workflow(root, "train", "--steps", "3", "--force")
        self.assertEqual(self.metadata(root)["train"]["steps"], 3)
        self.assertNotEqual(before, hashlib.sha256(checkpoint.read_bytes()).digest())
        self.assertEqual([p.name for p in (root / "weights").iterdir()], ["model"])
        (root / "weights/model/weightlang.json").unlink()
        rejected = self.run_workflow(root, "train", "--steps", "1", "--force", success=False)
        self.assertIn("no starter weights", rejected.stderr)
        self.assertTrue(checkpoint.exists())

    def test_inference_requires_saved_weights_and_valid_inputs(self):
        root = self.project("mlp")
        result = self.run_workflow(root, "infer", success=False, stdlib=True)
        self.assertIn("run wgt train first", result.stderr)
        self.assertFalse((root / "weights").exists())
        self.run_workflow(root, "train", "--steps", "1")
        rejected = self.run_workflow(root, "infer", "--inputs", "1", "2", success=False, stdlib=True)
        self.assertIn("1 finite numbers", rejected.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
