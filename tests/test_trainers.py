from pathlib import Path

import pytest
import torch

from weightlang import WgtError, load_artifact, parse
from weightlang.cli import main
from weightlang.compiler import compile_file
from weightlang.trainers import load_trainers

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/custom_trainer/model.wgt").read_text()
PLUGIN = ROOT / "examples/custom_trainer/search.py"


def build(tmp_path, trainer=None, text=SOURCE, name="out"):
    source = tmp_path / "model.wgt"
    source.write_text(text)
    return compile_file(source, tmp_path / name, trainers=trainer)


def forbidden(*args, **kwargs):
    raise AssertionError("backpropagation or optimizer was invoked")


def test_forward_only_plugin_build_reload_and_reproduce(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    trainers = load_trainers([f"search={PLUGIN}:train"])
    before = torch.random.get_rng_state()
    result = build(tmp_path, trainers)
    assert torch.equal(before, torch.random.get_rng_state())
    record = result.manifest["training"]
    assert record["final_loss"] < record["initial_loss"] * 0.01
    assert "optimizer" not in record and "weight_decay" not in record
    assert record["trainer"]["sha256"] == trainers["search"].sha256
    assert record["trainer"]["metrics"]["accepted_steps"] > 0
    again = build(tmp_path, trainers, name="again")
    assert again.manifest["weights"]["sha256"] == result.manifest["weights"]["sha256"]
    loaded = load_artifact(result.path)
    assert torch.equal(result.module(torch.tensor([[1.0]])), loaded.predict([[1.0]]))
    assert all(p.grad is None for p in result.module.parameters())


def test_python_callback_and_context_batches(tmp_path):
    def fit(context):
        assert context.config.optimizer is None
        assert len(list(context.batches())) == 4  # two batches per epoch
        with torch.no_grad():
            context.module.fit.weight.fill_(2)
            context.module.fit.bias.fill_(1)
        return {"updates": 1}

    result = build(tmp_path, {"search": fit}, SOURCE.replace("epochs=400", "epochs=2 batch=2"))
    assert result.manifest["training"]["final_loss"] == 0


def test_cli_explicit_plugin_and_check_without_execution(tmp_path, capsys):
    source = tmp_path / "model.wgt"
    source.write_text(SOURCE)
    assert main(["check", str(source)]) == 0
    assert (
        main(
            ["build", "--no-store", str(source), "-o", str(tmp_path / "weights"), "--trainer", f"search={PLUGIN}:train"]
        )
        == 0
    )
    assert "built" in capsys.readouterr().out


def test_missing_plugin_writes_no_artifact(tmp_path):
    with pytest.raises(WgtError, match="not registered"):
        build(tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda s: s.replace("wgt 0.3", "wgt 0.2"), "requires 'wgt 0.3'"),
        (lambda s: s.replace("trainer=search", "trainer=search optimizer=sgd"), "unknown train option"),
        (lambda s: s.replace("trainer=search", "trainer=search weight_decay=0.1"), "unknown train option"),
        (lambda s: s.replace("trainer=search", 'trainer="file.py:run"'), "identifier"),
        (lambda s: s.replace("lr=0.1", "lr=nan"), "positive"),
    ],
)
def test_invalid_selection(mutation, message):
    with pytest.raises(WgtError, match=message):
        parse(mutation(SOURCE))


@pytest.mark.parametrize(
    "behavior, message",
    [
        ("frozen", "frozen parameters"),
        ("nan", "non-finite"),
        ("policy", "freeze policy"),
        ("shape", "shapes or dtypes"),
        ("replace", "model structure"),
        ("metrics", "finite numbers"),
        ("raise", "deliberate failure"),
    ],
)
def test_bad_plugin_fails_before_publication(tmp_path, behavior, message):
    def bad(context):
        with torch.no_grad():
            if behavior == "frozen":
                context.module.fit.bias.add_(1)
            elif behavior == "nan":
                context.module.fit.weight.fill_(float("nan"))
            elif behavior == "policy":
                context.module.fit.bias.requires_grad_(True)
            elif behavior == "shape":
                context.module.fit.weight.data = torch.zeros(3, 3)
            elif behavior == "replace":
                context.module.fit = torch.nn.Linear(1, 1)
            elif behavior == "metrics":
                return {"score": float("nan")}
            else:
                raise RuntimeError("deliberate failure")

    text = SOURCE.replace("init seed=42", "init seed=42\nfreeze fit.bias")
    with pytest.raises(WgtError, match=message) as exc:
        build(tmp_path, {"search": bad}, text)
    assert exc.value.line == len(text.splitlines())
    assert not (tmp_path / "out").exists()


def test_freeze_and_validation(tmp_path):
    text = SOURCE.replace("init seed=42", "init seed=42\nfreeze fit.bias")
    text = text.replace("train data=", "data heldout:\n  [3] -> [7]\ntrain validate=heldout data=")
    result = build(tmp_path, load_trainers([f"search={PLUGIN}:train"]), text)
    initial = build(tmp_path, text=text[: text.index("freeze")] + "\n", name="initial")
    assert torch.equal(result.module.fit.bias, initial.module.fit.bias)
    assert "validation" in result.manifest["training"]


@pytest.mark.parametrize("binding", ["oops", "search=missing.py:train", "backprop=x.py:train"])
def test_invalid_bindings(binding):
    with pytest.raises(WgtError):
        load_trainers([binding])


def test_explicit_default_matches_legacy_training(tmp_path):
    text = SOURCE.replace("trainer=search", "optimizer=sgd").replace("epochs=400", "epochs=2")
    implicit = build(tmp_path, text=text)
    explicit = build(tmp_path, text=text.replace("optimizer=sgd", "trainer=backprop optimizer=sgd"), name="explicit")
    assert implicit.manifest["weights"]["sha256"] == explicit.manifest["weights"]["sha256"]
    assert implicit.manifest["training"] == explicit.manifest["training"]


def test_failed_trainer_preserves_existing_artifact(tmp_path):
    result = build(tmp_path, load_trainers([f"search={PLUGIN}:train"]))
    original = (result.path / "weights.safetensors").read_bytes()

    def fail(context):
        raise RuntimeError("failed update")

    with pytest.raises(WgtError, match="failed update"):
        compile_file(tmp_path / "model.wgt", tmp_path / "out", force=True, trainers={"search": fail})
    assert (result.path / "weights.safetensors").read_bytes() == original
