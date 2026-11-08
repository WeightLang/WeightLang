"""Editable starter host: standard uv projects, PyTorch training, local weights."""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent
FORMAT = "weightlang.starter.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(config, key):
    value = config.get(key)
    require(isinstance(value, dict), f"missing or invalid [tool.wgt.{key}]")
    return dict(value)


def fields(config, allowed, label):
    require(not (config.keys() - set(allowed.split())), f"unknown {label} setting: {config.keys() - set(allowed.split())}")


def integer(config, name, minimum=1, maximum=None):
    value = config.get(name)
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
            f"{name} must be an integer >= {minimum}" + (f" and <= {maximum}" if maximum is not None else ""))
    return value


def number(config, name, minimum=0, inclusive=False):
    value = config.get(name)
    require(type(value) in (int, float) and math.isfinite(value) and
            (value >= minimum if inclusive else value > minimum),
            f"{name} must be finite and {'>=' if inclusive else '>'} {minimum}")
    return value


def local_path(value, label, weights=False):
    require(isinstance(value, str) and value and "\\" not in value, f"{label} must be a relative project path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            f"{label} must be a relative project path without '..'")
    if weights:
        require(len(path.parts) > 1 and path.parts[0] == "weights", f"{label} must be under weights/")
    result = ROOT / path
    require(result.resolve().is_relative_to(ROOT), f"{label} escapes the project directory")
    return result


def validate_model(config):
    kind = config.get("kind")
    require(kind in ("tiny-lm", "mlp"), "model.kind must be tiny-lm or mlp")
    extra = "heads context_length" if kind == "tiny-lm" else "input_size output_size"
    fields(config, "kind hidden_size layers dropout " + extra, "model")
    width = integer(config, "hidden_size")
    integer(config, "layers")
    require(number(config, "dropout", inclusive=True) < 1, "dropout must be less than 1")
    if kind == "tiny-lm":
        require(width % integer(config, "heads") == 0, "hidden_size must be divisible by heads")
        integer(config, "context_length", 2)
    else:
        integer(config, "input_size")
        integer(config, "output_size")


def validate_train(config):
    fields(config, "data output steps batch_size learning_rate weight_decay seed log_every", "train")
    for key in ("steps", "batch_size", "log_every"):
        integer(config, key)
    integer(config, "seed", 0, 2**32 - 1)
    number(config, "learning_rate")
    config.setdefault("weight_decay", 0.01)
    number(config, "weight_decay", inclusive=True)
    local_path(config.get("output"), "train.output", weights=True)
    return local_path(config.get("data"), "train.data")


def validate_infer(config, model):
    common = "checkpoint seed "
    fields(config, common + ("prompt max_new_tokens temperature top_k" if model["kind"] == "tiny-lm" else "inputs"), "infer")
    integer(config, "seed", 0, 2**32 - 1)
    local_path(config.get("checkpoint"), "infer.checkpoint", weights=True)
    if model["kind"] == "tiny-lm":
        prompt = config.get("prompt")
        require(isinstance(prompt, str) and prompt, "infer.prompt must be nonempty text")
        count = integer(config, "max_new_tokens")
        number(config, "temperature")
        integer(config, "top_k", 0, 256)
        require(len(prompt.encode("utf-8")) + count <= model["context_length"],
                "prompt bytes + max_new_tokens exceed the saved model context_length; shorten the prompt or generation")
    else:
        values = config.get("inputs")
        require(isinstance(values, list) and len(values) == model["input_size"] and
                all(type(x) in (int, float) and math.isfinite(x) for x in values),
                f"infer.inputs must contain {model['input_size']} finite numbers")


def load_data(config, model):
    path = validate_train(config)
    require(path.is_file(), f"training data not found: {path}; set train.data in pyproject.toml")
    if model["kind"] == "tiny-lm":
        data = path.read_text(encoding="utf-8").encode("utf-8")
        require(len(data) >= model["context_length"],
                f"training text needs at least {model['context_length']} UTF-8 bytes; add text or reduce context_length")
        return list(data)
    with path.open(newline="", encoding="utf-8") as source:
        rows = [[float(cell) for cell in row] for row in csv.reader(source) if row]
    width = model["input_size"] + model["output_size"]
    require(rows and all(len(row) == width and all(math.isfinite(x) for x in row) for row in rows),
            f"each CSV row needs {width} finite numbers: input columns, then target columns; no header")
    return rows


def read_metadata(path):
    require(path.is_dir() and (path / "weightlang.json").is_file(),
            f"no starter weights at {path}; run wgt train first")
    metadata = json.loads((path / "weightlang.json").read_text(encoding="utf-8"))
    require(isinstance(metadata, dict) and metadata.get("format") == FORMAT, "not a WeightLang starter model directory")
    validate_model(table(metadata, "model"))
    return metadata


def check_output(path, force):
    for ancestor in (path, *path.parents):
        if ancestor == ROOT:
            break
        require(not ancestor.is_symlink(), "model output must not use symlinks")
    if path.exists():
        require(force, f"{path} already exists; choose a new train.output or use --force")
        read_metadata(path)  # --force only replaces a recognized starter output.


def requested_device(config, override=None):
    runtime = config.get("runtime", {})
    require(isinstance(runtime, dict), "invalid [tool.wgt.runtime]")
    fields(runtime, "device", "runtime")
    requested = override if override is not None else runtime.get("device", "auto")
    require(isinstance(requested, str), "runtime.device must be a device name")
    kind, separator, index = requested.partition(":")
    require(requested in ("auto", "cpu", "cuda", "xpu", "mps") or
            (kind in ("cuda", "xpu") and separator and index.isascii() and index.isdigit()),
            "device must be auto, cpu, cuda, cuda:N, xpu, xpu:N, or mps")
    return requested


def device_for(torch, requested):
    requested = requested_device({}, requested)
    # ROCm uses torch.cuda too. Use PyTorch's availability checks, not driver probes.
    backends = {"cuda": torch.cuda, "xpu": getattr(torch, "xpu", None), "mps": torch.backends.mps}
    selected = requested
    if requested == "auto":
        selected = next((name for name, backend in backends.items()
                         if backend is not None and backend.is_available()), "cpu")
    kind, separator, index = selected.partition(":")
    if kind != "cpu":
        backend = backends[kind]
        require(backend is not None and backend.is_available(),
                f"{kind.upper()} is unavailable; install a compatible PyTorch build or select cpu")
        if separator:
            count = backend.device_count()
            require(int(index) < count, f"{kind} device index {index} is out of range; {count} device(s) available")
            selected = f"{kind}:{int(index)}"
    device = torch.device(selected)
    if device.type == "cpu":
        torch.set_num_threads(1)  # These tiny examples are slower with many CPU threads.
    backend_name = "ROCm" if kind == "cuda" and torch.version.hip else kind.upper()
    print(f"Device: {device} ({backend_name}, {'auto' if requested == 'auto' else 'explicit'})",
          file=sys.stderr, flush=True)
    return device


def make_model(torch, config):
    if config["kind"] == "tiny-lm":
        from transformers import GPT2Config, GPT2LMHeadModel

        return GPT2LMHeadModel(GPT2Config(
            vocab_size=256, n_positions=config["context_length"], n_embd=config["hidden_size"],
            n_layer=config["layers"], n_head=config["heads"], resid_pdrop=config["dropout"],
            embd_pdrop=config["dropout"], attn_pdrop=config["dropout"],
            bos_token_id=None, eos_token_id=None, pad_token_id=0, use_cache=False,
        ))
    layers, width = [], config["input_size"]
    for _ in range(config["layers"]):
        layers.extend((torch.nn.Linear(width, config["hidden_size"]), torch.nn.Tanh(), torch.nn.Dropout(config["dropout"])))
        width = config["hidden_size"]
    return torch.nn.Sequential(*layers, torch.nn.Linear(width, config["output_size"]))


def publish(stage, output, force):
    check_output(output, force)
    backup = None
    if output.exists():
        backup = Path(tempfile.mkdtemp(prefix=".previous-", dir=output.parent))
        backup.rmdir()
        output.rename(backup)
    try:
        stage.rename(output)
    except OSError:
        if backup is not None:
            backup.rename(output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def train(torch, model_config, settings, data, device, force):
    output = local_path(settings["output"], "train.output", weights=True)
    torch.manual_seed(settings["seed"])
    model = make_model(torch, model_config).to(device)
    is_lm = model_config["kind"] == "tiny-lm"
    values = torch.tensor(data, dtype=torch.long if is_lm else torch.float32, device=device)
    count = len(values) - model_config["context_length"] + 1 if is_lm else len(values)

    def loss_at(indices):
        if is_lm:
            batch = values[indices[:, None] + torch.arange(model_config["context_length"], device=device)]
            # Every byte is data, including 0; Transformers handles the causal label shift.
            return model(input_ids=batch, attention_mask=torch.ones_like(batch), labels=batch).loss
        batch = values[indices]
        split = model_config["input_size"]
        return torch.nn.functional.mse_loss(model(batch[:, :split]), batch[:, split:])

    evaluation = torch.arange(min(settings["batch_size"], count), device=device)
    model.eval()
    with torch.no_grad():
        initial = float(loss_at(evaluation))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    print(f"Training {model_config['kind']}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}", flush=True)
    model.train()
    for step in range(1, settings["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_at(torch.randint(count, (settings["batch_size"],), device=device))
        require(bool(torch.isfinite(loss)), "non-finite training loss; reduce learning_rate or check the data")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % settings["log_every"] == 0 or step == settings["steps"]:
            print(f"step {step}/{settings['steps']}  loss {float(loss.detach()):.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        final = float(loss_at(evaluation))
    require(math.isfinite(initial) and math.isfinite(final), "non-finite evaluation loss; model was not saved")
    model.cpu()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".weightlang-", dir=output.parent))
    try:
        if is_lm:
            model.save_pretrained(stage, safe_serialization=True)
        else:
            from safetensors.torch import save_file

            save_file(model.state_dict(), stage / "model.safetensors")
        metadata = {"format": FORMAT, "model": model_config, "train": settings, "runtime": {"device": str(device)},
                    "metrics": {"initial_loss": initial, "final_loss": final}}
        (stage / "weightlang.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        shutil.copyfile(ROOT / "pyproject.toml", stage / "pyproject.toml")
        publish(stage, output, force)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Saved {settings['output']} (fixed-batch loss {initial:.6f} -> {final:.6f}). Next: wgt infer")


def infer(torch, settings, metadata, device):
    checkpoint = local_path(settings["checkpoint"], "infer.checkpoint", weights=True)
    model_config = metadata["model"]
    torch.manual_seed(settings["seed"])
    if model_config["kind"] == "tiny-lm":
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        prompt = torch.tensor([list(settings["prompt"].encode("utf-8"))], device=device)
        with torch.inference_mode():
            result = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                    max_new_tokens=settings["max_new_tokens"], do_sample=True,
                                    temperature=settings["temperature"], top_k=settings["top_k"],
                                    pad_token_id=0, use_cache=True)
        print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))
    else:
        from safetensors.torch import load_file

        model = make_model(torch, model_config)
        model.load_state_dict(load_file(checkpoint / "model.safetensors"))
        model.to(device).eval()
        with torch.inference_mode():
            result = model(torch.tensor([settings["inputs"]], dtype=torch.float32, device=device))
        print(json.dumps(result[0].cpu().tolist(), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("check", help="validate settings and local data without importing PyTorch")
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int)
    training.add_argument("--force", action="store_true")
    training.add_argument("--device")
    inference = commands.add_parser("infer")
    inference.add_argument("--device")
    inference.add_argument("--prompt")
    inference.add_argument("--max-new-tokens", type=int)
    inference.add_argument("--temperature", type=float)
    inference.add_argument("--top-k", type=int)
    inference.add_argument("--inputs", type=float, nargs="+")
    args = parser.parse_args()
    with (ROOT / "pyproject.toml").open("rb") as source:
        config = tomllib.load(source).get("tool", {}).get("wgt", {})
    requested = requested_device(config, getattr(args, "device", None))
    if args.action == "infer":
        settings = table(config, "infer")
        settings.setdefault("seed", 42)
        settings.update({key: value for key, value in vars(args).items()
                         if value is not None and key not in ("action", "device")})
        metadata = read_metadata(local_path(settings.get("checkpoint"), "infer.checkpoint", weights=True))
        validate_infer(settings, metadata["model"])
    else:
        model_config = table(config, "model")
        validate_model(model_config)
        settings = table(config, "train")
        if getattr(args, "steps", None) is not None:
            settings["steps"] = args.steps
        data = load_data(settings, model_config)
        if args.action == "check":
            inference_config = table(config, "infer")
            inference_config.setdefault("seed", 42)
            validate_infer(inference_config, model_config)
            print("ok: starter configuration and data; no initialization, training, or downloads")
            return
        check_output(local_path(settings["output"], "train.output", weights=True), args.force)
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is missing; run wgt sync, then retry") from error
    device = device_for(torch, requested)
    if args.action == "train":
        train(torch, model_config, settings, data, device, args.force)
    else:
        infer(torch, settings, metadata, device)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
