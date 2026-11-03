"""Read-only architecture analysis and bounded, in-memory execution diagnostics."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

from .diagnostics import WgtError, label, render, warning
from .syntax import apply_freezing

ITERATIONS = ContextVar("weightlang_iterations", default=())


@contextmanager
def iteration(index):
    token = ITERATIONS.set((*ITERATIONS.get(), index))
    try:
        yield
    finally:
        ITERATIONS.reset(token)


def layer_map(layers, prefix=""):
    result = {}
    for layer in layers:
        name = f"{prefix}.{layer.name}" if prefix else layer.name
        result[name] = layer
        result.update(layer_map(layer.children, name))
    return result


def lint(layers, prefix="", control_outputs=()):
    """Reachability follows the last output, not source-order adjacency alone."""
    if not layers:
        return []
    by_name = {layer.name: layer for layer in layers}
    deps, previous = {}, "input"
    for layer in layers:
        deps[layer.name] = layer.inputs or [previous]
        previous = layer.name
    reachable = set()
    pending = [layers[-1].name, *control_outputs]
    while pending:
        name = pending.pop()
        if name in reachable or name not in by_name:
            continue
        reachable.add(name)
        pending.extend(deps[name])
    result = []
    for layer in layers:
        name = f"{prefix}.{layer.name}" if prefix else layer.name
        if layer.name not in reachable:
            result.append(
                warning(
                    "W0201",
                    f"`{name}` does not contribute to the output",
                    layer,
                    "remove the unused branch or connect it to the result",
                )
            )
        else:
            result.extend(lint(layer.children, name, (layer.args["until"],) if layer.op in {"recur", "ponder"} and "until" in layer.args else ()))
    return result


def stats(tensor):
    data = tensor.detach()
    finite = torch.isfinite(data)
    values = data[finite].double()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "finite": bool(finite.all()),
        "min": float(values.min()) if values.numel() else None,
        "max": float(values.max()) if values.numel() else None,
        "mean": float(values.mean()) if values.numel() else None,
        "norm": float(torch.linalg.vector_norm(values.double())) if values.numel() else 0.0,
    }


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in tensors(item)]
    return []


class Observer:
    def __init__(self, program, model, *, watch=None, times=None, limit=2000):
        self.layers = layer_map(program.layers)
        if watch and not any(name == watch or name.startswith(watch + ".") for name in self.layers):
            raise WgtError(f"unknown watched layer {watch!r}", code="E0201")
        self.watch, self.times, self.limit = watch, times, limit
        self.records, self.handles, self.active = [], [], []
        self.calls, self.truncated = {}, False
        self.observations = {}
        self.step = None
        for name, layer in self.layers.items():
            child = model.get_submodule(name)
            self.handles.append(child.register_forward_pre_hook(self.before(name)))
            self.handles.append(child.register_forward_hook(self.after(name, layer)))

    def before(self, name):
        def hook(module, inputs):
            self.active.append((name, time.perf_counter(), ITERATIONS.get()))
            for value in tensors(inputs):
                if not torch.isfinite(value).all():
                    self.fail(name, "input contains NaN or infinity", "E0401")

        return hook

    def fail(self, name, message, code="E0401", indices=None):
        layer = self.layers[name]
        steps = indices if indices is not None else ITERATIONS.get() or (self.active[-1][2] if self.active else ())
        notes = [f"layer: {name}"]
        if steps:
            notes.append(f"loop/time indices (zero-based): {list(steps)}")
        if self.step is not None:
            notes.append(f"optimizer step: {self.step}")
        raise WgtError(
            message,
            layer.line,
            layer.path,
            code=code,
            labels=[label(layer.path, layer.line, name), *layer.frames],
            notes=notes,
            help="inspect this layer's inputs and weights; for unstable training, try a lower learning rate",
        )

    def after(self, name, layer):
        def hook(module, inputs, output):
            active_name, start, indices = self.active[-1]
            outputs = tensors(output)
            for value in outputs:
                if not torch.isfinite(value).all():
                    self.fail(name, "first non-finite layer output (NaN or infinity)")
            self.active.pop()
            assert active_name == name
            elapsed = (time.perf_counter() - start) * 1000
            count = self.calls.get(name, 0)
            self.calls[name] = count + 1
            selected = not self.watch or name == self.watch or name.startswith(self.watch + ".")
            selected &= self.times is None or bool(indices and indices[-1] in self.times)
            record = None
            if selected and len(self.records) < self.limit:
                record = {
                    "layer": name,
                    "op": layer.op,
                    "call": count,
                    "iterations": list(indices),
                    "step": self.step,
                    "elapsed_ms": round(elapsed, 4),
                    "source": label(layer.path, layer.line),
                    "inputs": [stats(t) for t in tensors(inputs)],
                    "outputs": [stats(t) for t in outputs],
                }
                self.records.append(record)
            elif selected:
                self.truncated = True
            for index, value in enumerate(outputs):
                if value.requires_grad:

                    def gradient_hook(gradient, output_index=index, event=record):
                        if not torch.isfinite(gradient).all():
                            self.fail(name, "non-finite activation gradient", "E0402", indices=indices)
                        if event is not None:
                            event["outputs"][output_index]["gradient"] = stats(gradient)
                        return gradient

                    value.register_hook(gradient_hook)
            if layer.op == "sigmoid" and outputs:
                value = outputs[0].detach()
                saturated = float(((value < 0.01) | (value > 0.99)).float().mean())
                self.observations[name] = max(saturated, self.observations.get(name, 0))

        return hook

    @contextmanager
    def running(self):
        try:
            yield self
        except (RuntimeError, IndexError, ValueError) as exc:
            if self.active:
                self.fail(self.active[-1][0], f"execution failed: {exc}", "E0400")
            raise WgtError(f"execution failed: {exc}", code="E0400") from exc
        finally:
            for handle in self.handles:
                handle.remove()

    def warnings(self):
        return [
            warning(
                "W0403",
                f"`{name}` sigmoid saturation reached {fraction:.0%}",
                self.layers[name],
                "this may be intentional; inspect gate inputs and learning rate",
            )
            for name, fraction in self.observations.items()
            if fraction >= 0.95
        ]


def prepare(program, max_parameters, *, weights=None, origin=None, training=True):
    from .artifact import build_module, load_artifact
    from .compiler import _apply_imports, _layer_dicts

    layers = _layer_dicts(program)
    with torch.device("meta"):
        preview = build_module(layers)
        count = sum(p.numel() for p in preview.parameters())
    if count > max_parameters:
        raise WgtError(
            f"debug allocation limit: {count} parameters exceeds {max_parameters}",
            code="E0405",
            help="reduce the model or explicitly increase --max-parameters",
        )
    torch.random.default_generator.manual_seed(program.seed)
    if weights is not None:
        root = Path(weights).resolve()
        try:
            manifest = json.loads((root / "manifest.json").read_text())
            compatible = manifest["layers"] == layers and all(
                manifest["interface"][role]["shape"] == list(port.shape)
                and manifest["interface"][role]["dtype"] == port.dtype
                for role, port in (("input", program.input), ("output", program.output))
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WgtError(f"cannot read debug weights: {exc}", path=str(root), code="E0404") from None
        # Reject a different graph before the artifact loader allocates its parameters.
        if not compatible:
            raise WgtError(
                "debug weights require identical architecture and input/output contracts",
                path=str(root), code="E0404",
            )
        artifact = load_artifact(root, warn_untrained=False)
        model = artifact.module
        apply_freezing(program, model, require_trainable=training)
        if origin is not None:
            origin.update(
                path=str(root), sha256=artifact.manifest["weights"]["sha256"],
                untrained_tensors=artifact.untrained_tensors,
            )
        return model, count
    model = build_module(layers)
    base = Path(program.path).parent
    if program.inherit:
        inherited = load_artifact((base / program.inherit.source).resolve(), warn_untrained=False)
        if inherited.manifest["layers"] != layers:
            raise WgtError("inherit requires identical architecture", program.inherit.line, program.path)
    if getattr(program, "derivation", None):
        from .derivation import apply

        apply(program, model)
    _apply_imports(program, model, layers, base)
    apply_freezing(program, model, require_trainable=training)
    return model, count


def sample_input(program, sample):
    if sample:
        try:
            raw = json.loads(Path(sample).read_text())
            if isinstance(raw, dict):
                raw = raw["inputs"]
            value = torch.tensor(raw)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise WgtError(f"cannot read sample: {exc}", code="E0404", path=str(sample)) from None
        kind = "file"
    elif program.datasets:
        ds = program.datasets.get(program.train.data) if program.train else next(iter(program.datasets.values()))
        if ds is None:
            raise WgtError("trace dataset is unavailable; provide --sample", code="E0404")
        if ds.python or ds.entrypoint:
            raise WgtError("Python datasets require an explicit --sample for trace/debug", code="E0404")
        if ds.file and not ds.inputs:
            from .syntax import _load_csv

            _load_csv(ds, math.prod(program.input.shape), program.path)
        try:
            value = torch.tensor(ds.inputs[:1]).reshape(1, *program.input.shape)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise WgtError(f"invalid trace dataset sample: {exc}", code="E0404") from None
        kind = "first dataset sample"
    else:
        value = torch.zeros(1, *program.input.shape)
        kind = "synthetic zeros (not a quality test)"
    shape = program.input.shape
    if tuple(value.shape) == shape:
        value = value.unsqueeze(0)
    if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape or value.shape[0] == 0:
        raise WgtError(
            f"sample shape {list(value.shape)} does not match [batch, {', '.join(map(str, shape))}]",
            program.input.line,
            program.path,
            code="E0404",
        )
    if not torch.isfinite(value).all():
        raise WgtError("sample contains NaN or infinity", program.input.line, program.path, code="E0404")
    if program.input.dtype == "i64":
        if (value < 0).any() or (value.is_floating_point() and ((value >= 2**63) | (value != value.trunc())).any()):
            raise WgtError("i64 samples must contain non-negative integers below 2**63", code="E0404")
        value = value.long()
    else:
        value = value.float()
    return value, kind


def execute(program, *, mode, sample=None, steps=10, watch=None, times=None, max_parameters=10_000_000, weights=None):
    from .compiler import _dataset_tensors, _loss_fn, _targets

    if not 1 <= steps <= 1000:
        raise WgtError("debug steps must be in [1, 1000]", code="E0404")
    with torch.random.fork_rng(devices=[]):
        origin = {}
        model, count = prepare(program, max_parameters, weights=weights, origin=origin, training=mode != "trace")
        observer = Observer(program, model, watch=watch, times=times)
        report = {
            "format": "weightlang.debug",
            "format_version": 1,
            "mode": mode,
            "model": program.model,
            "device": "cpu",
            "parameters": count,
            "initialization": "verified artifact weights" if weights is not None else "source seed and declared weight imports",
            "weight_source": origin or None,
            "warnings": lint(program.layers),
        }
        with observer.running():
            initial = {}
            weight_rows = []
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    raise WgtError(f"non-finite initial parameter `{name}`", path=program.path, code="E0401")
                owner = name.rsplit(".", 1)[0]
                if watch and owner != watch and not owner.startswith(watch + "."):
                    continue
                weight_rows.append({"parameter": name, "trainable": param.requires_grad, "before": stats(param)})
                if mode != "trace":
                    initial[name] = param.detach().clone()
            report["weights"] = weight_rows
            if mode == "trace":
                model.eval()
                x, kind = sample_input(program, sample)
                with torch.no_grad():
                    result = model(x)
                report.update(sample=kind, output=stats(result))
            else:
                spec = program.train
                if not spec:
                    raise WgtError(
                        "debug needs a train statement and its dataset",
                        path=program.path,
                        code="E0404",
                        help="add train data=... or use trace for inference-only inspection",
                    )
                if spec.trainer != "backprop":
                    raise WgtError(
                        "bounded debug currently supports only trainer=backprop", spec.line, program.path, code="E0404"
                    )
                x, y, _ = _dataset_tensors(program, spec.data)
                y = _targets(y, spec.loss)
                size = min(spec.batch or len(x), len(x), 64)
                x, y = x[:size], y[:size]
                model.train()
                loss_fn = _loss_fn(spec.loss)
                params = [p for p in model.parameters() if p.requires_grad]
                if not params:
                    raise WgtError("debug has no trainable parameters", path=program.path, code="E0402")
                optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[
                    spec.optimizer
                ]
                opt = optimizer(params, lr=spec.lr, weight_decay=spec.weight_decay)
                losses, gradient_rows = [], []
                seen_missing, seen_zero = set(), set()
                for step in range(steps):
                    observer.step = step
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x)
                    loss = loss_fn(prediction, y)
                    if not torch.isfinite(loss):
                        raise WgtError(
                            "loss is NaN or infinity",
                            spec.line,
                            program.path,
                            code="E0401",
                            notes=[f"optimizer step: {step}"],
                            help="check targets and the loss function",
                        )
                    losses.append(float(loss.detach()))
                    if not loss.requires_grad:
                        raise WgtError(
                            "loss has no gradient path to trainable parameters",
                            spec.line,
                            program.path,
                            code="E0402",
                            help="connect a trainable branch to the output",
                        )
                    loss.backward()
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        owner = name.rsplit(".", 1)[0]
                        owner = next(
                            (
                                key
                                for key in sorted(observer.layers, key=len, reverse=True)
                                if owner == key or owner.startswith(key + ".")
                            ),
                            None,
                        )
                        if param.grad is None:
                            seen_missing.add(name)
                        elif not torch.isfinite(param.grad).all():
                            if owner:
                                observer.fail(owner, f"non-finite gradient in `{name}`", "E0402")
                            raise WgtError(f"non-finite gradient: {name}", code="E0402")
                        else:
                            norm = float(torch.linalg.vector_norm(param.grad.detach().double()))
                            if norm == 0:
                                seen_zero.add(name)
                            if step == steps - 1:
                                gradient_rows.append({"parameter": name, "norm": norm})
                    opt.step()
                    for name, param in model.named_parameters():
                        if not torch.isfinite(param).all():
                            raise WgtError(
                                f"optimizer produced non-finite parameter `{name}`",
                                spec.line,
                                program.path,
                                code="E0402",
                                notes=[f"optimizer step: {step}"],
                                help="lower the learning rate and inspect gradient norms",
                            )
                observer.step = None
                with torch.no_grad():
                    final = float(loss_fn(model(x), y))
                if not math.isfinite(final):
                    raise WgtError("final loss is NaN or infinity", spec.line, program.path, code="E0401")
                for code, names, wording in [
                    ("W0401", seen_missing, "has no gradient path"),
                    ("W0402", seen_zero, "had an exactly zero gradient"),
                ]:
                    for name in sorted(names):
                        report["warnings"].append(
                            warning(
                                code,
                                f"`{name}` {wording}",
                                help="inspect branch connectivity and activations; a zero gradient can be intentional",
                            )
                        )
                for row in weight_rows:
                    name = row["parameter"]
                    param = model.get_parameter(name)
                    delta = param.detach().double() - initial[name].double()
                    update_norm = float(torch.linalg.vector_norm(delta))
                    before_norm = row["before"]["norm"]
                    row.update(
                        after=stats(param),
                        gradient_status=("frozen" if not param.requires_grad else
                                         "missing" if param.grad is None else
                                         "zero" if not torch.count_nonzero(param.grad) else "nonzero"),
                        gradient=stats(param.grad) if param.grad is not None else None,
                        update_norm=update_norm,
                        relative_update=update_norm / before_norm if before_norm else None,
                        changed_elements=int(torch.count_nonzero(delta)),
                    )
                report.update(
                    steps=steps,
                    batch_rows=size,
                    losses=losses,
                    final_loss=final,
                    gradients=gradient_rows,
                    data_note="fixed first minibatch; not a generalization benchmark",
                )
        report["loops"] = [
            {
                "layer": name,
                "op": layer.op,
                "parameter_sets": 1,
                **({"max_iterations_per_call": layer.args["steps"], "until": layer.args["until"]}
                   if layer.op in {"recur", "ponder"} and "until" in layer.args else
                   {"iterations_per_call": layer.args["steps"] if layer.op == "recur" else layer.input_spec["shape"][0]}),
            }
            for name, layer in observer.layers.items()
            if layer.op in ("scan", "recur", "ponder")
        ]
        report["warnings"].extend(observer.warnings())
        report.update(
            records=observer.records,
            calls=observer.calls,
            truncated=observer.truncated,
            timing_note="CPU wall time, inclusive of child execution and observer overhead",
        )
        return report


def command(args):
    from .syntax import parse_file

    times = None
    if args.time is not None:
        try:
            times = {int(item) for item in args.time.split(",")}
            if not times or min(times) < 0:
                raise ValueError()
        except ValueError:
            raise WgtError("--time expects comma-separated non-negative indices", code="E0404") from None
    report = execute(
        parse_file(args.source, check_training=args.command != "trace"),
        mode=args.command,
        sample=getattr(args, "sample", None),
        steps=getattr(args, "steps", 10),
        watch=args.watch,
        times=times,
        max_parameters=args.max_parameters,
        weights=getattr(args, "weights", None),
    )
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"{report['mode']}: {report['model']} · CPU · {report['parameters']} parameters")
        print(f"weights: {report['initialization']}")
        if report["weight_source"]:
            print(f"  {report['weight_source']['path']} · sha256={report['weight_source']['sha256']}")
            if report["weight_source"]["untrained_tensors"]:
                print("  note: artifact contains tensors without trained provenance")
        for row in report["weights"]:
            suffix = ""
            if "update_norm" in row:
                suffix = f" · gradient={row['gradient_status']} · update_norm={row['update_norm']:.4g}"
            print(f"  {row['parameter']}: norm={row['before']['norm']:.4g} · trainable={row['trainable']}{suffix}")
        if "losses" in report:
            print(f"{report['steps']} optimizer steps · loss {report['losses'][0]:.6g} -> {report['final_loss']:.6g}")
        else:
            print(f"sample: {report['sample']} · output {report['output']['shape']}")
        for loop in report["loops"]:
            print(f"{loop['layer']}: {loop['iterations_per_call']} iterations per call, 1 shared parameter set")
        if args.watch or args.command == "trace":
            for record in report["records"]:
                value = record["outputs"][0]
                where = f" time={record['iterations']}" if record["iterations"] else ""
                gradient = f" grad_norm={value['gradient']['norm']:.4g}" if "gradient" in value else ""
                print(
                    f"  {record['layer']}{where}: {value['shape']} [{value['min']:.4g}, {value['max']:.4g}] "
                    f"{record['elapsed_ms']:.3f} ms{gradient}"
                )
        for item in report["warnings"]:
            print(render(item))
        if report["truncated"]:
            print("note: event display capped at 2000; all operations still checked for non-finite values")
        print("No model weights or revisions were written.")
    return 0
