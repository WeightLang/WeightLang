"""Generate readable, standalone PyTorch source from checked WeightLang graphs."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import shutil

from .artifact import load_artifact
from .syntax import WgtError, parse_file


class _Emitter:
    def __init__(self):
        self.classes = []
        self.helpers = {}
        self.counter = 0

    def constructor(self, layer):
        op, a = layer["op"], layer.get("args", {})
        if op in {"seq", "residual", "recur", "scan"}:
            return self.block(layer["layers"], op, a) + "()"
        simple = {
            "relu": "ReLU",
            "tanh": "Tanh",
            "sigmoid": "Sigmoid",
            "gelu": "GELU",
            "silu": "SiLU",
            "identity": "Identity",
        }
        if op in simple:
            return f"nn.{simple[op]}()"
        if op == "linear":
            return f"nn.Linear({a['in']}, {a['out']})"
        if op == "layernorm":
            return f"nn.LayerNorm({a['dim']})"
        if op == "dropout":
            return f"nn.Dropout(p={a['p']!r})"
        if op == "softmax":
            return "nn.Softmax(dim=-1)"
        if op == "embedding":
            return f"nn.Embedding({a['vocab']}, {a['dim']})"
        if op == "conv2d":
            return f"nn.Conv2d({a['in']}, {a['out']}, {a['kernel']}, stride={a['stride']}, padding={a['padding']})"
        if op == "maxpool2d":
            return f"nn.MaxPool2d({a['kernel']}, stride={a['stride']})"
        if op == "flatten":
            return "nn.Flatten(start_dim=1)"
        if op in {"position_embedding", "self_attention", "meanpool", "last", "globalavgpool2d"}:
            from . import modern

            helper_name = {
                "position_embedding": "PositionEmbedding",
                "self_attention": "SelfAttention",
                "meanpool": "MeanPool",
                "last": "Last",
                "globalavgpool2d": "GlobalAvgPool2d",
            }[op]
            cls = getattr(modern, helper_name)
            # These audited helpers import only torch/nn. Copy just the used class,
            # not the backend dispatcher or any WeightLang runtime dependency.
            self.helpers[cls.__name__] = inspect.getsource(cls)
            return f"{cls.__name__}({', '.join(f'{k}={v!r}' for k, v in a.items())})"
        if op in {"rearrange", "reduce"}:
            from .axes import plan

            p = plan(a["pattern"], tuple(layer["input"]["shape"]), a.get("sizes", {}), a.get("reduction"))
            self.helpers["_AxisTransform"] = _AXIS_HELPER
            return f"_AxisTransform({tuple(layer['input']['shape'])!r}, {p!r}, {a.get('reduction')!r})"
        if op in {"add", "mul", "concat", "mix", "where"}:
            return "nn.Identity()"
        raise WgtError(f"Python export does not support operator {op!r}")

    def block(self, layers, op="seq", args=None, root=False):
        args = args or {}
        name = "Model" if root else f"Block{self.counter}"
        self.counter += 1
        init = ["        super().__init__()"]
        body = []
        values = {"input": "x", "state": "state"}
        previous = "x"
        for index, layer in enumerate(layers):
            constructor = self.constructor(layer)
            # _modules avoids Python keywords and method-name collisions while
            # retaining exact WeightLang state_dict paths.
            access = f"self._modules[{layer['name']!r}]"
            init.append(f"        {access} = {constructor}")
            refs = [values[ref] for ref in layer.get("inputs", [])] or [previous]
            kind = layer["op"]
            if kind in {"add", "mul"}:
                symbol = " + " if kind == "add" else " * "
                expression = symbol.join(refs)
            elif kind == "concat":
                expression = f"torch.cat([{', '.join(refs)}], dim=-1)"
            elif kind == "mix":
                g, a, b = refs
                expression = f"{g} * {a} + (1 - {g}) * {b}"
            elif kind == "where":
                c, a, b = refs
                expression = f"torch.where({c} > 0, {a}, {b})"
            else:
                expression = f"{access}({refs[0]})"
            previous = f"v{index}"
            body.append(f"        {previous} = {expression}  # {layer['name']}")
            values[layer["name"]] = previous
        body.append(f"        return {previous}")
        method = "_step" if op in {"scan", "recur"} else "forward"
        if op == "residual":
            body[-1] = f"        return x + {previous}"
        signature = "self, x, state" if op == "scan" else "self, x"
        code = f"class {name}(nn.Module):\n    def __init__(self):\n" + "\n".join(init)
        code += f"\n\n    def {method}({signature}):\n" + "\n".join(body) + "\n"
        if op == "recur":
            code += f"\n    def forward(self, x):\n        for _ in range({args['steps']}):\n            x = self._step(x)\n        return x\n"
        elif op == "scan":
            code += f"\n    def forward(self, x):\n        state = x.new_zeros((*x.shape[:-2], {args['hidden']}))\n        outputs = []\n        for item in x.unbind(dim=-2):\n            state = self._step(item, state)\n            outputs.append(state)\n        return torch.stack(outputs, dim=-2)\n"
        self.classes.append(code)
        return name


_AXIS_HELPER = """class _AxisTransform(nn.Module):
    def __init__(self, shape, plan, reduction):
        super().__init__()
        self.shape, self.plan, self.reduction = shape, plan, reduction

    def forward(self, x):
        rank = len(self.shape)
        if x.ndim < rank or tuple(x.shape[-rank:]) != self.shape:
            raise ValueError(f"expected sample tail {self.shape}, got {tuple(x.shape)}")
        batch = tuple(x.shape[:-rank])
        offset = len(batch)
        x = x.reshape(batch + tuple(self.plan['expanded_shape']))
        x = x.permute(tuple(range(offset)) + tuple(offset + i for i in self.plan['permutation']))
        axes = tuple(offset + i for i in self.plan['reduction_axes'])
        if axes:
            x = x.mean(dim=axes) if self.reduction == 'mean' else x.sum(dim=axes)
        return x.reshape(batch + tuple(self.plan['output_shape']))
"""


def export_python(source: str | Path, output: str | Path) -> dict:
    """Export source architecture or verified artifact weights; never overwrite.

    Source export parses only: it does not train, import weights or run plugins.
    Artifact export preserves current weights and frozen parameters, not optimizer
    state or device placement. Generated models accept a leading batch dimension.
    """
    source, output = Path(source), Path(output)
    if output.suffix != ".py":
        raise WgtError("Python export output must end in .py")
    sidecar = output.with_suffix(".safetensors")
    if output.exists() or output.is_symlink():
        raise WgtError(f"Python export refuses existing output: {output}")
    frozen = []
    weights = None
    digest = None
    if source.is_dir():
        artifact = load_artifact(source)
        layers = artifact.manifest["layers"]
        inputs, outputs = artifact.manifest["interface"]["input"], artifact.manifest["interface"]["output"]
        frozen = [name for name, param in artifact.module.named_parameters() if not param.requires_grad]
        weights = source / "weights.safetensors"
        digest = artifact.manifest["weights"]["sha256"]
        if sidecar.exists() or sidecar.is_symlink():
            raise WgtError(f"Python export refuses existing weights: {sidecar}")
    else:
        program = parse_file(source)
        layers = [layer.to_dict() for layer in program.layers]
        inputs = {"dtype": program.input.dtype, "shape": list(program.input.shape)}
        outputs = {"dtype": program.output.dtype, "shape": list(program.output.shape)}
    emitter = _Emitter()
    emitter.block(layers, root=True)
    code = '"""Generated PyTorch model. Input tensors must include a leading batch dimension.\n'
    code += (
        "Weights exported from a verified artifact.\n"
        if weights
        else "Architecture only: training/import/inherit/init policies are not executed.\nModel() creates fresh PyTorch initialization, not trained weights.\n"
    )
    code += '"""\n\nfrom pathlib import Path\nimport hashlib\n\nimport torch\nfrom torch import nn\n\n'
    code += f"INPUT = {inputs!r}\nOUTPUT = {outputs!r}\n\n"
    code += "\n\n".join([*emitter.helpers.values(), *emitter.classes])
    if weights:
        code += f'''\n\ndef load_model(weights_path=None, device="cpu"):
    """Verify and load the exported weights; returns an eval-mode model."""
    from safetensors.torch import load_file

    path = Path(weights_path) if weights_path is not None else Path(__file__).with_name({sidecar.name!r})
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != {digest!r}:
        raise ValueError("exported weights SHA256 mismatch")
    model = Model()
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    for name in {frozen!r}:
        model.get_parameter(name).requires_grad_(False)
    return model.to(device).eval()
'''
    # Check generated syntax before any output is created.
    compile(code, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        if weights:
            with sidecar.open("xb") as target:
                created.append(sidecar)
                with weights.open("rb") as stream:
                    shutil.copyfileobj(stream, target, length=1024 * 1024)
            with sidecar.open("rb") as stream:
                copied_hash = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    copied_hash.update(chunk)
                actual = copied_hash.hexdigest()
            if actual != digest:
                raise WgtError("weights changed during Python export")
        with output.open("x", encoding="utf-8") as stream:
            created.append(output)
            stream.write(code)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return {
        "python": str(output),
        "weights": str(sidecar) if weights else None,
        "mode": "artifact" if weights else "architecture-only",
    }
