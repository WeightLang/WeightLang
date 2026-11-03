"""Generate readable, standalone PyTorch source from checked WeightLang graphs."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import shutil

from .artifact import load_artifact
from .syntax import WgtError, parse_file


class _Emitter:
    def __init__(self, interface=None):
        self.interface = interface
        self.classes = []
        self.helpers = {}
        self.counter = 0

    def constructor(self, layer):
        op, a = layer["op"], layer.get("args", {})
        if op == 'scan_state' or (op == 'seq' and layer.get('returns')):
            return self.record_block(layer) + '()'
        if op == "route":
            from .control import Route
            self.helpers["Route"] = inspect.getsource(Route)
            branches = ", ".join(f"{c['name']!r}: {self.constructor(c)}" for c in layer["layers"])
            return f"Route({{{branches}}}, {len(layer['output']['shape'])})"
        if op in {"seq", "residual", "recur", "scan", "ponder"}:
            return self.block(layer["layers"], op, {**a, **({"_rank":len(layer["input"]["shape"])} if op in {"ponder", "recur"} else {})}) + "()"
        from .primitives import SPECS
        if op in SPECS:
            from .primitive_runtime import TensorPrimitive
            self.helpers["TensorPrimitive"] = inspect.getsource(TensorPrimitive)
            return f"TensorPrimitive({op!r}, {a!r}, {len(layer['input']['shape'])})"
        simple = {
            "relu": "ReLU",
            "tanh": "Tanh",
            "sigmoid": "Sigmoid",
            "gelu": "GELU",
            "silu": "SiLU",
            "identity": "Identity",
        }
        if op == "gelu" and a.get("approximate", 0):
            return "nn.GELU(approximate='tanh')"
        if op in simple:
            return f"nn.{simple[op]}()"
        if op == "linear":
            return f"nn.Linear({a['in']}, {a['out']}, bias={bool(a.get('bias', 1))})"
        if op == "layernorm":
            return f"nn.LayerNorm({a['dim']}, eps={a.get('eps', 1e-5)}, bias={bool(a.get('bias', 1))}, elementwise_affine={bool(a.get('affine', 1))})"
        if op == "dropout":
            return f"nn.Dropout(p={a['p']!r})"
        if op == "softmax":
            return "nn.Softmax(dim=-1)"
        if op == "embedding":
            return f"nn.Embedding({a['vocab']}, {a['dim']}, padding_idx={a.get('padding_idx')!r})"
        if op == "conv2d":
            return f"nn.Conv2d({a['in']}, {a['out']}, {a['kernel']}, stride={a['stride']}, padding={a['padding']})"
        if op == "maxpool2d":
            return f"nn.MaxPool2d({a['kernel']}, stride={a['stride']})"
        if op == "flatten":
            return "nn.Flatten(start_dim=1)"
        if op in {"conv1d", "position_embedding", "self_attention", "rope_attention", "sparse_moe", "rmsnorm", "scale", "meanpool", "last", "globalavgpool2d"}:
            from . import modern

            helper_name = {
                "conv1d": "Conv1d",
                "position_embedding": "PositionEmbedding",
                "rmsnorm": "RMSNorm",
                "scale": "Scale",
                "rope_attention": "RopeAttention",
                "sparse_moe": "SparseMoE",
                "self_attention": "SelfAttention",
                "meanpool": "MeanPool",
                "last": "Last",
                "globalavgpool2d": "GlobalAvgPool2d",
            }[op]
            ports = a.get('ports',0)
            if op in {'rope_attention','sparse_moe'}:
                a = {k:v for k,v in a.items() if k != 'ports'}
            cls = getattr(modern, helper_name)
            # These audited helpers import only torch/nn. Copy just the used class,
            # not the backend dispatcher or any WeightLang runtime dependency.
            if op == "rope_attention":
                self.helpers["RMSNorm"] = inspect.getsource(modern.RMSNorm)
            self.helpers[cls.__name__] = inspect.getsource(cls)
            if ports:
                cls = modern.CachedAttention if op == 'rope_attention' else modern.MoEPorts
                self.helpers[cls.__name__] = inspect.getsource(cls)
            return f"{cls.__name__}(**{a!r})"
        if op in {"rearrange", "reduce"}:
            from .axes import plan

            p = plan(a["pattern"], tuple(layer["input"]["shape"]), a.get("sizes", {}), a.get("reduction"))
            self.helpers["_AxisTransform"] = _AXIS_HELPER
            return f"_AxisTransform({tuple(layer['input']['shape'])!r}, {p!r}, {a.get('reduction')!r})"
        if op in {"add", "mul", "concat", "mix", "where"}:
            return "nn.Identity()"
        raise WgtError(f"Python export does not support operator {op!r}")

    def record_block(self, layer):
        from .dataflow import Flow, RecordFlow, StateScan
        for cls in (Flow,RecordFlow,StateScan):
            self.helpers[cls.__name__] = inspect.getsource(cls)
        name = f'Block{self.counter}'
        self.counter += 1
        base = 'StateScan' if layer['op'] == 'scan_state' else 'RecordFlow'
        init = ['        nn.Sequential.__init__(self)',f"        self._return_routes = {layer['returns']!r}"]
        routes = [(child['name'],child.get('inputs',[]),child['op']) for child in layer['layers']]
        init.append(f'        self._flow_routes = {routes!r}')
        if base == 'StateScan':
            spec = layer['results']['output']
            init.append(f"        self._emission_shape, self._emission_dtype = {spec['shape'][1:]!r}, {spec['dtype']!r}")
        from .dataflow import Merge
        self.helpers['Merge'] = inspect.getsource(Merge)
        embeddings = []
        for child in layer['layers']:
            access = f"self._modules[{child['name']!r}]"
            constructor = f"Merge({child['op']!r})" if child['op'] in {'add','mul','concat','mix','where'} else self.constructor(child)
            init.append(f'        {access} = {constructor}')
            if child['op'] == 'embedding':
                embeddings.append(access)
            tie = child.get('args',{}).get('tie_embedding',0)
            if tie:
                init.append(f'        {access}.weight = {embeddings[tie-1]}.weight')
        self.classes.append(f'class {name}({base}):\n    def __init__(self):\n'+'\n'.join(init)+'\n')
        return name

    def block(self, layers, op="seq", args=None, root=False):
        args = args or {}
        if root and self.interface:
            from .dataflow import NamedFlow, Merge
            from .symbolic import DimensionContract
            self.helpers["DimensionContract"] = inspect.getsource(DimensionContract)
            self.helpers["NamedFlow"] = inspect.getsource(NamedFlow)
            self.helpers["Merge"] = inspect.getsource(Merge)
            init = [f"        super().__init__({layers!r}, {self.interface!r})"]
            embeddings = []
            for layer in layers:
                access = f"self._modules[{layer['name']!r}]"
                constructor = f"Merge({layer['op']!r})" if layer['op'] in {"add", "mul", "concat", "mix", "where"} else self.constructor(layer)
                init.append(f"        {access} = {constructor}")
                if layer['op'] == 'embedding':
                    embeddings.append(access)
                tied = layer.get('args', {}).get('tie_embedding', 0)
                if tied:
                    init.append(f"        {access}.weight = {embeddings[tied - 1]}.weight")
            from .parameter_identity import bind_parameters
            self.helpers['bind_parameters'] = inspect.getsource(bind_parameters)
            from .adapters import LoRALinear, DoRALinear, PiSSALinear, QLoRALinear, NF4LinearFunction, attach_adapters
            for helper in (LoRALinear, DoRALinear, PiSSALinear, NF4LinearFunction, QLoRALinear):
                self.helpers[helper.__name__] = inspect.getsource(helper)
            self.helpers['attach_adapters'] = inspect.getsource(attach_adapters)
            init.append(f"        bind_parameters(self, {self.interface.get('parameter_ids',{})!r}, {self.interface.get('parameter_shares',{})!r})")
            init.append(f"        attach_adapters(self, {self.interface.get('adapters',{})!r})")
            from .weight_types import validate_types, project_weight, weight_view, WeightTyped, attach_weight_types
            for helper in (validate_types, project_weight, weight_view, WeightTyped, attach_weight_types):
                self.helpers[helper.__name__] = inspect.getsource(helper)
            init.append(f"        attach_weight_types(self, {self.interface.get('weight_types',{})!r}, {self.interface.get('weight_assignments',[])!r})")
            self.classes.append("class Model(NamedFlow):\n    def __init__(self):\n" + "\n".join(init) + "\n")
            return "Model"
        name = "Model" if root else f"Block{self.counter}"
        self.counter += 1
        init = ["        super().__init__()"]
        body = []
        values = {"input": "x", "state": "state"}
        previous = "x"
        embeddings = []
        for index, layer in enumerate(layers):
            constructor = self.constructor(layer)
            # _modules avoids Python keywords and method-name collisions while
            # retaining exact WeightLang state_dict paths.
            access = f"self._modules[{layer['name']!r}]"
            init.append(f"        {access} = {constructor}")
            if layer["op"] == "embedding":
                embeddings.append(access)
            tied = layer.get("args", {}).get("tie_embedding", 0)
            if tied:
                init.append(f"        {access}.weight = {embeddings[tied - 1]}.weight")
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
                expression = f"{access}({', '.join(refs)})"
            previous = f"v{index}"
            body.append(f"        {previous} = {expression}  # {layer['name']}")
            if layer.get('results'):
                body.append(f"        {previous}_results = {previous}")
                for field in layer['results']:
                    values[f"{layer['name']}.{field}"] = f"{previous}_results[{field!r}]"
                body.append(f"        {previous} = {previous}_results['output']")
            values[layer["name"]] = previous
        body.append(f"        return {previous}")
        if op in {"recur", "ponder"} and "until" in args:
            from .dataflow import halt_update
            self.helpers["halt_update"] = inspect.getsource(halt_update)
            body[-1] = f"        return {previous}, {values[args['until']]}"
        method = "_step" if op in {"scan", "recur", "ponder"} else "forward"
        if op == "residual":
            body[-1] = f"        return x + {previous}"
        signature = "self, x, state" if op == "scan" else "self, x"
        code = f"class {name}(nn.Module):\n    def __init__(self):\n" + "\n".join(init)
        code += f"\n\n    def {method}({signature}):\n" + "\n".join(body) + "\n"
        if op == "ponder" or (op == "recur" and args.get("compact")):
            from .control import check_probability, compact_recur, ponder_steps
            runner = ponder_steps if op == "ponder" else compact_recur
            self.helpers["check_probability"] = inspect.getsource(check_probability)
            self.helpers[runner.__name__] = inspect.getsource(runner)
            code += f"\n    def forward(self, x):\n        return {runner.__name__}(self._step, x, {args['steps']}, {args['_rank']})\n"
        elif op == "recur":
            if "until" in args:
                code += f"\n    def forward(self, x):\n        active = None\n        for _ in range({args['steps']}):\n            candidate, probability = self._step(x)\n            x, active = halt_update(x, candidate, probability, active)\n            if not bool(active.any()):\n                break\n        return x\n"
            else:
                code += f"\n    def forward(self, x):\n        for _ in range({args['steps']}):\n            x = self._step(x)\n        return x\n"
        elif op == "scan":
            code += f"\n    def forward(self, x, state=None):\n        if state is None:\n            state = x.new_zeros((*x.shape[:-2], {args['hidden']}))\n        outputs = []\n        for item in x.unbind(dim=-2):\n            state = self._step(item, state)\n            outputs.append(state)\n        return torch.stack(outputs, dim=-2) if outputs else x[..., :0, :1].expand(*x.shape[:-2], 0, {args['hidden']})\n"
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
    named_interface = None
    frozen = []
    weights = None
    digest = None
    if source.is_dir():
        artifact = load_artifact(source)
        layers = artifact.manifest["layers"]
        if artifact.manifest["format_version"] >= 4:
            named_interface = artifact.manifest["interface"]
        inputs, outputs = artifact.manifest["interface"]["input"], artifact.manifest["interface"]["output"]
        frozen = [name for name, param in artifact.module.named_parameters() if not param.requires_grad]
        weights = source / "weights.safetensors"
        digest = artifact.manifest["weights"]["sha256"]
        if sidecar.exists() or sidecar.is_symlink():
            raise WgtError(f"Python export refuses existing weights: {sidecar}")
    else:
        program = parse_file(source)
        from .ports import interface
        named_interface = interface(program)
        layers = [layer.to_dict() for layer in program.layers]
        inputs = {"dtype": program.input.dtype, "shape": list(program.input.shape)}
        outputs = {"dtype": program.output.dtype, "shape": list(program.output.shape)}
    emitter = _Emitter(named_interface)
    emitter.block(layers, root=True)
    code = '"""Generated PyTorch model. Named interfaces accept an optional leading batch; legacy graphs require it.\n'
    code += (
        "Weights exported from a verified artifact.\n"
        if weights
        else "Architecture only: training/import/inherit/init policies are not executed.\nModel() creates fresh PyTorch initialization, not trained weights.\n"
    )
    code += '"""\n\nfrom pathlib import Path\nimport hashlib\nimport math\nimport re\n\nimport torch\nfrom torch import nn\n\n'
    code += f"INPUT = {inputs!r}\nOUTPUT = {outputs!r}\n\n"
    if named_interface:
        code += f"INTERFACE = {named_interface!r}\n\n"
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
