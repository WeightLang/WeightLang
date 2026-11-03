"""Named tensor flow and bounded, differentiable loops. No Python source evaluation."""

from collections import OrderedDict

import torch
from torch import nn

from .symbolic import DimensionContract


class Merge(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, values):
        first, *rest = values
        rest = [value.to(first.device) for value in rest]
        if self.op == "concat":
            return torch.cat([first, *rest], dim=-1)
        if self.op == "mix":
            a, b = rest
            return first * a + (1 - first) * b
        if self.op == "where":
            a, b = rest
            return torch.where(first > 0, a, b)
        for value in rest:
            first = first + value if self.op == "add" else first * value
        return first


class Flow(nn.Sequential):
    def __init__(self, layers, build):
        super().__init__(OrderedDict((layer["name"], build(layer)) for layer in layers))
        self._flow_routes = [(layer["name"], layer.get("inputs", []), layer["op"]) for layer in layers]

    def forward(self, x, state=None):
        return self._evaluate(x, state)[0]

    def _evaluate(self, x, state=None, bindings=None):
        values = {"input": x, **(bindings or {})}
        if state is not None:
            values["state"] = state
        for name, refs, op in self._flow_routes:
            args = [values[ref] for ref in refs] if refs else [x]
            if op == "scan" and len(args) == 2:
                x = self._modules[name](*args)
            else:
                x = self._modules[name](args) if op in {"add", "mul", "concat", "mix", "where"} else self._modules[name](*args)
            if isinstance(x, dict):
                values.update({f'{name}.{field}': value for field,value in x.items()})
                x = x['output']
            values[name] = x
        return x, values


def halt_update(previous, candidate, probability, active):
    """Freeze completed samples; the discrete stopping decision has no gradient."""
    p = probability.detach()
    if not bool(torch.isfinite(p).all()) or not bool(((p >= 0) & (p <= 1)).all()):
        raise ValueError("recur until probability must be finite and in [0, 1]")
    if active is None:
        active = torch.ones_like(p, dtype=torch.bool)
    mask = active.reshape((*active.shape[:-1], *([1] * (candidate.ndim - active.ndim + 1))))
    result = torch.where(mask.to(candidate.device), candidate, previous.to(candidate.device))
    active = active & (torch.rand_like(p) >= p)
    return result, active


class Recur(Flow):
    def __init__(self, layers, build, steps, until=None, compact=False, rank=None, ponder=False):
        super().__init__(layers, build)
        self._loop_steps = steps
        self._loop_until = until
        self._loop_compact, self._loop_rank, self._loop_ponder = compact, rank, ponder

    def forward(self, x):
        from .debugger import iteration

        if self._loop_compact or self._loop_ponder:
            from .control import compact_recur, ponder_steps
            index = 0
            def step(value):
                nonlocal index
                with iteration(index):
                    candidate, values = self._evaluate(value)
                index += 1
                return candidate, values[self._loop_until]
            runner = ponder_steps if self._loop_ponder else compact_recur
            return runner(step, x, self._loop_steps, self._loop_rank)
        active = None
        for index in range(self._loop_steps):
            with iteration(index):
                candidate, values = self._evaluate(x)
                if self._loop_until is None:
                    x = candidate
                else:
                    x, active = halt_update(x, candidate, values[self._loop_until], active)
                    if not bool(active.any()):
                        break
        return x


class Scan(Flow):
    def __init__(self, layers, build, hidden):
        super().__init__(layers, build)
        self._state_width = hidden

    def forward(self, x, state=None):
        if state is None:
            state = x.new_zeros((*x.shape[:-2], self._state_width))
        elif state.shape != (*x.shape[:-2], self._state_width) or state.dtype != x.dtype or state.device != x.device:
            raise ValueError("scan initial state shape/dtype/device mismatch")
        outputs = []
        from .debugger import iteration

        for index, item in enumerate(x.unbind(dim=-2)):
            with iteration(index):
                state = super().forward(item, state)
                outputs.append(state)
        return torch.stack(outputs, dim=-2) if outputs else x[..., :0, :1].expand(*x.shape[:-2], 0, self._state_width)


class ResidualFlow(Flow):
    def forward(self, x):
        out = super().forward(x)
        return x.to(out.device) + out


class NamedFlow(nn.Sequential):
    """An explicit tensor record in and out; request state never lives on modules."""
    def __init__(self, layers, interface):
        super().__init__()
        self._port_interface = interface
        self._port_routes = [(layer["name"], layer.get("inputs", []), layer["op"]) for layer in layers]

    @staticmethod
    def _validate(value, spec, batch=None, device=None):
        dtype = {"f32": torch.float32, "i64": torch.int64, "bool": torch.bool}[spec["dtype"]]
        shape = tuple(spec["shape"])
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim not in {len(shape), len(shape) + 1} or (tuple(value.shape[-len(shape):]) if shape else ()) != shape:
            raise ValueError(f"port {spec['name']}: expected {spec['dtype']}{list(shape)} with optional leading batch")
        leading = tuple(value.shape[:-len(shape)]) if shape else tuple(value.shape)
        if batch is not None and leading != batch:
            raise ValueError(f"port {spec['name']}: batch dimensions must agree")
        if device is not None and value.device != device:
            raise ValueError(f"port {spec['name']}: input devices must agree")
        return leading

    def forward(self, *args, **kwargs):
        ports = self._port_interface["inputs"]
        if len(args) > len(ports):
            raise TypeError("too many positional tensor inputs")
        values = dict(kwargs)
        for port, value in zip(ports, args):
            if port["name"] in values:
                raise TypeError(f"duplicate input {port['name']}")
            values[port["name"]] = value
        if set(values) != {p["name"] for p in ports}:
            raise TypeError("inputs must match the declared port names exactly")
        active = DimensionContract.bind(self._port_interface, values) if self._port_interface.get("dimensions") else self._port_interface
        ports = active["inputs"]
        batch, device = None, None
        for port in ports:
            value = values[port["name"]]
            batch = self._validate(value, port, batch, device)
            device = value.device
        if batch == ():
            values = {name: value.unsqueeze(0) for name, value in values.items()}
        x = values[ports[0]["name"]]
        values["input"] = x
        for name, refs, op in self._port_routes:
            operands = [values[ref] for ref in refs] if refs else [x]
            if op in {"add", "mul", "concat", "mix", "where"}:
                x = self._modules[name](operands)
            else:
                x = self._modules[name](*operands)
            if isinstance(x, dict):
                values.update({f'{name}.{field}': value for field,value in x.items()})
                x = x['output']
            values[name] = x
        result = {}
        for port in active["outputs"]:
            value = values[port["from"]]
            if batch == ():
                value = value.squeeze(0)
            self._validate(value, port, batch)
            result[port["name"]] = value
        return result


class RecordFlow(Flow):
    def __init__(self,layers,build,returns):
        super().__init__(layers,build)
        self._return_routes = returns

    def forward(self,x,*args):
        _,values = self._evaluate(x,bindings={f'arg{i+1}':value for i,value in enumerate(args)})
        return {field:values[ref] for field,ref in self._return_routes.items()}


class StateScan(RecordFlow):
    def __init__(self,layers,build,returns,emission_shape,emission_dtype):
        super().__init__(layers,build,returns)
        self._emission_shape, self._emission_dtype = emission_shape, emission_dtype

    def forward(self,x,*args):
        states = {f'state{i}':value for i,value in enumerate(args)}
        outputs = []
        for item in x.unbind(1):
            _,values = self._evaluate(item,bindings=states)
            outputs.append(values[self._return_routes['output']])
            states = {field:values[ref] for field,ref in self._return_routes.items() if field != 'output'}
        if outputs:
            emitted = torch.stack(outputs,dim=1)
        else:
            dtype = {'f32':torch.float32,'i64':torch.int64,'bool':torch.bool}[self._emission_dtype]
            emitted = torch.empty((x.shape[0],0,*self._emission_shape),device=x.device,dtype=dtype)
            if dtype == torch.float32:
                # Keep an empty chunk connected without inventing cell/parameter work.
                emitted = emitted + x.sum()*0
        return {'output':emitted,**states}
