# Execution devices and parameter placement

Source builds support CPU, CUDA and MPS in `weightc build`, `run`, and `evaluate`.
The default is `cpu`; `cuda` selects PyTorch's current CUDA device, and `cuda:N`
selects its visible index (after `CUDA_VISIBLE_DEVICES` remapping). `mps` selects
the macOS Metal backend when available. An unavailable
or invalid device is an error, never a silent CPU fallback.

```sh
weightc build --no-store examples/xor.wgt --device cuda:0
weightc run weights/xor_mlp --device cpu --input '[[0,1]]'
weightc build --no-store examples/xor.wgt -o weights/mixed --device cuda:0 \
  --place hidden.weight=cpu --place head=cpu
```

`--device` sets the default execution and parameter residence device.
`--place TARGET=DEVICE` accepts exact dotted module or parameter names:

- A module rule sets its execution device and the residence of descendant
  parameters. Nested module rules override parent rules.
- An exact parameter rule changes only that parameter's residence. It does not
  change the layer's execution device. For example `hidden.weight=cpu` can keep
  the weight and its optimizer state on CPU while `hidden` computes on CUDA.
- The most specific matching rule wins, independently of argument order.
  Unknown targets and repeated CLI targets fail rather than being ignored.
- Internal attention modules are addressable, e.g. `body.attn.qkv.weight`.
  Use artifact tensor names from `weightc inspect`.

A JSON object works for longer maps:

```json
{"hidden": "cuda:0", "hidden.weight": "cpu", "head": "cpu"}
```

```sh
weightc build --no-store examples/xor.wgt --device-map devices.json
```

Explicit `--place` entries override entries in the JSON file. Input activations
move at leaf-layer boundaries. Parameter tensors on another device are copied
for that operation without moving the resident parameter. These copies remain
in autograd: gradients return to the resident parameter, and optimizers update
it there. Frozen parameters remain frozen. Residual inputs move to the branch's
output device before addition. Integer token inputs preserve their dtype.
Datasets and minibatch indices stay on CPU; selected inputs and loss targets
move as needed. `predict` returns a tensor on the last layer's execution device;
CLI output remains JSON.

```python
from weightlang import load_artifact
from weightlang.compiler import compile_file

result = compile_file("examples/xor.wgt", "weights/mixed",
                      device="cuda:0", placements={"hidden.weight": "cpu"})
model = load_artifact(result.path, device="cuda:0", placements={"head": "cpu"})
print(model.predict([[0, 1]]).cpu().tolist())
print({name: str(p.device) for name, p in model.module.named_parameters()})
```

Initialization and imports occur on CPU before placement. The compiler preserves
CPU and participating CUDA/MPS RNG states. The manifest records the resolved plan
under `built_with.execution` (default, rules, leaf execution devices, parameter
residence). Safetensors remain portable CPU-serialized float32 data. Loading
never replays the build machine's device plan: CPU is still the default, and
callers may supply a new plan. Existing artifact versions remain loadable.
Use the loader's placement arguments instead of subsequently calling
`module.to(...)`, which does not update execution rules.

`wgt build/run/test` forward the same options, with project defaults available
in `wgt.toml` (see [project tooling](wgt.md)). Options are execution configuration,
not `.wgt` syntax; the same model source works on different machines.

This is explicit placement, not automatic memory scheduling, tensor sharding,
DDP, asynchronous offload, or mixed precision. Temporary copies and optimizer
state still consume memory; cross-device transfers can be expensive. A single
operator executes on one device. CUDA results are not promised bit-identical
to CPU results or across hardware/runtime versions. CPU-only installations
cannot run CUDA commands.
