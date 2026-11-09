"""Differential acceptance against the independent Rust execution backend."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import pytest
import torch

import weightlang as wgt

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "crates/wgtime-core/tests/fixtures"


@pytest.fixture(scope="module", autouse=True)
def native_tools():
    compiler = Path(os.environ.get("WGT_COMPILER", ROOT / "target/debug/wgt"))
    reference = Path(os.environ.get("WGT_REFERENCE", ROOT / "target/debug/examples/torch_reference"))
    assert compiler.is_file(), "Build the wgt binary before SDK acceptance"
    assert reference.is_file(), "Build cargo build -p weightlang-wgt --example torch_reference"
    previous = os.environ.get("WGT_COMPILER")
    os.environ["WGT_COMPILER"] = str(compiler)
    yield
    if previous is None:
        os.environ.pop("WGT_COMPILER", None)
    else:
        os.environ["WGT_COMPILER"] = previous


def native(source, *, error=None, **request):
    executable = os.environ.get("WGT_REFERENCE", str(ROOT / "target/debug/examples/torch_reference"))
    process = subprocess.run(
        [executable], input=json.dumps({"source": source, **request}),
        text=True, capture_output=True, timeout=30, check=False,
    )
    if error is not None:
        assert process.returncode != 0
        assert error in process.stderr, process.stderr
        return process.stderr
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def source(body, declarations="", input_width=1, output_width=1, loss=False):
    result = (
        f"wgt 0.5\nmodel Parity\ninput x:f32[{input_width}]\n"
        f"output y:f32[{output_width}]\n{declarations}\nfn forward(x) {{ {body} }}\n"
    )
    if loss:
        result += "fn loss(prediction,target) { return 0.5*square(prediction-target).mean; }\n"
    return result


MEMORY = source("""
    let bank=memory_alloc(2,2);
    let a=softmax(address);
    memory_write(bank,a,x);
    memory_write(bank,a,value);
    let y=memory_read(bank,a);
    write(state,y);
    memory_free(bank);
    return y;
""", """
parameter address:f32[2]=[0.2,-0.3]
parameter value:f32[2]=[0.4,0.9]
parameter state:f32[2]=[0,0]
""", 2, 2)

WRITE = source("""
    let observation=slice(x,0,1), query=slice(x,1,1);
    let delta=linear(observation,updater), next=read(memory)+delta;
    write(memory,next);
    if positive(observation) { return linear(query,memory); }
    return query;
""", "parameter memory:f32[1,1]=[0]\nparameter updater:f32[1,1]=[0.5]", 2, 1)

LOOP = source("""
    let count=[3];
    let value=x;
    while true {
        value=value*gain;
        count=count+[-1];
        if positive(count) { } else { break; }
    }
    return value;
""", "parameter gain:f32[1]=[1.1]")

BLOCK = """model Block {
input a:f32[1]
output b:f32[1]
parameter weight:f32[1]=[2]
fn forward(a) { return a*weight; }
}
blocks:Block[2];
"""

CASES = [
    pytest.param(source("return linear(x,w,b);", "parameter w:f32[2,2]=[0.2,-0.1,0.3,0.4]\nparameter b:f32[2]=[0.1,0.2]", 2, 2), [0.5, 0.7], {}, id="linear"),
    pytest.param(source("let a=relu(tanh(x*p))+sigmoid(x); let b=log(exp(a)); return sum(clip(minimum(b,[2,2]),-1,1))+mean(-detach(a));", "parameter p:f32[2]=[0.4,-0.2]", 2, 1), [0.2, -0.3], {}, id="arithmetic-and-unary"),
    pytest.param(source("return log_softmax(x+p)+softmax(x);", "parameter p:f32[2]=[0.2,-0.1]", 2, 2), [0.3, 0.7], {}, id="vector-normalization"),
    pytest.param(source("if positive(mean(x)) { return x*gain; } return -x;", "parameter gain:f32[2]=[2,3]", 2, 2), [0.2, 0.3], {}, id="branch-positive"),
    pytest.param(source("if positive(mean(x)) { return x*gain; } return -x;", "parameter gain:f32[2]=[2,3]", 2, 2), [-0.2, -0.3], {}, id="branch-negative"),
    pytest.param(LOOP, [0.7], {}, id="dynamic-loop"),
    pytest.param(WRITE, [2.0, 3.0], {}, id="read-after-write"),
    pytest.param(WRITE, [-2.0, 3.0], {}, id="unselected-write-gradient"),
    pytest.param(MEMORY, [0.7, -0.2], {}, id="differentiable-memory"),
    pytest.param(source("return route(p,x,x*x);", "parameter p:f32[1]=[0]"), [2.0], {}, id="route-zero-gradient"),
    pytest.param(source("return route(p,x,x*x);", "parameter p:f32[1]=[1]"), [2.0], {}, id="route-one-gradient"),
    pytest.param(source("return gate(p,x*x);", "parameter p:f32[1]=[0]"), [2.0], {}, id="gate-zero-gradient"),
    pytest.param(source("return repeat depth carrying h=x {2*h};", "learn depth:int in 0..=3=1;"), [1.0], {}, id="learned-depth"),
    pytest.param(source("return if enabled {2*x} else {x};", "learn enabled:bool=false;"), [1.0], {"enabled": [0.5]}, id="bool-threshold-ste"),
    pytest.param(source("return x*count;", "learn count:int[2] in -3..=3=[0,0];", 2, 2), [1.0, 2.0], {"count": [-0.5, 0.5]}, id="integer-ties-ste"),
    pytest.param(source("return blend {p => 3*x; (1-p) => x;};", "learn p:float in 0..=1=0.25;"), [1.0], {}, id="bounded-blend"),
    pytest.param(source("return blocks[0](blocks[0](x))+blocks[1](x);", BLOCK), [2.0], {}, id="shared-submodels"),
    pytest.param(source("return reshape(conv(reshape(slice(x,0,3),[1,3]),reshape(slice(x,3,1),[1,1,1]),[0],[1],[0],[1],1),[3]);", "", 4, 3), [1.0, 2.0, 3.0, 0.5], {}, id="input-generated-kernel"),
    pytest.param(source("return mean(x);", input_width=2), [3e38, 3e38], {}, id="mean-wide-accumulation"),
    pytest.param(source("return sum(x);", input_width=3), [3e38, 3e38, -3e38], {}, id="sum-wide-cancellation"),
    pytest.param(source("return layer_norm(x,[1,1],[0,0],0,0.0001);", input_width=2, output_width=2), [1e20, -1e20], {}, id="layer-norm-wide-variance"),
    pytest.param(source("return linear(x,w,b);", "parameter w:f32[1,2]=[1,1]\nparameter b:f32[1]=[1]", 2, 1), [1e8, -1e8], {}, id="linear-bias-first"),
    pytest.param(source("return reshape(matmul(reshape(x,[1,2]),reshape(w,[2,1])),[1]);", "parameter w:f32[2,1]=[2,-2]", 2, 1), [3e38, 3e38], {}, id="matmul-wide-products"),
    pytest.param(source("return reshape(conv(reshape(x,[1,2]),reshape(w,[1,1,2]),[0],[1],[0],[1],1),[1]);", "parameter w:f32[1,1,2]=[2,-2]", 2, 1), [3e38, 3e38], {}, id="conv-wide-products"),
    pytest.param(source("return reshape(avg_pool2d(reshape(x,[1,2,2]),[2,2],[2,2]),[1]);", input_width=4), [3e38] * 4, {}, id="pool-wide-accumulation"),
]


def compare_forward(text, inputs, bindings=None, gradient=None):
    expected = native(text, input=inputs, mode="forward", bindings=bindings or {}, gradient=gradient)
    model = wgt.loads(text, device="cpu")
    with torch.no_grad():
        for name, values in (bindings or {}).items():
            parameter = model.get_parameter(model.parameter_names[name])
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    x = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
    output = model(x)
    torch.testing.assert_close(output, torch.tensor(expected["output"]), atol=2e-5, rtol=5e-5)
    if output.requires_grad:
        output.backward(torch.ones_like(output) if gradient is None else torch.tensor(gradient))
    torch.testing.assert_close(
        x.grad if x.grad is not None else torch.zeros_like(x),
        torch.tensor(expected["input_gradient"]), atol=2e-5, rtol=5e-5,
    )
    for name, key in model.parameter_names.items():
        parameter = model.get_parameter(key)
        torch.testing.assert_close(parameter, original[key], rtol=0, atol=0)
        if parameter.requires_grad:
            actual = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            torch.testing.assert_close(
                actual.flatten(), torch.tensor(expected["gradients"][name]), atol=2e-5, rtol=5e-5,
            )
    return model


@pytest.mark.parametrize("text,inputs,bindings", CASES)
def test_forward_and_gradients_match_native(text, inputs, bindings):
    compare_forward(text, inputs, bindings)


@pytest.mark.parametrize("name,width", [
    ("attention", 6), ("cnn_1d", 10), ("cnn_2d", 30), ("cnn_3d", 48),
    ("residual_mlp", 2), ("recurrent_cell", 6), ("autoencoder", 3),
])
def test_existing_native_model_fixtures_match(name, width):
    text = (FIXTURES / name / "model.wgt").read_text()
    compare_forward(text, torch.linspace(-0.4, 0.6, width).tolist())


@pytest.mark.parametrize("text,inputs", [(WRITE, [[2.0, 3.0], [4.0, 3.0]]), (MEMORY, [[0.7, -0.2], [1.0, 2.0]])])
def test_committed_execution_matches_native(text, inputs):
    expected = native(text, inputs=inputs, mode="run")
    model = wgt.loads(text, device="cpu")
    for values, result in zip(inputs, expected["results"]):
        assert "error" not in result
        actual = model.run(torch.tensor(values))
        torch.testing.assert_close(actual, torch.tensor(result["output"]), atol=2e-5, rtol=5e-5)
        for name, values in result["parameters"].items():
            parameter = model.get_parameter(model.parameter_names[name])
            torch.testing.assert_close(parameter.flatten(), torch.tensor(values), atol=2e-5, rtol=5e-5)


@pytest.mark.parametrize("declaration,target,rate", [
    ("learn gain:float in 0..=1=0.5;", 10.0, 1.0),
    ("learn gain:float > 0=1;", -10.0, 1.0),
    ("learn gain:int in -3..=3=1;", 2.0, 0.6),
    ("learn gain:bool=false;", 1.0, 0.6),
])
def test_native_loss_sgd_and_projection(declaration, target, rate):
    text = source("return x*gain;", declaration, loss=True)
    expected = native(text, mode="train", inputs=[[1.0]], targets=[[target]], rate=rate)
    model = wgt.loads(text, device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    wgt.bind_optimizer(model, optimizer)
    loss = model.loss(model(torch.tensor([1.0])), torch.tensor([target]))
    torch.testing.assert_close(loss.reshape(()), torch.tensor(expected["loss"]), atol=1e-6, rtol=1e-6)
    loss.backward()
    optimizer.step()
    for name, values in expected["parameters"].items():
        parameter = model.get_parameter(model.parameter_names[name])
        torch.testing.assert_close(parameter.flatten(), torch.tensor(values), atol=0, rtol=1e-6)


def test_lazy_inference_skips_invalid_unused_branch():
    text = source("return route(p,x,linear(x,large));", "parameter p:f32[1]=[1]\nfrozen large:f32[1,1]=[1e30]")
    expected = native(text, mode="evaluate", input=[1e20])
    model = wgt.loads(text, device="cpu")
    with torch.inference_mode():
        torch.testing.assert_close(model(torch.tensor([1e20])), torch.tensor(expected["results"][0]["output"]))
    # Native differentiation deliberately evaluates both route candidates.
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        model(torch.tensor([1e20], requires_grad=True))


def test_forward_snapshot_survives_later_committed_write():
    model = wgt.loads(WRITE, device="cpu")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    output = model(x)
    model.run(torch.tensor([4.0, 3.0]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.5, 1.0]))
    memory = model.get_parameter(model.parameter_names["memory"])
    torch.testing.assert_close(memory.flatten(), torch.tensor([2.0]))
    torch.testing.assert_close(memory.grad.flatten(), torch.tensor([3.0]))


def test_nested_optimizer_binding_and_removal():
    module = wgt.loads(source("return x*gain;", "learn gain:float in 0..=1=0.5;"), device="cpu")
    model = torch.nn.Sequential(module, torch.nn.Identity())
    optimizer = torch.optim.SGD(model.parameters(), lr=10)
    handle = wgt.bind_optimizer(model, optimizer)
    (-model(torch.tensor([1.0]))).sum().backward()
    optimizer.step()
    gain = module.get_parameter(module.parameter_names["gain"])
    torch.testing.assert_close(gain, torch.tensor([1.0]))
    handle.remove()
    optimizer.step()
    assert gain.item() > 1.0
    with pytest.raises((ValueError, RuntimeError), match="domain|interval|bound|constraint|project"):
        model(torch.tensor([1.0]))


@pytest.mark.parametrize("text,rows", [
    (source("if positive(mean(x)) { return x*gain; } return -x;", "parameter gain:f32[2]=[2,3]", 2, 2), [[0.2, 0.3], [-0.2, -0.3], [0.0, 0.0]]),
    (source("return reshape(conv(reshape(slice(x,0,3),[1,3]),reshape(slice(x,3,1),[1,1,1]),[0],[1],[0],[1],1),[3]);", "", 4, 3), [[1.0, 2.0, 3.0, 0.5], [2.0, 3.0, 4.0, -2.0]]),
    (MEMORY, [[0.7, -0.2], [1.0, 2.0]]),
])
def test_dynamic_batches_keep_sample_paths_and_gradients_independent(text, rows):
    expected = [native(text, mode="forward", input=row) for row in rows]
    model = wgt.loads(text, device="cpu")
    x = torch.tensor(rows, requires_grad=True)
    output = model(x)
    torch.testing.assert_close(output, torch.tensor([item["output"] for item in expected]), atol=2e-5, rtol=5e-5)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([item["input_gradient"] for item in expected]), atol=2e-5, rtol=5e-5)
    for name, key in model.parameter_names.items():
        parameter = model.get_parameter(key)
        if parameter.requires_grad:
            gradients = torch.tensor([item["gradients"][name] for item in expected]).sum(dim=0)
            actual = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            torch.testing.assert_close(actual.flatten(), gradients, atol=2e-5, rtol=5e-5)


def test_typed_transactions_and_failure_rollback_match_native(tmp_path, monkeypatch):
    text = """wgt 0.5
model Counter
parameter gain:f32[1]=[2]
state count:f32[1]=[0]
fn advance(step:f32[1], factor:f32[1]) -> f32[1] {
    count <- count+step;
    gain <- gain+factor;
    return count*gain;
}
fn fail(step:f32[1]) -> f32[1] {
    count <- count+step;
    gain <- gain+step;
    return exp([1000]);
}
fn exhaust() { count <- [99]; while true {} }
fn observe() -> f32[1] { return count; }
fn reset() { count <- [0]; }
"""
    calls = [
        {"name": "advance", "inputs": [[1.0], [0.5]]},
        {"name": "fail", "inputs": [[10.0]]},
        {"name": "exhaust", "inputs": [], "max_work": 50},
        {"name": "observe", "inputs": []},
        {"name": "advance", "inputs": [[2.0], [0.5]]},
        {"name": "reset", "inputs": []},
        {"name": "observe", "inputs": []},
    ]
    expected = native(text, calls=calls)
    model = wgt.loads(text, device="cpu")
    for request, result in zip(calls, expected["results"]):
        model.max_work = request.get("max_work", 1_000_000)
        args = [torch.tensor(row) for row in request["inputs"]]
        if "error" in result:
            with pytest.raises((ValueError, RuntimeError)):
                model.call(request["name"], *args)
        else:
            actual = model.call(request["name"], *args)
            if result["output"] is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, torch.tensor(result["output"]))
        for name, values in result["states"].items():
            torch.testing.assert_close(model.get_buffer(model.state_names[name]).flatten(), torch.tensor(values))
        for name, values in result["parameters"].items():
            torch.testing.assert_close(model.get_parameter(model.parameter_names[name]).flatten(), torch.tensor(values))
    model.advance(torch.tensor([4.0]), torch.tensor([0.5]))
    saved = wgt.save(model, tmp_path / "counter.pt")
    monkeypatch.setenv("WGT_COMPILER", str(tmp_path / "absent"))
    restored = wgt.load(saved, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(restored.observe(), torch.tensor([4.0], dtype=torch.float64))
    restored.advance(torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64))
    torch.testing.assert_close(model.observe(), torch.tensor([4.0]))
    torch.testing.assert_close(restored.observe(), torch.tensor([5.0], dtype=torch.float64))


def test_typed_learn_writes_validate_the_final_transaction():
    text = """wgt 0.5
model TypedLearn
learn gain:int in 0..=3=1;
fn set(value:f32[1]) { gain <- value; }
fn bounce(value:f32[1]) -> f32[1] {
    gain <- value;
    let candidate=gain;
    gain <- [1];
    return candidate;
}
fn observe() -> f32[1] { return gain; }
"""
    calls = [
        {"name": "set", "inputs": [[2.0]]},
        {"name": "set", "inputs": [[2.5]]},
        {"name": "bounce", "inputs": [[4.5]]},
        {"name": "observe", "inputs": []},
    ]
    expected = native(text, calls=calls)
    model = wgt.loads(text, device="cpu")
    for request, result in zip(calls, expected["results"]):
        args = [torch.tensor(row) for row in request["inputs"]]
        if "error" in result:
            with pytest.raises((ValueError, RuntimeError)):
                model.call(request["name"], *args)
        else:
            actual = model.call(request["name"], *args)
            if result["output"] is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, torch.tensor(result["output"]))
        for name, values in result["parameters"].items():
            torch.testing.assert_close(model.get_parameter(model.parameter_names[name]).flatten(), torch.tensor(values))


@pytest.mark.parametrize("body", [
    "return resource_state();",
    "let bank=memory_alloc(3,2); let snapshot=resource_state(); memory_free(bank); return snapshot;",
])
def test_resource_state_and_instruction_budget_match_native(body):
    text = source(body, input_width=1, output_width=3)
    expected = native(text, input=[1.0], mode="evaluate", max_work=100, memory_bytes=4096)["results"][0]
    model = wgt.loads(text, device="cpu")
    actual = model.evaluate(torch.tensor([1.0]), max_work=100, memory_bytes=4096)
    torch.testing.assert_close(actual, torch.tensor(expected["output"]), rtol=0, atol=0)
    assert model.last_execution["work"] == expected["steps"]


def test_sequential_initialization_matches_native_without_touching_host_rng():
    text = """wgt 0.5
model Seeded
input x:f32[2]
output y:f32[1]
layer hidden:linear(3)
layer activation:tanh
layer output:linear(1)
init seed=7
"""
    expected = native(text, input=[0.2, -0.3], target=[0.7])
    before = torch.random.get_rng_state()
    model = wgt.loads(text, device="cpu")
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    for name, values in expected["parameters"].items():
        parameter = model.get_parameter(model.parameter_names[name])
        torch.testing.assert_close(parameter.flatten(), torch.tensor(values), rtol=0, atol=0)
    prediction = model(torch.tensor([0.2, -0.3]))
    torch.testing.assert_close(prediction, torch.tensor(expected["output"]), atol=1e-6, rtol=1e-6)
    loss = torch.nn.functional.mse_loss(prediction, torch.tensor([0.7]))
    loss.backward()
    for name, values in expected["gradients"].items():
        parameter = model.get_parameter(model.parameter_names[name])
        torch.testing.assert_close(parameter.grad.flatten(), torch.tensor(values), atol=1e-6, rtol=1e-6)


def test_pure_fx_rejects_nonfinite_input_and_intermediate_values():
    model = wgt.loads(source("return exp(x);"), device="cpu")
    assert isinstance(model, torch.fx.GraphModule)
    for values in ([float("nan")], [1000.0]):
        with pytest.raises((ValueError, RuntimeError), match="finite"):
            model(torch.tensor(values))


def test_softmax_backward_uses_the_stored_forward_values():
    text = source("return softmax(scale(x,100000000));", input_width=2, output_width=2)
    compare_forward(text, [-1.04e-6, 0.0], gradient=[3e38, 0.0])


@pytest.mark.parametrize("seed", [1.0, 1.401298464324817e-45])
def test_log_backward_rejects_native_nonfinite_reciprocal(seed):
    text = source("return log(x);")
    value = 1.401298464324817e-45
    native(text, input=[value], gradient=[seed], error="nonfinite")
    model = wgt.loads(text, device="cpu")
    x = torch.tensor([value], requires_grad=True)
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        model(x).backward(torch.tensor([seed]))


def test_zero_upstream_gradient_skips_singular_log_derivative():
    compare_forward(source("return log(x);"), [1.401298464324817e-45], gradient=[0.0])


def test_matmul_backward_preserves_native_overflow_detection():
    text = source("return reshape(matmul(reshape(x,[1,1]),reshape([2,-2],[1,2])),[2]);", output_width=2)
    native(text, input=[1.0], gradient=[3e38, 3e38], error="nonfinite")
    model = wgt.loads(text, device="cpu")
    x = torch.tensor([1.0], requires_grad=True)
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        model(x).backward(torch.tensor([3e38, 3e38]))


@pytest.mark.parametrize("text", [
    source("return log(tiny)*x;", "frozen tiny:f32[1]=[1.401298464324817e-45]"),
    source("return log([1.401298464324817e-45]);"),
])
def test_internal_frozen_and_constant_gradients_keep_native_diagnostics(text):
    native(text, input=[1.0], error="nonfinite")
    model = wgt.loads(text, device="cpu")
    x = torch.tensor([1.0], requires_grad=True)
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        model(x).sum().backward()
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


def test_concurrent_program_transactions_do_not_lose_updates():
    text = source("write(count,count+x); return count;", "parameter count:f32[1]=[0]")
    model = wgt.loads(text, device="cpu")
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _: model.run(torch.ones(1)), range(32)))
    torch.testing.assert_close(model.get_parameter(model.parameter_names["count"]), torch.tensor([32.0]))
    assert sorted(output.item() for output in outputs) == list(range(1, 33))
    assert model.revision == 32
