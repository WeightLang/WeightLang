"""Runtime indexing: native differential, address stop gradients and batching."""
from copy import deepcopy

import pytest
import torch

from weightlang import ops
from weightlang.program import ProgramExecutor
from test_native_parity import compare_forward, native, native_tools, source


@pytest.mark.parametrize("body,declarations,width,output", [
    ("return gather(x,[2,0,2]);", "", 4, 3),
    ("return gather(x as [2,3],ids,1) as [6];", "parameter ids:f32[3]=[2,0,2]", 6, 6),
    ("return gather((x as [3,2]).T,[2,0,2],1) as [6];", "", 6, 6),
    ("return gather(table as [3,2],x) as [6];", "parameter table:f32[3,2]=[1,2,3,4,5,6]", 3, 6),
    ("return dynamic_slice(x,start,2);", "parameter start:f32[1]=[2]", 5, 2),
    ("return concat(x,x);", "", 3, 6),
    ("return concat(x as [2,2],x as [2,2],1) as [8];", "", 4, 8),
    ("return gather(x as [1,1,1,1,1,1,1,2],[1,0,1],7) as [3];", "", 2, 3),
])
def test_native_python_indexing_vjp(body, declarations, width, output):
    values = [2., 0., 2.] if "gather(table" in body else [float(i + 1) for i in range(width)]
    compare_forward(source(body, declarations, width, output), values)


@pytest.mark.parametrize("bad", [-1., 0.5, 3., float("inf"), float("nan")])
def test_tensor_addresses_reject_fractional_nonfinite_and_out_of_range(bad):
    with pytest.raises(ValueError, match="runtime index"):
        ops.gather(torch.ones(3), torch.tensor([0., bad]))
    with pytest.raises(ValueError, match="runtime index"):
        ops.dynamic_slice(torch.ones(4), torch.tensor([bad]), 2)


def test_batch_specific_ids_shared_table_and_gradients():
    table = torch.arange(8., requires_grad=True)
    indices = torch.tensor([[3., 1., 3.], [0., 2., 0.]], requires_grad=True)
    actual = ops.gather(table.reshape(4, 2), indices, 0, 2)
    assert actual.shape == (2, 3, 2)
    actual.sum().backward()
    torch.testing.assert_close(table.grad, torch.tensor([2.,2.,1.,1.,1.,1.,2.,2.]))
    assert indices.grad is None
    starts = torch.tensor([[0.], [2.]], requires_grad=True)
    values = torch.arange(10.).reshape(2,5).requires_grad_()
    out = ops.dynamic_slice(values, starts, 3)
    torch.testing.assert_close(out, torch.tensor([[0.,1.,2.],[7.,8.,9.]]))
    out.sum().backward()
    assert starts.grad is None
    torch.testing.assert_close(values.grad, torch.tensor([[1.,1.,1.,0.,0.],[0.,0.,1.,1.,1.]]))


def test_concat_batch_shared_branch_alias_gradients():
    x = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    shared = torch.tensor([5.,6.], requires_grad=True)
    result = ops.concat(x,shared,0,1)
    torch.testing.assert_close(result,torch.tensor([[1.,2.,5.,6.],[3.,4.,5.,6.]]))
    result.sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x))
    torch.testing.assert_close(shared.grad,torch.tensor([2.,2.]))


def test_untrusted_ir_rejects_index_shape_and_version():
    program = {"version":3,"input":"x","registers":[
        {"name":"x","shape":[3],"len":3},
        {"name":"ids","shape":[2],"len":2},
        {"name":"y","shape":[2],"len":2}],"parameters":[],"instructions":[
            {"op":"constant","out":"ids","data":[0.,2.]},
            {"op":"gather","out":"y","input":"x","indices":"ids","axis":0},
            {"op":"return","value":"y"}]}
    ProgramExecutor(program)
    for change in ["version", "axis", "indices"]:
        invalid = deepcopy(program)
        if change == "version":
            invalid["version"] = 2
        elif change == "axis":
            invalid["instructions"][1]["axis"] = 1
        else:
            invalid["registers"][1]["shape"] = [1,2]
        with pytest.raises(ValueError):
            ProgramExecutor(invalid)


def test_executor_and_fx_consume_same_indexing_ir():
    """Exercise both Python consumers even before a compiler binary is rebuilt."""
    from weightlang.runtime import build_module
    registers = [("x", [4]), ("ids", [3]), ("picked", [3]),
                 ("start", [1]), ("window", [2]), ("y", [5])]
    program = {"version": 3, "input": "x", "parameters": [],
               "registers": [{"name": name, "shape": shape, "len": shape[0]} for name, shape in registers],
               "instructions": [
                   {"op": "constant", "out": "ids", "data": [3., 0., 3.]},
                   {"op": "gather", "out": "picked", "input": "x", "indices": "ids", "axis": 0},
                   {"op": "constant", "out": "start", "data": [1.]},
                   {"op": "dynamic_slice", "out": "window", "input": "x", "start": "start", "length": 2},
                   {"op": "concat", "out": "y", "left": "picked", "right": "window", "axis": 0},
                   {"op": "return", "value": "y"}]}
    definition = {"format": "weightlang.torch.v1", "kind": "forward", "name": "Index",
                  "interface": {"input": 4, "output": 5}, "program": program}
    model = build_module(definition)
    x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    fx_output = model(x)
    eager_output = ProgramExecutor(program).run(x, {}, differentiable=True).output
    torch.testing.assert_close(fx_output, torch.tensor([4., 1., 4., 2., 3.]))
    torch.testing.assert_close(eager_output, fx_output)
    for output in [fx_output, eager_output]:
        torch.testing.assert_close(torch.autograd.grad(output.sum(), x, retain_graph=True)[0], torch.tensor([1., 1., 1., 2.]))
    # FX accepts host batching while preserving each model's logical axes.
    torch.testing.assert_close(model(torch.stack((x, x + 10))), torch.stack((fx_output, fx_output + 10)))
