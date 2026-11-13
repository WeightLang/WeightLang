"""Embedding shares native row indexing; padding affects gradients only."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from weightlang import ops
from weightlang.program import ProgramExecutor
from weightlang.runtime import build_module
from test_native_parity import compare_forward, native, native_tools, source


def definition(padding=0):
    shapes = [("ids",[4]),("flat",[6]),("table",[3,2]),("rows",[4,2]),("y",[8])]
    return {"format":"weightlang.torch.v1","kind":"forward","name":"Embedding",
            "interface":{"input":4,"output":8},"program":{
                "version":3,"input":"ids","parameters":[
                    {"name":"table","shape":[3,2],"data":[11.,12.,21.,22.,31.,32.]}],
                "registers":[{"name":name,"shape":shape,"len":torch.Size(shape).numel()} for name,shape in shapes],
                "instructions":[
                    {"op":"read","out":"flat","parameter":"table"},
                    {"op":"reshape","out":"table","input":"flat"},
                    {"op":"embedding","out":"rows","table":"table","indices":"ids","padding_idx":padding},
                    {"op":"reshape","out":"y","input":"rows"},
                    {"op":"return","value":"y"}]}}


@pytest.mark.parametrize("padding",[None,0,1,2])
def test_embedding_program_and_fx_match_torch_with_nonzero_padding(padding):
    exported=definition(padding)
    model=build_module(exported)
    ids=torch.tensor([0.,2.,2.,0.],requires_grad=True)
    table=torch.tensor([[11.,12.],[21.,22.],[31.,32.]],requires_grad=True)
    expected=F.embedding(ids.detach().long(),table,padding_idx=padding).flatten()
    gradient=torch.arange(1.,9.)
    expected_grad=torch.autograd.grad(expected,table,gradient)[0]
    actual=model(ids)
    torch.testing.assert_close(actual,expected)
    # Padding rows retain their nonzero original values.
    actual.backward(gradient)
    assert ids.grad is None or torch.count_nonzero(ids.grad)==0
    registered=model.get_parameter(model.parameter_names["table"])
    torch.testing.assert_close(registered.grad,expected_grad)
    other_table=table.detach().clone().requires_grad_()
    eager=ProgramExecutor(exported["program"]).run(ids,{"table":other_table},differentiable=True).output
    torch.testing.assert_close(eager,expected)
    torch.testing.assert_close(torch.autograd.grad(eager,other_table,gradient)[0],expected_grad)
    batched=model(torch.tensor([[0.,2.,2.,0.],[1.,0.,1.,1.]]))
    torch.testing.assert_close(batched,F.embedding(torch.tensor([[0,2,2,0],[1,0,1,1]]),table,padding_idx=padding).reshape(2,8))


@pytest.mark.parametrize("shape",[(1,1),(1,3),(3,1),(3,2),(5,4)])
def test_embedding_rank_two_shapes_shared_and_batched_tables(shape):
    rows,width=shape
    for padding in [None,0,rows-1]:
        table=(torch.arange(rows*width,dtype=torch.float32)+1).reshape(shape).requires_grad_()
        ids=torch.tensor([0.,float(rows-1),float(rows-1)],requires_grad=True)
        actual=ops.embedding(table,ids,padding)
        expected=F.embedding(ids.detach().long(),table,padding_idx=padding)
        torch.testing.assert_close(actual,expected)
        grad=torch.arange(actual.numel(),dtype=torch.float32).reshape(actual.shape)+1
        torch.testing.assert_close(torch.autograd.grad(actual,table,grad)[0],torch.autograd.grad(expected,table,grad)[0])
        assert ids.grad is None
        batch=(torch.stack([table,table+10])).detach().requires_grad_()
        batch_ids=torch.stack([ids,ids.flip(0)])
        output=ops.embedding(batch,batch_ids,padding)
        oracle=torch.stack([F.embedding(batch_ids[i].long(),batch[i],padding_idx=padding) for i in range(2)])
        torch.testing.assert_close(output,oracle)
        torch.testing.assert_close(torch.autograd.grad(output.sum(),batch)[0],torch.autograd.grad(oracle.sum(),batch)[0])


def test_embedding_default_and_generic_gather_stay_unmasked():
    table=torch.tensor([[11.,12.],[21.,22.],[31.,32.]],requires_grad=True)
    ids=torch.tensor([0.,2.,2.,0.],requires_grad=True)
    for result in [ops.embedding(table,ids),ops.gather(table,ids,0)]:
        torch.testing.assert_close(torch.autograd.grad(result.sum(),table)[0],torch.tensor([[2.,2.],[0.,0.],[2.,2.]]))
    exported=definition()
    del exported["program"]["instructions"][2]["padding_idx"]
    model=build_module(exported)
    model(ids).sum().backward()
    torch.testing.assert_close(model.get_parameter(model.parameter_names["table"]).grad,torch.tensor([[2.,2.],[0.,0.],[2.,2.]]))


@pytest.mark.parametrize("bad",[-1.,0.5,3.,float("inf"),float("nan")])
def test_embedding_strict_runtime_index_validation(bad):
    with pytest.raises(ValueError,match="runtime index"):
        ops.embedding(torch.ones(3,2),torch.tensor([0.,bad]),0)


def test_embedding_untrusted_ir_shape_padding_and_version():
    for kind in ["padding","fractional","negative","boolean","version","indices","table"]:
        exported=definition()
        program=exported["program"]
        instruction=program["instructions"][2]
        if kind in {"padding","fractional","negative","boolean"}:
            instruction["padding_idx"]={"padding":3,"fractional":0.5,"negative":-1,"boolean":True}[kind]
        elif kind=="version":
            program["version"]=2
        elif kind=="indices":
            program["registers"][0]["shape"]=[2,2]
        else:
            program["registers"][2]["shape"]=[6]
        with pytest.raises(ValueError):
            ProgramExecutor(program)


@pytest.mark.parametrize("padding",[None,-3,-2,-1,0,1,2])
def test_native_embedding_padding_vjp(padding):
    suffix="" if padding is None else f",{padding}"
    text=source(f"return embedding(table as [3,2],x{suffix}) as [8];",
                "parameter table:f32[3,2]=[11,12,21,22,31,32]",4,8)
    compare_forward(text,[0.,2.,2.,0.],gradient=list(range(1,9)))
    expected=native(text,input=[0.,2.,2.,0.],mode="forward",gradient=list(range(1,9)))
    table=torch.tensor([[11.,12.],[21.,22.],[31.,32.]],requires_grad=True)
    result=F.embedding(torch.tensor([0,2,2,0]),table,padding_idx=padding)
    torch.testing.assert_close(result.flatten(),torch.tensor(expected["output"]))
    torch.testing.assert_close(torch.autograd.grad(result.flatten(),table,torch.arange(1.,9.))[0].flatten(),torch.tensor(expected["gradients"]["table"]))
    assert expected["input_gradient"]==[0.]*4


def test_native_embedding_strided_runtime_table_and_parameter_indices():
    text=source("return embedding((x as [2,3]).T,ids,0) as [6];","parameter ids:f32[3]=[2,0,2]",6,6)
    compare_forward(text,[1.,2.,3.,4.,5.,6.])


@pytest.mark.parametrize("padding",[-3,-2,-1])
def test_negative_padding_matches_positive_row_and_torch(padding):
    table=torch.tensor([[11.,12.],[21.,22.],[31.,32.]],requires_grad=True)
    ids=torch.tensor([0.,2.,2.,0.],requires_grad=True)
    result=ops.embedding(table,ids,padding)
    positive=ops.embedding(table,ids,padding+3)
    expected=F.embedding(ids.detach().long(),table,padding_idx=padding)
    torch.testing.assert_close(result,positive)
    torch.testing.assert_close(result,expected)
    gradient=torch.arange(1.,9.).reshape(4,2)
    actual_grad=torch.autograd.grad(result,table,gradient)[0]
    torch.testing.assert_close(actual_grad,torch.autograd.grad(positive,table,gradient)[0])
    torch.testing.assert_close(actual_grad,torch.autograd.grad(expected,table,gradient)[0])
    assert torch.count_nonzero(actual_grad[padding+3])==0
    assert ids.grad is None


@pytest.mark.parametrize("padding",[-4,3,0.5,True])
def test_embedding_rejects_invalid_signed_padding(padding):
    with pytest.raises(ValueError,match="padding_idx"):
        ops.embedding(torch.ones(3,2),torch.tensor([0.]),padding)
