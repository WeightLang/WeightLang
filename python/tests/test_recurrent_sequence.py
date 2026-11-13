"""Cross-call hidden-state VJPs against the independent native tape backend."""
import json
import os
from pathlib import Path
import subprocess
import pytest
import torch
from weightlang import torch_api as wgt

ROOT=Path(__file__).resolve().parents[2]
SOURCE='''wgt 0.5
model Recurrent
input x:f32[2]
output y:f32[1]
parameter gain:f32[1]=[0.5]
fn forward(x){return x[0:1]+x[1:2]*gain;}
'''
INPUTS=[[1.,.4],[2.,99.],[3.,-99.]]

def oracle(mode,optimizer='sgd',steps=1):
    executable=os.environ.get('WGT_REFERENCE',str(ROOT/'target/debug/examples/torch_reference'))
    process=subprocess.run([executable],input=json.dumps({
        'source':SOURCE,'mode':mode,'inputs':INPUTS,'sequence_targets':[None,None,[0.]],
        'state_link':{'input_offset':1,'output_offset':0,'width':1},
        'supervised':{'offset':0,'width':1},'rate':.001,'optimizer':optimizer,'steps':steps,
    }),text=True,capture_output=True,timeout=30)
    assert process.returncode==0,process.stderr
    return json.loads(process.stdout)

def unroll(model,x):
    hidden=x[0,1:2]
    outputs=[]
    for step in range(len(x)):
        hidden=model(torch.cat((x[step,:1],hidden)))
        outputs.append(hidden)
    return torch.stack(outputs)

def test_native_cross_call_input_and_parameter_gradients():
    model=wgt.loads(SOURCE,device='cpu')
    x=torch.tensor(INPUTS,requires_grad=True)
    output=unroll(model,x)
    loss=output[-1].square().mean()
    loss.backward()
    expected=oracle('recurrent')
    torch.testing.assert_close(output,torch.tensor(expected['outputs']))
    torch.testing.assert_close(loss,torch.tensor(expected['loss']))
    torch.testing.assert_close(x.grad,torch.tensor(expected['input_gradients']))
    for name,attr in model.parameter_names.items():
        torch.testing.assert_close(model.get_parameter(attr).grad,torch.tensor(expected['parameter_gradients'][name]))
    # Hidden placeholders supplied after step zero are not free graph inputs.
    torch.testing.assert_close(x.grad[1:,1],torch.zeros(2))

@pytest.mark.parametrize('optimizer',['sgd','adam'])
def test_native_recurrent_optimizer_preserves_history_across_sequences(optimizer):
    model=wgt.loads(SOURCE,device='cpu')
    opt=getattr(torch.optim,'SGD' if optimizer=='sgd' else 'Adam')(model.parameters(),lr=.001)
    for _ in range(4):
        opt.zero_grad()
        loss=unroll(model,torch.tensor(INPUTS))[-1].square().mean()
        loss.backward();opt.step()
    expected=oracle('train_recurrent',optimizer,4)
    torch.testing.assert_close(loss.detach(),torch.tensor(expected['loss']),atol=2e-5,rtol=2e-5)
    for name,attr in model.parameter_names.items():
        torch.testing.assert_close(model.get_parameter(attr),torch.tensor(expected['parameters'][name]),atol=2e-6,rtol=2e-5)
