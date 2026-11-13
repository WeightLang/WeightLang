"""First-class WGT objects against a real, isolated wgtime service."""
import os
from pathlib import Path
import subprocess
import time

import pytest
import torch
import weightlang as wgt
from weightlang.model import NativeConnection

SOURCE = '''wgt 0.5
model Scale
input x:f32[1]
output y:f32[1]
parameter gain:f32[1]=[2]
fn forward(x){return x*gain;}
'''

@pytest.fixture
def service(tmp_path, monkeypatch):
    daemon = os.environ.get("WGTIME_TEST_DAEMON")
    if not daemon:
        pytest.skip("requires a prebuilt real WGTIME_TEST_DAEMON")
    import tempfile
    scratch=tempfile.TemporaryDirectory(prefix='wgt-model-',dir='/tmp/weightlang')
    directory=Path(scratch.name)
    process = subprocess.Popen([daemon,'--test-directory',str(directory)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    endpoint=directory/'service.sock'
    try:
        for _ in range(100):
            if endpoint.exists():break
            if process.poll() is not None:raise AssertionError(process.stderr.read().decode())
            time.sleep(.02)
        assert endpoint.exists()
        monkeypatch.setenv('WGTIME_ENDPOINT',str(endpoint))
        yield endpoint
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stderr.close()
        scratch.cleanup()


def test_default_native_lifecycle_optimizer_and_parameter_identity(service):
    with wgt.loads(SOURCE,optimizer='sgd') as model:
        assert isinstance(model,wgt.Model) and model.backend=='native'
        assert not isinstance(model,torch.nn.Module)
        assert model([3.])==[6.]
        assert model.parameters()[0].data==(2.,)
        loss=model.train_step([[1.]],[[0.]],rate=.1)
        assert loss==4.
        assert model([1.])[0]==pytest.approx(1.6)
        with pytest.raises(wgt.ServiceError):model([1.,2.])
        assert model([1.])[0]==pytest.approx(1.6)  # complete errors do not poison
        original=model.parameters()
        with pytest.raises(wgt.ServiceError):model.set_parameters([wgt.Parameter('unknown',(1,),(9.,))])
        assert model.parameters()==original
    model.close()
    with pytest.raises(ConnectionError):model([1.])


def test_explicit_torch_object_and_custom_native_subclass(service):
    model=wgt.loads(SOURCE,backend='torch',device='cpu')
    assert isinstance(model,wgt.Model) and model.backend=='torch'
    assert isinstance(model.to_torch(),torch.nn.Module)
    model(torch.ones(1,requires_grad=True)).sum().backward()
    assert next(model.parameters()).grad is not None
    class Custom(wgt.Model):
        def forward(self,value):return super().forward([value[0]+1])
    with Custom.from_source(SOURCE) as native:
        assert native([2.])==[6.]
        with pytest.raises(TypeError,match='autograd'):native(torch.ones(1,requires_grad=True))


def test_native_rejects_torch_dialect_without_connecting():
    with pytest.raises(ValueError,match='target torch'):
        wgt.from_ir({'kind':'torch_module','source':'not executed'})
    with pytest.raises(ValueError,match='backend'):
        wgt.loads(SOURCE,backend='auto')


def test_native_dynamic_ports_and_optimizer_continuity(service):
    source='''wgt 0.5
    dim B=2
    model Hidden
    parameter gain:f32[1]=[0.5]
    fn forward(x:f32[B,1],hidden:f32[B,1])->(f32[B,1],f32[B,1]){
      return (scale(x,gain)+hidden,scale(hidden,gain)+x);
    }
    '''
    with wgt.loads_dynamic(source,input_shapes={'x':(2,1),'hidden':(2,1)},optimizer='adam') as model:
        before=model.parameters()
        assert model([[2.],[4.]],[[0.],[0.]])==([[1.],[2.]],[[2.],[4.]])
        assert model([[6.]],[[0.]])==([[3.]],[[6.]])
        assert model.parameters()==before
        model.train_step([[2.,0.]],[[0.,0.]],rate=.01)
        assert model.parameters()!=before
        assert len(model([[2.],[4.]],[[0.],[0.]])[0])==2


def test_uid_check_and_closed_client(service):
    with pytest.raises(PermissionError):NativeConnection(service,expected_uid=os.geteuid()+1)
    client=NativeConnection(service);client.close()
    with pytest.raises(ConnectionError):client.call('Stats')


def test_native_import_needs_no_pytorch():
    import sys
    program='''
import sys
class NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch','safetensors'}:
            raise AssertionError('native import attempted ' + name)
sys.meta_path.insert(0,NoTorch())
import weightlang
assert weightlang.Model and 'torch' not in sys.modules
from weightlang import *
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',program],check=True,env=os.environ.copy())


def test_foreign_process_is_rejected_before_inherited_locks(service):
    class InheritedLock:
        def __enter__(self):raise AssertionError('must not acquire inherited locks')
        def __exit__(self,*args):pass
    model=wgt.loads(SOURCE)
    model._client._pid=-1
    model._lock=InheritedLock()
    model._client._lock=InheritedLock()
    with pytest.raises(ConnectionError,match='fork'):model([1.])
    with pytest.raises(ConnectionError,match='fork'):model._client.call('Stats')
    model.close()  # closes only this process's descriptor without acquiring locks
    model.close()


def test_metadata_uses_authoritative_service_ports(service):
    source='''wgt 0.5
    model Ports
    fn forward(x:f32[2])->(f32[2],){return (x,);}
    '''
    definition=wgt.compile_source(source)
    definition['forward_interface']['inputs'][0]['shape']=[1,2]
    with wgt.from_ir(definition) as model:
        assert model([1.,2.])==([1.,2.],)
        with pytest.raises(ValueError,match='shape'):model([[1.,2.]])


def test_parameter_replacement_requires_explicit_optimizer_reset(service):
    with wgt.loads(SOURCE,optimizer='adam') as model:
        model.set_parameters([wgt.Parameter('gain',(1,),(3.,))])
        with pytest.raises(wgt.ServiceError,match='optimizer'):model.train_step([[1.]],[[0.]],rate=.1)
        model.set_optimizer('sgd')
        assert model.train_step([[1.]],[[0.]],rate=.1)==9.
        assert model([1.])[0]==pytest.approx(2.4)
        before=model.parameters()
        with pytest.raises(ValueError):model.set_optimizer('unsupported')
        assert model.parameters()==before


@pytest.mark.parametrize('bad',[
    b'{"value":null,"error":null}',
    b'{"value":{},"error":null,"error":null}',
    b'{"value":{"bad":NaN},"error":null}',
    b'oversize',b'truncated',
])
def test_incomplete_or_malformed_reply_closes_without_replay(bad):
    import json
    import socket
    import struct
    import tempfile
    import threading
    with tempfile.TemporaryDirectory(prefix='wgt-wire-',dir='/tmp/weightlang') as directory:
        path=Path(directory)/'socket'
        listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        listener.bind(str(path));listener.listen(1)
        requests=[]
        def read(stream,count):
            result=b''
            while len(result)<count:
                block=stream.recv(count-len(result))
                if not block:raise ConnectionError('incomplete test request')
                result+=block
            return result
        def respond():
            with listener.accept()[0] as stream:
                for index in range(2):
                    size,=struct.unpack('<I',read(stream,4))
                    requests.append(json.loads(read(stream,size)))
                    if index==0:
                        data=b'{"value":{"protocol":1},"error":null}'
                        stream.sendall(struct.pack('<I',len(data))+data)
                    elif bad==b'oversize':stream.sendall(struct.pack('<I',NativeConnection.MAX_FRAME+1))
                    elif bad==b'truncated':stream.sendall(struct.pack('<I',10)+b'{')
                    else:stream.sendall(struct.pack('<I',len(bad))+bad)
        worker=threading.Thread(target=respond,daemon=True);worker.start()
        try:
            client=NativeConnection(path)
            with pytest.raises((ValueError,ConnectionError)):client.call('Stats')
            with pytest.raises(ConnectionError):client.call('Stats')
            client.close();worker.join(timeout=3)
            assert not worker.is_alive()
            assert len(requests)==2 and requests[1]=='Stats'
        finally:listener.close()


def test_compilation_default_is_wgt_without_creating_a_service():
    definition=wgt.compile_source(SOURCE)
    assert definition['format']=='weightlang.wgt.v1'
    assert definition['kind']=='forward'
    torch_definition=wgt.compile_source(SOURCE,target='torch')
    assert torch_definition['format']=='weightlang.torch.v1'


def test_native_typed_state_is_owned_and_not_differentiable(service):
    source='''wgt 0.5
    model Counter
    state value:f32[1]=[0]
    fn tick(x:f32[1])->f32[1]{value <- value+x;return value;}
    '''
    with wgt.loads(source) as model:
        assert model.call('tick',[2.])==[2.]
        with pytest.raises(wgt.ServiceError):model.call('tick',[1.,2.])
        assert model.call('tick',[3.])==[5.]
        with pytest.raises(wgt.ServiceError):model.train_step([[1.]],[[0.]],rate=.1)
