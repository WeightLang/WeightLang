import json
from pathlib import Path

import pytest

from weightlang.native import NativeSession, available_devices
from test_native_backend import BINARY

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build native runtime first")


def test_native_controller_changes_target_without_torch_autograd():
    source = ROOT / 'examples/parameter_updates/model.wgt'
    # The second native model is a programmer: its output is the next target weight.
    with NativeSession.from_source(source, {'projection.weight': [1., 2.]},
                                   budget_bytes=128, executable=BINARY, device='cpu') as target, \
         NativeSession(ROOT / 'examples/native_backend/model.json', budget_bytes=128,
                       executable=BINARY, device='cpu') as controller:
        assert target([1., 1.]) == [3.]
        proposal = controller([0.1, 0.2])
        assert target.write_parameters([dict(name='projection.weight', values=proposal)]) == 1
        assert target.read_parameter('projection.weight') == pytest.approx(proposal)
        assert target([1., 1.]) == pytest.approx([sum(proposal)])
        assert target.parameter_info() == [dict(name='projection.weight', shape=[1,2], writable=True)]
        with pytest.raises(ValueError, match='bounds'):
            target.write_parameters([dict(name='projection.weight', values=[9.]),
                                     dict(name='projection.weight', start=2, values=[8.])])
        assert target([1., 1.]) == pytest.approx([sum(proposal)])
        with pytest.raises(ValueError, match='overlapping'):
            target.write_parameters([dict(name='projection.weight', values=[9.]),
                                     dict(name='projection.weight', values=[8.])])
        assert target.read_parameter('projection.weight', start=1, length=1) == pytest.approx(proposal[1:])
        with pytest.raises(ValueError, match='bounds'):
            target.read_parameter('projection.weight', length=99999999)
        assert target.write_parameters([]) == 1


def test_native_frozen_and_read_only_file(tmp_path):
    source = tmp_path / 'model.wgt'
    source.write_text((ROOT / 'examples/parameter_updates/model.wgt').read_text() + 'freeze projection\n')
    with NativeSession.from_source(source, {'projection.weight': [1.,2.]}, budget_bytes=128,
                                   executable=BINARY, device='cpu') as session:
        assert not session.parameter_info()[0]['writable']
        with pytest.raises(ValueError, match='frozen'):
            session.write_parameters([dict(name='projection.weight', values=[0.,0.])])
        assert session([1., 1.]) == [3.]
    import struct
    file = tmp_path / 'weights.f32'
    raw = struct.pack('<2f', 1., 2.)
    file.write_bytes(raw)
    source.write_text(source.read_text().replace('freeze projection', ''))
    with NativeSession.from_source(source, {'projection.weight': dict(kind='file', path=str(file), offset=0)},
                                   budget_bytes=128, executable=BINARY, device='cpu') as session:
        assert session.read_parameter('projection.weight') == [1., 2.]
        with pytest.raises(ValueError, match='read-only'):
            session.write_parameters([dict(name='projection.weight', values=[0.,0.])])
        assert session([1., 1.]) == [3.]
    assert file.read_bytes() == raw


def test_updates_reach_cached_and_streamed_gpu_weights(tmp_path):
    devices = [d['id'] for d in available_devices(executable=BINARY) if d['backend'] != 'cpu']
    if not devices:
        pytest.skip('no GPU adapter')
    import struct
    model = json.loads((ROOT / 'examples/native_backend/model.json').read_text())
    for streamed in (False, True):
        if streamed:
            (tmp_path / 'base.f32').write_bytes(struct.pack('<4f', 1, 2, 3, 4))
            model['parameters'][0] = dict(name='base', shape=[2,2], kind='file', path='base.f32', offset=0)
        path = tmp_path / 'model.json'
        path.write_text(json.dumps(model))
        for device in devices:
            with NativeSession(path, budget_bytes=4096, executable=BINARY, device=device) as gpu, \
                 NativeSession(path, budget_bytes=4096, executable=BINARY, device='cpu') as cpu:
                before = gpu([0.1, 0.2])
                for values in ([1., 2., 3., 4.], [-1., -2., -3., -4.]):
                    writes = [dict(name='delta', values=values)]
                    gpu.write_parameters(writes)
                    cpu.write_parameters(writes)
                    assert gpu([0.1,0.2]) == pytest.approx(cpu([0.1,0.2]), abs=1e-5)
                assert gpu([0.1,0.2]) != pytest.approx(before)
