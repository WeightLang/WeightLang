import pytest
import torch

from weightlang import WgtError, compile_file, load, model_config, package_model, parse_file
from weightlang.bundle import install_bundle

CODE = '''wgt 0.3
model configurable
config WIDTH: int default=4 min=2 max=8
config DEPTH: int default=2 min=1 max=4
config BIAS: bool default=true
config SCALE: float default=0.5 min=0.1 max=1
require WIDTH % 2 == 0
input x: f32[WIDTH]
output y: f32[WIDTH]
block cell():
  layer projection: linear(in=WIDTH,out=WIDTH,bias=BIAS)
  layer activation: scale(factor=SCALE)
layer body: repeat(DEPTH,cell())
init seed=19
'''


def test_typed_configuration_load_compile_and_package_are_fixed(tmp_path):
    path = tmp_path/'model.wgt'
    path.write_text(CODE)
    cfg = {'WIDTH':6,'DEPTH':3,'BIAS':False,'SCALE':0.75}
    assert model_config(path)['values']['WIDTH'] == 4
    module = load(path,config=cfg)
    x = torch.randn(2,6)
    expected = module(x)
    assert not (tmp_path/'weights').exists()
    built = compile_file(path,tmp_path/'weights',config=cfg,no_train=True)
    torch.testing.assert_close(built.module(x),expected)
    fixed = parse_file(built.path/'source.wgt')
    assert fixed.input.shape == (6,)
    assert built.manifest['configuration']['values'] == cfg
    archive = tmp_path/'configured.wgtpkg'
    package_model(path,archive,config=cfg)
    installed = tmp_path/'installed'
    install_bundle(archive,installed)
    # Mutating the original configuration cannot change the frozen package.
    path.write_text(CODE.replace('default=4','default=8'))
    restored = load(installed)
    torch.testing.assert_close(restored(x),expected)
    with pytest.raises(WgtError,match='fixed'):
        load(installed,config=cfg)


@pytest.mark.parametrize('cfg',[{'WIDTH':3},{'WIDTH':10},{'WIDTH':True},{'SCALE':float('nan')},{'BIAS':1},{'MISSPELLED':2}])
def test_invalid_configuration_fails_before_build(tmp_path,cfg):
    path = tmp_path/'model.wgt'
    path.write_text(CODE)
    with pytest.raises(WgtError):
        load(path,config=cfg)
