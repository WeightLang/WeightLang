import pytest
import torch
from torch.nn import functional as F
from weightlang import parse, WgtError
from test_primitives import variants, source


@pytest.mark.parametrize('op,shape,expected',[
 ('resize2d(height=5,width=7)', '2,5,7', lambda x:F.interpolate(x,size=(5,7),mode='nearest')),
 ('resize2d(height=3,width=2,mode=1)', '2,3,2', lambda x:F.interpolate(x,size=(3,2),mode='bilinear',align_corners=False)),
 ('resize2d(height=5,width=7,mode=1,align_corners=1)', '2,5,7', lambda x:F.interpolate(x,size=(5,7),mode='bilinear',align_corners=True)),
 ('pad2d(left=1,right=2,top=1,bottom=0)', '2,5,8', lambda x:F.pad(x,(1,2,1,0))),
 ('pad2d(left=1,right=2,top=1,bottom=0,mode=1)', '2,5,8', lambda x:F.pad(x,(1,2,1,0),mode='reflect')),
 ('pad2d(left=1,right=2,top=1,bottom=0,mode=2)', '2,5,8', lambda x:F.pad(x,(1,2,1,0),mode='replicate')),
 ('avgpool2d(kernel=3,stride=2,padding=1,count_include_pad=0)', '2,2,3', lambda x:F.avg_pool2d(x,3,2,1,count_include_pad=False)),
])
def test_spatial_reference_gradient_and_roundtrip(tmp_path,op,shape,expected):
    code=source([('x','f32','2,4,5')],shape,f'layer result: {op}')
    for m in variants(tmp_path,code):
        x=torch.randn(3,2,5,4).transpose(-1,-2).requires_grad_()
        actual=m(x)['y'];reference=expected(x)
        torch.testing.assert_close(actual,reference)
        ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        gb=torch.autograd.grad(reference.square().sum(),x)[0]
        torch.testing.assert_close(ga,gb)
        torch.testing.assert_close(m(x[0])['y'],reference[0])


@pytest.mark.parametrize('op',[
 'resize2d(height=0,width=4)', 'resize2d(height=4,width=4,mode=2)',
 'resize2d(height=4,width=4,align_corners=1)',
 'pad2d(left=-1,right=0,top=0,bottom=0)',
 'pad2d(left=5,right=0,top=0,bottom=0,mode=1)',
 'avgpool2d(kernel=3,stride=0)', 'avgpool2d(kernel=3,stride=1,padding=2)',
])
def test_invalid_spatial_contracts(op):
    with pytest.raises(WgtError):parse(source([('x','f32','2,4,5')],'2,4,5',f'layer result: {op}'))
