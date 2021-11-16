# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.decoders.pwcnet_decoder import PWCModule, PWCNetDecoder
from mmflow.models.decoders.utils import CorrBlock


def _get_test_data():

    input_feat1 = dict(
        level6=torch.randn(1, 196, 8, 8).cuda(),
        level5=torch.randn(1, 128, 16, 16).cuda())

    input_feat2 = dict(
        level6=torch.randn(1, 196, 8, 8).cuda(),
        level5=torch.randn(1, 128, 16, 16).cuda())
    return input_feat1, input_feat2


def _get_corr_block_cfg():
    return dict(
        corr_cfg=dict(type='Correlation', max_displacement=1, padding=0))


@pytest.mark.parametrize('scaled', [True, False])
def test_corr_block(scaled):
    feat1 = torch.randn(1, 10, 10, 10).cuda()
    feat2 = torch.randn(1, 10, 10, 10).cuda()
    corr_block_cfg = _get_corr_block_cfg()
    out = CorrBlock(**corr_block_cfg, scaled=scaled)(feat1, feat2)

    assert out.shape == torch.Size((1, 9, 10, 10))


@pytest.mark.parametrize('up_flow', [False, True])
def test_pwcmodule(up_flow):
    submodule = PWCModule(
        in_channels=10, densefeat_channels=(20, 30, 40), up_flow=up_flow)

    input = torch.randn(1, 10, 56, 56)
    # test head can upsample flow next level
    if up_flow:
        assert hasattr(submodule, 'upflow_layer') and hasattr(
            submodule, 'upfeat_layer')
        assert submodule.upflow_layer.out_channels == 2
        assert submodule.upfeat_layer.out_channels == 2
        flow, feat, up_flow, up_feat = submodule(input)
        assert up_flow.shape == torch.Size((1, 2, 112, 112))
        assert up_feat.shape == torch.Size((1, 2, 112, 112))

    else:
        assert not (hasattr(submodule, 'upflow_layer')
                    or hasattr(submodule, 'upfeat_layer'))
        flow, feat, up_flow, up_feat = submodule(input)
        assert up_flow is None
        assert up_feat is None

    assert flow.shape == torch.Size((1, 2, 56, 56))
    assert feat.shape == torch.Size((1, 100, 56, 56))


def test_pwcnet_decoder():

    # test invalid in_channels
    with pytest.raises(AssertionError):
        PWCNetDecoder(in_channels='invalid type')

    # test output of pwcnet decoder
    in_channels = dict(level6=81, level5=213)
    densefeat_channels = (128, 128)
    model = PWCNetDecoder(
        in_channels=in_channels,
        densefeat_channels=densefeat_channels,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp'),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={
                'level5': 0.08,
                'level6': 0.32
            })).cuda()

    feat1, feat2 = _get_test_data()

    flow_gt = torch.randn(1, 2, 32, 32).cuda()

    loss = model.forward_train(feat1, feat2, flow_gt)
    assert float(loss['loss_flow']) > 0

    out = model.forward_test(feat1, feat2, H=32, W=32)
    assert isinstance(out, list)
    assert out[0]['flow'].shape == (32, 32, 2)
