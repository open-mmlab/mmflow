# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.decoders.maskflownet_decoder import (BasicDeformWarpBlock,
                                                        DeformWarpBlock,
                                                        MaskFlowNetSDecoder,
                                                        MaskModule)


def _get_test_data():

    input_feat1 = dict(
        level6=torch.randn(1, 196, 4, 4).cuda(),
        level5=torch.randn(1, 128, 8, 8).cuda())

    input_feat2 = dict(
        level6=torch.randn(1, 196, 4, 4).cuda(),
        level5=torch.randn(1, 128, 8, 8).cuda())
    return input_feat1, input_feat2


@pytest.mark.parametrize(('up_flow', 'with_mask'),
                         [(True, True), (True, False), (False, True),
                          (False, False)])
def test_mask_module(up_flow, with_mask):

    in_channels = 10
    densefeat_channels = (20, 30, 40)
    input = torch.randn(1, 10, 56, 56)
    input_upflow = torch.randn(1, 2, 56, 56)

    submodule = MaskModule(
        up_channels=10,
        with_mask=with_mask,
        up_flow=up_flow,
        in_channels=in_channels,
        densefeat_channels=densefeat_channels)

    outputs = submodule(input, input_upflow)

    if up_flow:
        assert outputs[3].shape == torch.Size((1, 2, 56 * 2, 56 * 2))
        assert outputs[-1].shape == torch.Size((1, 10, 56 * 2, 56 * 2))
        if with_mask:
            assert outputs[-2].shape == torch.Size((1, 1, 56 * 2, 56 * 2))
    else:
        assert outputs[3] is None and outputs[-1] is None and outputs[
            -2] is None

    if with_mask:
        # test predict mask
        assert outputs[1].shape == torch.Size((1, 1, 56, 56))
    else:
        # test only predict flow without mask
        assert outputs[1] is None

    assert outputs[0].shape == torch.Size((1, 2, 56, 56))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('with_deform_bias', [True, False])
def test_warp_block(with_deform_bias):

    in_channels = 10
    up_channels = 20
    feat2 = torch.randn(1, in_channels, 56, 56).cuda()
    flow = torch.randn(1, 2, 56, 56).cuda()
    # test BasicDeformWarmBlock
    warp_block = BasicDeformWarpBlock(
        channels=in_channels,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        with_deform_bias=with_deform_bias).cuda()

    assert with_deform_bias == hasattr(warp_block, 'deconv_bias')

    assert warp_block(feat2, flow).shape == feat2.shape

    # test DeformWarpBlock
    warp_block = DeformWarpBlock(
        channels=in_channels,
        up_channels=up_channels,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        with_deform_bias=with_deform_bias).cuda()

    assert with_deform_bias == hasattr(warp_block, 'deconv_bias')
    upfeat = torch.randn((1, 20, 56, 56)).cuda()
    mask = torch.randn((1, 1, 56, 56)).cuda()
    assert warp_block(feat2, flow, mask, upfeat).shape == feat2.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_maskflownets_decoder():

    in_channels = dict(level6=81, level5=227)
    warp_in_channels = dict(level6=196, level5=128)
    up_channels = dict(level6=16, level5=16)
    warp_type = 'AsymOFMM'
    densefeat_channels = (128, 128)

    # test invalid warp_in_channels
    with pytest.raises(AssertionError):
        MaskFlowNetSDecoder(
            warp_in_channels='invalid type',
            up_channels=up_channels,
            in_channels=in_channels)

    # test invalid up_channels
    with pytest.raises(AssertionError):
        MaskFlowNetSDecoder(
            warp_in_channels=warp_in_channels,
            up_channels='invalid type',
            in_channels=in_channels)

    # test invalid in_channels
    with pytest.raises(AssertionError):
        MaskFlowNetSDecoder(
            warp_in_channels=warp_in_channels,
            up_channels=up_channels,
            in_channels='invalid type')

    model = MaskFlowNetSDecoder(
        warp_in_channels=warp_in_channels,
        warp_type=warp_type,
        densefeat_channels=densefeat_channels,
        up_channels=up_channels,
        in_channels=in_channels,
        corr_cfg=dict(type='Correlation', max_displacement=4),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights=dict(level6=0.32, level5=0.08)),
        scaled=False,
    ).cuda()

    feat1, feat2 = _get_test_data()
    flow_gt = torch.randn(1, 2, 16, 16).cuda()

    # test forward train
    loss = model.forward_train(feat1, feat2, flow_gt=flow_gt)
    assert float(loss['loss_flow']) > 0

    # test forward test
    out = model.forward_test(feat1, feat2, H=16, W=16)
    assert isinstance(out, list)
    assert out[0]['flow'].shape == (16, 16, 2)

    # test forward function
    out = model(feat1, feat2)
    assert out['level6'].shape == torch.Size((1, 2, 4, 4))
    assert out['level5'].shape == torch.Size((1, 2, 8, 8))
