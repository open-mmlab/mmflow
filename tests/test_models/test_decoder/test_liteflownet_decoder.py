# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmflow.models.decoders.liteflownet_decoder import (BasicBlock,
                                                        MatchingBlock, NetE,
                                                        RegularizationBlock,
                                                        SubpixelBlock,
                                                        Upsample)


def test_uspample():
    upsample_layer = Upsample(scale_factor=2, channels=1)
    assert upsample_layer.kernel_size == 4
    assert upsample_layer.stride == 2
    assert upsample_layer.pad == 1
    assert upsample_layer.weight.shape == torch.Size((1, 1, 4, 4))
    assert not hasattr(upsample_layer, 'bias')


def test_basic_block():
    feat_channels = (10, 20, 20)
    layers = BasicBlock(
        3,
        feat_channels=feat_channels,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None)
    in_feat = torch.randn(1, 3, 10, 10)

    # basic block can't be called
    with pytest.raises(NotImplementedError):
        layers(in_feat)

    # test out channels of each layer
    for i, feat_ch in enumerate(feat_channels):
        assert layers.layers[i].conv.out_channels == feat_ch
        assert isinstance(layers.layers[i].activate, nn.LeakyReLU)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('scaled_corr', (True, False))
def test_matching_block(scaled_corr):
    in_channels = (2 * 1 + 1)**2  # corr feat channels
    feat_channels = (10, 20, 20)
    corr_cfg = dict(type='Correlation', max_displacement=1)
    warp_cfg = dict(type='Warp', align_corners=False, use_mask=False)
    last_kernel_size = 3

    mnet_module = MatchingBlock(
        in_channels,
        feat_channels,
        None,
        None,
        dict(type='LeakyReLU', negative_slope=0.1),
        None,
        corr_cfg=corr_cfg,
        warp_cfg=warp_cfg,
        last_kernel_size=last_kernel_size,
        scaled_corr=scaled_corr).cuda()

    # test pred flow layer
    assert mnet_module.pred_flow.out_channels == 2
    assert mnet_module.pred_flow.in_channels == 20

    in_feat1 = torch.randn(1, 3, 10, 10).cuda()
    in_feat2 = torch.randn(1, 3, 10, 10).cuda()
    up_flow = torch.randn(1, 2, 10, 10).cuda()

    out_flow = mnet_module(in_feat1, in_feat2, up_flow)

    # test predicted flow shape
    assert out_flow.shape == torch.Size((1, 2, 10, 10))


def test_subpixel_block():
    in_channels = 3 * 2 + 2  # input feat channels * 2 + flow channels
    feat_channels = (10, 20, 20)

    warp_cfg = dict(type='Warp', align_corners=False, use_mask=False)
    last_kernel_size = 3

    snet_module = SubpixelBlock(
        in_channels,
        feat_channels,
        None,
        None,
        dict(type='LeakyReLU', negative_slope=0.1),
        None,
        warp_cfg=warp_cfg,
        last_kernel_size=last_kernel_size)

    # test pred flow layer
    assert snet_module.pred_flow.out_channels == 2
    assert snet_module.pred_flow.in_channels == 20

    in_feat1 = torch.randn(1, 3, 10, 10)
    in_feat2 = torch.randn(1, 3, 10, 10)
    flow = torch.randn(1, 2, 10, 10)

    out_flow = snet_module(in_feat1, in_feat2, flow, 1.)

    # test predicted flow shape
    assert out_flow.shape == torch.Size((1, 2, 10, 10))


def test_regularization_block():

    # diff img channels + nomean flow channels + feat channels
    in_channels = 1 + 2 + 5
    feat_channels = (10, 20, 20)
    warp_cfg = dict(type='Warp', align_corners=False, use_mask=False)
    last_kernel_size = 3
    out_channels = 3**2

    rnet_module = RegularizationBlock(
        in_channels,
        feat_channels,
        None,
        None,
        dict(type='LeakyReLU', negative_slope=0.1),
        None,
        last_kernel_size=last_kernel_size,
        out_channels=out_channels,
        warp_cfg=warp_cfg,
    )

    # test dist feat layer
    assert rnet_module.dist_layer.in_channels == 20
    assert rnet_module.dist_layer.out_channels == 9

    img1 = torch.randn(1, 3, 10, 10)
    img2 = torch.randn(1, 3, 10, 10)
    feat = torch.randn(1, 5, 10, 10)
    flow = torch.randn(1, 2, 10, 10)

    pred_flow = rnet_module(img1, img2, feat, flow, 1.)

    # test predicted flow shape
    assert pred_flow.shape == torch.Size((1, 2, 10, 10))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize(('extra_training_loss', 'regularized_flow'),
                         [(True, True), (False, False)])
def test_nete(extra_training_loss, regularized_flow):
    if extra_training_loss:
        flow_loss = dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights=dict(level6=0.32, level5=0.08, level0=0.0003125))
    else:
        flow_loss = dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights=dict(level6=0.32, level5=0.08))
    nete = NetE(
        in_channels=dict(level5=128, level6=192),
        corr_channels=dict(level5=49, level6=49),
        sin_channels=dict(level5=258, level6=386),
        rin_channels=dict(level5=131, level6=195),
        feat_channels=64,
        mfeat_channels=(128, 64, 32),
        sfeat_channels=(128, 64, 32),
        rfeat_channels=(128, 128, 64, 64, 32, 32),
        patch_size=dict(level5=3, level6=3),
        corr_cfg=dict(
            level5=dict(type='Correlation', max_displacement=3),
            level6=dict(type='Correlation', max_displacement=3)),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        flow_div=20.,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled_corr=True,
        regularized_flow=regularized_flow,
        extra_training_loss=extra_training_loss,
        flow_loss=flow_loss,
        init_cfg=None).cuda()

    feat1 = dict(
        level6=torch.randn(1, 192, 4, 4).cuda(),
        level5=torch.randn(1, 128, 8, 8).cuda())
    feat2 = dict(
        level6=torch.randn(1, 192, 4, 4).cuda(),
        level5=torch.randn(1, 128, 8, 8).cuda())
    img1 = torch.randn(1, 3, 16, 16).cuda()
    img2 = torch.randn(1, 3, 16, 16).cuda()

    flow_gt = torch.randn(1, 2, 16, 16).cuda()

    loss = nete.forward_train(img1, img2, feat1, feat2, flow_gt)
    assert float(loss['loss_flow']) > 0

    out = nete.forward_test(img1, img2, feat1, feat2)
    assert isinstance(out, list)
    assert out[0]['flow'].shape == (16, 16, 2)

    # test forward
    flow_pred = nete.forward(img1, img2, feat1, feat2)
    assert flow_pred['level5'].shape == torch.Size((1, 2, 8, 8))
    assert flow_pred['level6'].shape == torch.Size((1, 2, 4, 4))
