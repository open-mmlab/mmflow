# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.decoders.raft_decoder import (ConvGRU, CorrelationPyramid,
                                                 MotionEncoder, RAFTDecoder,
                                                 XHead)


def test_correlation_pyramid():
    corr_pyramid_layer = CorrelationPyramid(num_levels=4)

    H = 64
    W = 64
    feat1 = torch.randn(1, 1, H, W)
    feat2 = torch.randn(1, 1, H, W)

    corr_pyramid = corr_pyramid_layer(feat1, feat2)

    assert len(corr_pyramid) == 4
    H_ = 64
    W_ = 64
    for i in range(4):
        assert corr_pyramid[i].shape == torch.Size((H * W, 1, H_, W_))
        H_ = H_ // 2
        W_ = W_ // 2


@pytest.mark.parametrize('net_type', ['Basic', 'Small'])
def test_motion_encoder(net_type):

    # test invalid net_type
    with pytest.raises(AssertionError):
        MotionEncoder(net_type='invalid value')

    module = MotionEncoder(
        net_type=net_type, conv_cfg=None, norm_cfg=None, act_cfg=None)
    num_levels = 4
    radius = 4

    input_corr = torch.randn((1, num_levels * (2 * radius + 1)**2, 56, 56))
    input_flow = torch.randn((1, 2, 56, 56))

    corr_feat = module.corr_net(input_corr)
    flow_feat = module.flow_net(input_flow)
    our_feat = module.out_net(torch.cat([corr_feat, flow_feat], dim=1))

    if net_type == 'Basic':
        assert corr_feat.shape == torch.Size((1, 192, 56, 56))
        assert flow_feat.shape == torch.Size((1, 64, 56, 56))
        assert our_feat.shape == torch.Size((1, 126, 56, 56))
    elif net_type == 'Small':
        assert corr_feat.shape == torch.Size((1, 96, 56, 56))
        assert flow_feat.shape == torch.Size((1, 32, 56, 56))
        assert our_feat.shape == torch.Size((1, 80, 56, 56))


@pytest.mark.parametrize('net_type', ('Conv', 'SeqConv'))
def test_convgru(net_type):
    h_ch = 10
    x_ch = 10
    GRUmodule = ConvGRU(h_channels=h_ch, x_channels=x_ch, net_type=net_type)
    assert GRUmodule.conv_z[0].conv.in_channels == h_ch + x_ch
    assert GRUmodule.conv_r[0].conv.in_channels == h_ch + x_ch
    assert GRUmodule.conv_q[0].conv.in_channels == h_ch + x_ch

    if net_type == 'SeqConv':
        assert GRUmodule.conv_z[1].conv.in_channels == h_ch + x_ch
        assert GRUmodule.conv_r[1].conv.in_channels == h_ch + x_ch
        assert GRUmodule.conv_q[1].conv.in_channels == h_ch + x_ch


@pytest.mark.parametrize('x', ('flow', 'mask'))
def test_xhead(x):
    in_ch = 32
    out_ch = [32, 64]
    if x == 'flow':
        x_ch = 2
    elif x == 'mask':
        x_ch = 1

    # test invalid x input
    with pytest.raises(ValueError):
        XHead(
            in_channels=in_ch,
            feat_channels=out_ch,
            x_channels=x_ch,
            x='invalid')

    xhead_module = XHead(
        in_channels=in_ch, feat_channels=out_ch, x_channels=x_ch, x=x)
    for i, out_ch_ in enumerate(out_ch):
        assert xhead_module.layers[i].conv.out_channels == out_ch_
    assert xhead_module.predict_layer.out_channels == x_ch


def test_raft_decoder():
    model = RAFTDecoder(
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        flow_loss=dict(type='SequenceLoss'))
    mask = torch.ones((1, 64 * 9, 10, 10))
    flow = torch.randn((1, 2, 10, 10))
    assert model._upsample(flow, mask).shape == torch.Size((1, 2, 80, 80))

    feat1 = torch.randn(1, 256, 8, 8)
    feat2 = torch.randn(1, 256, 8, 8)
    h_feat = torch.randn(1, 128, 8, 8)
    cxt_feat = torch.randn(1, 128, 8, 8)
    flow = torch.zeros((1, 2, 8, 8))

    flow_gt = torch.randn(1, 2, 64, 64)
    # test forward function
    out = model(feat1, feat2, flow, h_feat, cxt_feat)
    assert isinstance(out, list)
    assert out[0].shape == torch.Size((1, 2, 64, 64))

    # test forward train
    loss = model.forward_train(
        feat1, feat2, flow, h_feat, cxt_feat, flow_gt=flow_gt)
    assert float(loss['loss_flow']) > 0.

    # test forward test
    out = model.forward_test(feat1, feat2, flow, h_feat, cxt_feat)
    assert out[0]['flow'].shape == (64, 64, 2)
