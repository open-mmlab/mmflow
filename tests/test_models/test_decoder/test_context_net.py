# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.decoders.context_net import ContextNet


def test_context_net():

    # test invalid inchannels type
    with pytest.raises(AssertionError):
        ContextNet(in_channels='invalid')

    feat_channels = (128, 128, 128, 96, 64, 32)
    context_net = ContextNet(
        in_channels=512,
        out_channels=2,
        feat_channels=feat_channels,
        dilations=(1, 2, 4, 8, 16, 1),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None)

    # test layer out_channels
    for i, feat_in in enumerate(feat_channels):
        assert context_net.layers[i].conv.out_channels == feat_in
    assert context_net.layers[-1].out_channels == 2

    # test predicted flow shape
    in_feat = torch.randn(1, 512, 100, 100)
    out_flow = context_net(in_feat)
    assert out_flow.shape == torch.Size((1, 2, 100, 100))
