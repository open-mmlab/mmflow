# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.encoders import NetC


def test_netc():
    pyramid_levels = [
        'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
    ]
    out_channels = (32, 32, 64, 96, 128, 192)
    strides = (1, 2, 2, 2, 2, 2)

    # test the number of out_channels, num_convs, stride, dilation is not equal
    with pytest.raises(AssertionError):

        NetC(
            in_channels=3,
            pyramid_levels=['level1', 'level2'],
            out_channels=out_channels,
            strides=(2, 2),
            num_convs=(1, 3, 2, 2),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))

    # test NetC
    model = NetC(
        in_channels=3,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(32, 32, 64, 96, 128, 192),
        strides=strides,
        num_convs=(1, 3, 2, 2, 1, 1),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None)
    model.train()

    H = 256
    imgs = torch.randn(1, 3, H, H)

    feat = model(imgs)
    assert isinstance(feat, dict)
    feat_keys = list(feat.keys())
    feat_keys.sort()
    assert feat_keys == pyramid_levels

    for i, k in enumerate(pyramid_levels):
        stride = strides[i]
        H = int(H / stride)
        assert feat[k].shape == torch.Size([1, out_channels[i], H, H])
