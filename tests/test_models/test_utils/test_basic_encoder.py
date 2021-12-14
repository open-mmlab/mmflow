# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.utils import BasicEncoder


def test_basic_encoder():
    in_channels = 3
    pyramid_levels = ('level1', 'level2', 'level3')

    # test the numbers of out_channels, num_convs, stride, dilation are
    # not equal
    with pytest.raises(AssertionError):
        BasicEncoder(
            in_channels=in_channels,
            pyramid_levels=pyramid_levels,
            out_channels=(16, 32),
            strides=(2, 2, 2, 2),
            dilations=(1, 1, 1))

    # test output shape
    strides = (1, 2, 2)
    out_channels = (16, 32, 64)
    num_convs = (1, 1, 1)
    model = BasicEncoder(
        in_channels=in_channels,
        pyramid_levels=pyramid_levels,
        out_channels=out_channels,
        num_convs=num_convs,
        strides=strides,
        dilations=(1, 1, 1))

    model.train()
    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert isinstance(feat, dict)
    feat_keys = list(feat.keys())
    feat_keys.sort()
    assert tuple(feat_keys) == tuple(pyramid_levels)

    H = 256
    for s, out_ch, level in zip(strides, out_channels, pyramid_levels):
        H = int(H / s)
        assert feat[level].shape == torch.Size([1, out_ch, H, H])
