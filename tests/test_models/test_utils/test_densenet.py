# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.models.utils import BasicDenseBlock


def test_denseblock():
    in_channels = 2
    feat_channels = [4, 8]
    x = torch.randn(1, 2, 5, 5)

    dense_block = BasicDenseBlock(
        in_channels=in_channels, feat_channels=feat_channels)

    assert dense_block(x).shape == torch.Size((1, 2 + 4 + 8, 5, 5))
