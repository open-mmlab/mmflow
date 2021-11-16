# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.ops import build_operators


def test_warp():
    warp_cfg = dict(
        type='Warp',
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
        use_mask=True)
    warp_layer = build_operators(warp_cfg)

    feat = torch.ones(1, 1, 5, 5)
    # test zero flow, and no warp
    flow = torch.zeros(size=(1, 2, 5, 5))
    warp_feat = warp_layer(feat, flow)

    assert torch.all(feat == warp_feat)
