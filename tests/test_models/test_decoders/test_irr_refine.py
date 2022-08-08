# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.models.decoders.irr_refine import (FlowRefine, OccRefine,
                                               OccShuffleUpsample)


def test_flow_refine():

    feat_channels = (128, 128, 96, 64, 64, 32, 32)
    patch_size = 3

    flow_refine = FlowRefine(
        in_channels=35,
        feat_channels=feat_channels,
        patch_size=patch_size,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        warp_cfg=dict(type='Warp', align_corners=True),
    )

    # test layer out_channels
    for i, feat_in in enumerate(feat_channels):
        assert flow_refine.layers[i].conv.out_channels == feat_in
    assert flow_refine.layers[-1].out_channels == patch_size**2

    img1 = torch.randn(1, 3, 100, 100)
    img2 = torch.randn(1, 3, 100, 100)
    feat = torch.randn(1, 32, 100, 100)
    flow = torch.randn(1, 2, 100, 100)

    out_flow = flow_refine(img1, img2, feat, flow)

    assert out_flow.shape == torch.Size((1, 2, 100, 100))


def test_occ_refine():
    feat_channels = (128, 128, 96, 64, 64, 32, 32)
    patch_size = 3

    occ_refine = OccRefine(
        in_channels=65,
        feat_channels=feat_channels,
        patch_size=patch_size,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        warp_cfg=dict(type='Warp', align_corners=True),
    )

    # test layer out_channels
    for i, feat_in in enumerate(feat_channels):
        assert occ_refine.layers[i].conv.out_channels == feat_in
    assert occ_refine.layers[-1].out_channels == patch_size**2

    feat1 = torch.randn(1, 32, 100, 100)
    feat2 = torch.randn(1, 32, 100, 100)
    occ = torch.randn(1, 1, 100, 100)
    flow = torch.randn(1, 2, 100, 100)

    out_occ = occ_refine(feat1, feat2, occ, flow)

    assert out_occ.shape == torch.Size((1, 1, 100, 100))


def test_occ_Shuffle_upsample():

    occ_Shuffle = OccShuffleUpsample(
        in_channels=11,
        feat_channels=32,
        infeat_channels=16,
        out_channels=1,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        warp_cfg=dict(type='Warp', align_corners=True))

    occ = torch.randn(1, 1, 100, 100)
    feat1 = torch.randn(1, 16, 100, 100)
    feat2 = torch.randn(1, 16, 100, 100)
    flow_f = torch.randn(1, 2, 100, 100)
    flow_b = torch.randn(1, 2, 100, 100)
    flow_div = 20
    H_img = 200
    W_img = 200

    out_pred_flow = occ_Shuffle(occ, feat1, feat2, flow_f, flow_b, flow_div,
                                H_img, W_img)
    assert out_pred_flow.shape == torch.Size((1, 1, 100, 100))
