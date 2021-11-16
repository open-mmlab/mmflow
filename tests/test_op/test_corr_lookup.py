# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.models.decoders.raft_decoder import CorrelationPyramid
from mmflow.ops.builder import build_operators
from mmflow.ops.corr_lookup import bilinear_sample, coords_grid


def test_coords_grid():
    W = 10
    H = 5
    xx = torch.arange(0, W)
    yy = torch.arange(0, H)

    grid = coords_grid(2, xx, yy)

    assert grid.shape == torch.Size((2, 2, H, W))
    for i in range(H):
        for j in range(W):

            assert torch.all(grid[0, :, i, j] == torch.Tensor((j, i)))


def test_bilinear_sample():
    W = 10
    H = 5
    xx = torch.arange(0, W)
    yy = torch.arange(0, H)

    grid = coords_grid(2, xx, yy)

    map = torch.randn(2, 1, H, W)

    out_map = bilinear_sample(map, grid, mode='bilinear', align_corners=True)

    assert torch.allclose(map, out_map, atol=1e-5)


def test_corr_lookup():
    corr_pyramid_layer = CorrelationPyramid(num_levels=4)

    H = 8
    W = 16
    feat1 = torch.randn(1, 1, H, W)
    feat2 = torch.randn(1, 1, H, W)

    corr_pyramid = corr_pyramid_layer(feat1, feat2)

    corr_lookup_cfg = dict(
        type='CorrLookup',
        radius=4,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    corr_lookup_op = build_operators(corr_lookup_cfg)

    corr_lpt = corr_lookup_op(corr_pyramid, torch.randn(1, 2, H, W))
    assert corr_lpt.shape == torch.Size((1, 81 * 4, H, W))
