# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.models import build_components
from mmflow.models.decoders.raft_decoder import CorrelationPyramid
from mmflow.models.utils.corr_lookup import bilinear_sample, coords_grid
from mmflow.models.utils.correlation1d import Correlation1D


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
    corr_lookup_op = build_components(corr_lookup_cfg)

    corr_lpt = corr_lookup_op(corr_pyramid, torch.randn(1, 2, H, W))
    assert corr_lpt.shape == torch.Size((1, 81 * 4, H, W))


def test_corr_lookup_flow1d():
    corr_block = Correlation1D()
    feat1 = torch.arange(0, 24)
    feat1 = feat1.view(1, 2, 3, 4)
    feat2 = feat1 + 1
    flow = torch.ones_like(feat1)
    b, _, h, w = feat1.size()
    radius = 32

    correlation_x = corr_block(feat1, feat2, True)
    correlation_y = corr_block(feat1, feat2, False)
    correlation = [correlation_x, correlation_y]
    corr_lookup_cfg = dict(
        type='CorrLookupFlow1D',
        radius=radius,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    corr_lookup_op = build_components(corr_lookup_cfg)

    corr_xy = corr_lookup_op(correlation, flow)
    assert corr_xy.size() == (b, 2 * (2 * radius + 1), h, w)
