# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import Tensor

from mmflow.models.decoders.raft_decoder import CorrelationPyramid
from mmflow.models.utils.correlation1d import Correlation1D
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


@pytest.mark.parametrize('mode', ['bilinear', 'nearest', 'bicubic'])
@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
@pytest.mark.parametrize('align_corners', [True, False])
def test_corr_lookup_flow1d(mode, padding_mode, align_corners):
    corr_block = Correlation1D()
    feat1 = torch.arange(0, 24)
    feat1 = feat1.view(1, 2, 3, 4)
    feat2 = feat1 + 1
    flow = torch.ones_like(feat1)
    b, c, h, w = feat1.size()
    radius = 1

    # gronud truth
    gt_corr_x = Tensor([[[[110.3087, 120.2082, 130.1077, 140.0071],
                          [206.4752, 222.0315, 237.5879, 253.1442],
                          [347.8965, 369.1097, 390.3229, 411.5362]],
                         [[118.7939, 130.1077, 141.4214, 152.7351],
                          [220.6173, 237.5879, 254.5584, 271.5290],
                          [367.6955, 390.3229, 412.9504, 435.5778]],
                         [[127.2792, 140.0071, 152.7351, 165.4630],
                          [234.7595, 253.1442, 271.5290, 289.9138],
                          [387.4945, 411.5362, 435.5778, 459.6194]]]])
    gt_corr_y = Tensor([[[[110.3087, 130.1077, 152.7351, 178.1909],
                          [149.9066, 175.3625, 203.6468, 234.7595],
                          [189.5046, 220.6173, 254.5584, 291.3280]],
                         [[144.2498, 169.7056, 197.9899, 229.1026],
                          [206.4752, 237.5879, 271.5290, 308.2986],
                          [268.7006, 305.4701, 345.0681, 387.4945]],
                         [[178.1909, 209.3036, 243.2447, 280.0143],
                          [263.0437, 299.8133, 339.4113, 381.8377],
                          [347.8965, 390.3229, 435.5778, 483.6610]]]])
    gt_corr = torch.cat((gt_corr_x, gt_corr_y), dim=1)
    correlation_x = corr_block(feat1, feat2, False)
    correlation_y = corr_block(feat1, feat2, True)
    correlation = [correlation_x, correlation_y]
    corr_lookup_cfg = dict(
        type='CorrLookupFlow1D',
        radius=radius,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True)
    corr_lookup_op = build_operators(corr_lookup_cfg)

    corr_xy = corr_lookup_op(correlation, flow)
    assert corr_xy.size() == (b, 2 * (2 * radius + 1), h, w)
    assert torch.allclose(gt_corr, corr_xy, atol=1e-4)
