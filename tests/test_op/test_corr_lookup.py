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
    feat1 = torch.arange(0, 64)
    feat1 = feat1.view(1, 2, 4, 8)
    feat2 = feat1 + 1
    flow = torch.ones_like(feat1)
    b, c, h, w = feat1.size()
    radius = 1

    # gronud truth
    gt_corr_x = Tensor([[[[
        746.7048, 770.7464, 794.7880, 818.8297, 842.8713, 866.9129, 890.9546,
        914.9962
    ],
                          [
                              1210.5668, 1245.9221, 1281.2775, 1316.6328,
                              1351.9882, 1387.3435, 1422.6989, 1458.0542
                          ],
                          [
                              1855.4482, 1902.1173, 1948.7864, 1995.4553,
                              2042.1244, 2088.7935, 2135.4624, 2182.1316
                          ],
                          [
                              2681.3489, 2739.3318, 2797.3145, 2855.2971,
                              2913.2800, 2971.2627, 3029.2456, 3087.2283
                          ]],
                         [[
                             769.3322, 794.7880, 820.2439, 845.6997, 871.1556,
                             896.6114, 922.0673, 947.5231
                         ],
                          [
                              1244.5079, 1281.2775, 1318.0471, 1354.8167,
                              1391.5862, 1428.3557, 1465.1252, 1501.8948
                          ],
                          [
                              1900.7030, 1948.7864, 1996.8696, 2044.9529,
                              2093.0359, 2141.1194, 2189.2026, 2237.2859
                          ],
                          [
                              2737.9175, 2797.3145, 2856.7114, 2916.1084,
                              2975.5054, 3034.9023, 3094.2993, 3153.6963
                          ]],
                         [[
                             791.9596, 818.8297, 845.6997, 872.5698, 899.4398,
                             926.3099, 953.1799, 980.0500
                         ],
                          [
                              1278.4491, 1316.6328, 1354.8167, 1393.0004,
                              1431.1842, 1469.3679, 1507.5516, 1545.7355
                          ],
                          [
                              1945.9579, 1995.4553, 2044.9529, 2094.4504,
                              2143.9478, 2193.4453, 2242.9426, 2292.4402
                          ],
                          [
                              2794.4861, 2855.2971, 2916.1084, 2976.9197,
                              3037.7307, 3098.5420, 3159.3533, 3220.1643
                          ]]]])
    gt_corr_y = Tensor([[[[
        746.7048, 794.7880, 845.6997, 899.4398, 956.0084, 1015.4053, 1077.6307,
        1142.6846
    ],
                          [
                              939.0378, 998.4348, 1060.6602, 1125.7140,
                              1193.5963, 1264.3070, 1337.8461, 1414.2136
                          ],
                          [
                              1131.3708, 1202.0815, 1275.6206, 1351.9882,
                              1431.1842, 1513.2085, 1598.0614, 1685.7426
                          ],
                          [
                              1323.7039, 1405.7283, 1490.5812, 1578.2623,
                              1668.7720, 1762.1101, 1858.2766, 1957.2716
                          ]],
                         [[
                             927.7241, 987.1211, 1049.3464, 1114.4003,
                             1182.2826, 1252.9933, 1326.5323, 1402.8999
                         ],
                          [
                              1210.5668, 1281.2775, 1354.8167, 1431.1842,
                              1510.3801, 1592.4045, 1677.2573, 1764.9386
                          ],
                          [
                              1493.4095, 1575.4340, 1660.2867, 1747.9680,
                              1838.4777, 1931.8158, 2027.9823, 2126.9773
                          ],
                          [
                              1776.2523, 1869.5903, 1965.7568, 2064.7520,
                              2166.5752, 2271.2271, 2378.7073, 2489.0159
                          ]],
                         [[
                             1108.7434, 1179.4541, 1252.9933, 1329.3607,
                             1408.5568, 1490.5812, 1575.4340, 1663.1152
                         ],
                          [
                              1482.0958, 1564.1202, 1648.9730, 1736.6543,
                              1827.1639, 1920.5021, 2016.6686, 2115.6636
                          ],
                          [
                              1855.4482, 1948.7864, 2044.9529, 2143.9478,
                              2245.7712, 2350.4231, 2457.9033, 2568.2119
                          ],
                          [
                              2228.8005, 2333.4524, 2440.9326, 2551.2412,
                              2664.3784, 2780.3440, 2899.1379, 3020.7603
                          ]]]])
    gt_corr = torch.cat((gt_corr_x, gt_corr_y), dim=1)
    correlation = corr_block(feat1, feat2, feat2)

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
