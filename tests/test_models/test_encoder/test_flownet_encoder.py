# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.encoders import (CorrEncoder, FlowNetEncoder,
                                    FlowNetSDEncoder)


def test_flownet_encoder():
    # test FlowNetS encoder
    pyramid_levels = [
        'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
    ]
    strides = (2, 2, 2, 2, 2, 2)
    out_channels = (64, 128, 256, 512, 512, 1024)
    model = FlowNetEncoder(
        in_channels=6,
        pyramid_levels=pyramid_levels,
        num_convs=(1, 1, 2, 2, 2, 2),
        out_channels=out_channels,
        kernel_size=(7, 5, (5, 3), 3, 3, 3),
        strides=strides,
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1))

    model.train()

    H = 256
    imgs = torch.randn(1, 6, H, H)
    feat = model(imgs)
    assert isinstance(feat, dict)
    feat_keys = list(feat.keys())
    feat_keys.sort()
    assert feat_keys == pyramid_levels

    for i, k in enumerate(pyramid_levels):
        stride = strides[i]
        H = int(H / stride)
        assert feat[k].shape == torch.Size([1, out_channels[i], H, H])

    # test FlowNetC encoder
    pyramid_levels = ['level1', 'level2', 'level3']
    strides = (2, 2, 2)
    model = FlowNetEncoder(
        in_channels=3,
        pyramid_levels=pyramid_levels,
        out_channels=(64, 128, 256),
        kernel_size=(7, 5, 5),
        strides=strides,
        num_convs=(1, 1, 1),
        dilations=(1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    )

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


def test_flownet_dc_encoder():
    pyramid_levels = [
        'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
    ]
    strides = (2, 2, 2, 2, 2, 2)
    out_channels = ((64, 128), 128, 256, 512, 512, 1024)
    model = FlowNetSDEncoder(
        in_channels=6,
        plugin_channels=64,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        num_convs=(2, 2, 2, 2, 2, 2),
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    )
    model.train()

    assert model.plugin_layer.conv.in_channels == 6
    assert model.plugin_layer.conv.out_channels == 64

    assert model.layers[0].layers[0].conv.in_channels == 64
    assert model.layers[0].layers[0].conv.out_channels == 64

    assert model.layers[0].layers[1].conv.in_channels == 64
    assert model.layers[0].layers[1].conv.out_channels == 128

    H = 256
    imgs = torch.randn(1, 6, H, H)
    feat = model(imgs)
    assert isinstance(feat, dict)
    feat_keys = list(feat.keys())
    feat_keys.sort()
    assert feat_keys == pyramid_levels

    for i, k in enumerate(pyramid_levels):
        stride = strides[i]
        H = int(H / stride)
        out_channel = out_channels[i][-1] if isinstance(
            out_channels[i], (tuple, list)) else out_channels[i]
        assert feat[k].shape == torch.Size([1, out_channel, H, H])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_corr_encoder():
    pyramid_levels = ['level3', 'level4', 'level5', 'level6']
    strides = (1, 2, 2, 2)
    out_channels = (256, 512, 512, 1024)
    model = CorrEncoder(
        in_channels=473,
        pyramid_levels=pyramid_levels,
        kernel_size=(3, 3, 3, 3),
        num_convs=(1, 2, 2, 2),
        out_channels=out_channels,
        redir_in_channels=256,
        redir_channels=32,
        strides=strides,
        dilations=(1, 1, 1, 1),
        conv_cfg=None,
        corr_cfg=dict(
            type='Correlation',
            kernel_size=1,
            max_displacement=10,
            stride=1,
            padding=0,
            dilation_patch=2),
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False)

    model.cuda()

    feat1 = torch.randn(1, 256, 32, 32).cuda()
    feat2 = torch.randn(1, 256, 32, 32).cuda()
    out = model(feat1, feat2)
    s = 1
    for level in pyramid_levels:
        i = int(level[-1]) - 3
        s *= 2**(strides[i] - 1)
        assert out[level].shape == torch.Size(
            (1, out_channels[i], int(32 / s), int(32 / s)))
