# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.ops import DeformConv2dPack
from torch.nn.modules import AvgPool2d

from mmflow.models.utils import BasicBlock, Bottleneck, ResLayer


def test_basicblock():
    with pytest.raises(AssertionError):
        # Not implemented yet.
        dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
        BasicBlock(64, 64, dcn=dcn)

    with pytest.raises(AssertionError):
        # Not implemented yet.
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3')
        ]
        BasicBlock(64, 64, plugins=plugins)

    with pytest.raises(AssertionError):
        # Not implemented yet
        plugins = [
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0010',
                    kv_stride=2),
                position='after_conv2')
        ]
        BasicBlock(64, 64, plugins=plugins)

    # test BasicBlock structure and forward
    block = BasicBlock(64, 64)
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 64
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv2.in_channels == 64
    assert block.conv2.out_channels == 64
    assert block.conv2.kernel_size == (3, 3)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test BasicBlock with checkpoint forward
    block = BasicBlock(64, 64, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_bottleneck():
    # Style must be in ['pytorch', 'caffe']
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, style='tensorflow')

    # Test invalid plugins cfg
    with pytest.raises(AssertionError):
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv4')
        ]
        Bottleneck(64, 16, plugins=plugins)

    # Test duplicate plugin name
    with pytest.raises(AssertionError):
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3'),
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3')
        ]
        Bottleneck(64, 16, plugins=plugins)

    # Plugin type is not supported
    with pytest.raises(KeyError):
        plugins = [dict(cfg=dict(type='WrongPlugin'), position='after_conv3')]
        Bottleneck(64, 16, plugins=plugins)

    # Test Bottleneck with checkpoint forward
    block = Bottleneck(64, 16, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck style
    block = Bottleneck(64, 64, stride=2, style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = Bottleneck(64, 64, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    # Test Bottleneck DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, dcn=dcn, conv_cfg=dict(type='Conv'))
    block = Bottleneck(64, 64, dcn=dcn)
    assert isinstance(block.conv2, DeformConv2dPack)

    # Test Bottleneck forward
    block = Bottleneck(64, 16)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck with 1 ContextBlock after conv3
    plugins = [
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            position='after_conv3')
    ]
    block = Bottleneck(64, 16, plugins=plugins)
    assert block.context_block.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnet_res_layer():
    # Test ResLayer of 3 Bottleneck w\o downsample
    layer = ResLayer(Bottleneck, 64, 16, 3)
    assert len(layer) == 3
    assert layer[0].conv1.in_channels == 64
    assert layer[0].conv1.out_channels == 16
    for i in range(1, len(layer)):
        assert layer[i].conv1.in_channels == 64
        assert layer[i].conv1.out_channels == 16
    for i in range(len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test ResLayer of 3 Bottleneck with downsample
    layer = ResLayer(Bottleneck, 64, 64, 3)
    assert layer[0].downsample[0].out_channels == 256
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 56, 56])

    # Test ResLayer of 3 Bottleneck with stride=2
    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2)
    assert layer[0].downsample[0].out_channels == 256
    assert layer[0].downsample[0].stride == (2, 2)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])

    # Test ResLayer of 3 Bottleneck with stride=2 and average downsample
    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2, avg_down=True)
    assert isinstance(layer[0].downsample[0], AvgPool2d)
    assert layer[0].downsample[1].out_channels == 256
    assert layer[0].downsample[1].stride == (1, 1)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])
