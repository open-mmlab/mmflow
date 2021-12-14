# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class BasicConvBlock(BaseModule):
    """Basic convolution block for PWC-Net.

    This module consists of several plain convolution layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolution layers. Default: 3.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolution layer to downsample the input feature
            map. Options are 1 or 2. Default: 2.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolution layer and
            the dilation rate of the first convolution layer is always 1.
            Default: 1.
        kernel_size (int): Kernel size of each feature level. Default: 3.
        conv_cfg (dict , optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_convs: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 kernel_size: int = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None) -> None:
        super(BasicConvBlock, self).__init__()

        convs = []
        in_channels = in_channels
        for i in range(num_convs):
            k = kernel_size[i] if isinstance(kernel_size,
                                             (tuple, list)) else kernel_size
            out_ch = out_channels[i] if isinstance(out_channels,
                                                   (tuple,
                                                    list)) else out_channels

            convs.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=k // 2 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = out_ch

        self.layers = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""

        out = self.layers(x)
        return out


class BasicEncoder(BaseModule):
    """A basic pyramid feature extraction sub-network.

    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        num_convs (Sequence[int]): Numbers of conv layers for each
            pyramid level. Default: (3, 3, 3, 3, 3, 3).
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
            Default: (16, 32, 64, 96, 128, 196).
        strides (Sequence[int]): List of strides of each pyramid level.
            Default: (2, 2, 2, 2, 2, 2).
        dilations (Sequence[int]): List of dilation of each pyramid level.
            Default: (1, 1, 1, 1, 1, 1).
        kernel_size (Sequence, int): Kernel size of each feature
            level. Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict): Config dict for each activation layer in ConvModule.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 pyramid_levels: Sequence[str],
                 num_convs: Sequence[int] = (3, 3, 3, 3, 3, 3),
                 out_channels: Sequence[int] = (16, 32, 64, 96, 128, 196),
                 strides: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 kernel_size: Union[Sequence, int] = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg)

        assert len(out_channels) == len(num_convs) == len(strides) == len(
            dilations) == len(pyramid_levels)
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        self.out_channels = out_channels
        self.num_convs = num_convs
        self.strides = strides
        self.dilations = dilations

        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        convs = []
        for i in range(len(out_channels)):
            if isinstance(self.kernel_size, (list, tuple)) and len(
                    self.kernel_size) == len(out_channels):
                kernel_size_ = self.kernel_size[i]
            elif isinstance(self.kernel_size, int):
                kernel_size_ = self.kernel_size
            else:
                TypeError('kernel_size must be list, tuple or int, '
                          f'but got {type(kernel_size)}')

            convs.append(
                self._make_layer(
                    in_channels,
                    out_channels[i],
                    num_convs[i],
                    strides[i],
                    dilations[i],
                    kernel_size=kernel_size_))
            in_channels = out_channels[i][-1] if isinstance(
                out_channels[i], (tuple, list)) else out_channels[i]

        self.layers = nn.Sequential(*convs)

    def _make_layer(self,
                    in_channels: int,
                    out_channel: int,
                    num_convs: int,
                    stride: int,
                    dilation: int,
                    kernel_size: int = 3) -> torch.nn.Module:
        return BasicConvBlock(
            in_channels=in_channels,
            out_channels=out_channel,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward function for BasicEncoder.

        Args:
            x (Tensor): The input data.

        Returns:
            dict: The feature pyramid extracted from input data.
        """
        outs = dict()
        for i, convs_layer in enumerate(self.layers):
            x = convs_layer(x)
            if 'level' + str(i + 1) in self.pyramid_levels:
                outs['level' + str(i + 1)] = x

        return outs
