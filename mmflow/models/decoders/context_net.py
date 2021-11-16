# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import COMPONENTS


@COMPONENTS.register_module()
class ContextNet(BaseModule):
    """The Context network to exploit contextual information for PWC to refine
    the optical flow.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Default: 2.
        feat_channels (Sequence[int]): List of numbers of outputs feature
            channels. Default: (128, 128, 128, 96, 64, 32).
        dilation (Sequence[int]): List of dilation of each layer. Default:
            (1, 2, 4, 8, 16, 1).
        conv_cfg (dict , optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='LeakyReLU').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 2,
                 feat_channels: Sequence[int] = (128, 128, 128, 96, 64, 32),
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, Sequence]] = None) -> None:

        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)
        layers = []
        for _feat_channels, _dilation in zip(feat_channels, dilations):
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=_feat_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=_dilation,
                    padding=_dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = _feat_channels

        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True))
        self.out_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for Context network.

        Args:
            x (Tensor): Input feature.

        Returns:
            Tensor: The predicted result.
        """
        return self.layers(x)
