# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class DenseLayer(BaseModule):
    """Densely connected layer.

    Args:
        in_channels (int): Input channels of convolution module.
        feat_channels (int): Output channel of convolution module.
        conv_cfg (dict, optional): Config of convolution layer in module.
            Default: None.
        norm_cfg (dict, optional): Config of norm layer in module.
            Default: None.
        act_cfg (dict): Config of activation layer in module.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config dict of initialization of
            BaseModule. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layers = ConvModule(
            in_channels=in_channels,
            out_channels=feat_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for DenseLayer.

        Args:
            x (Tensor): The input feature.

        Returns:
            Tensor: The output feature of DenseLayer.
        """
        out = self.layers(x)
        return torch.cat((out, x), dim=1)


class BasicDenseBlock(BaseModule):
    """Basic Dense Block.

    A basic block which consists of several dense layers.

    Args:
        in_channels (int): Input channels of the block.
        feat_channels (Sequence[int]): Output channels of convolution module
            in dense layers. Default: (128, 128, 96, 64, 32).
        conv_cfg (dict, optional): Config of convolution layer in dense layers.
            Default: None.
        norm_cfg (dict, optional): Config of norm layer in dense layers.
            Default: None.
        act_cfg (dict, optional): Config of activation layer in dense layers.
            Default: None.
        init_cfg (dict, list, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        layers = []
        for _feat_channels in feat_channels:
            layers.append(
                DenseLayer(in_channels, _feat_channels, conv_cfg, norm_cfg,
                           act_cfg))
            in_channels += _feat_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for BasicDenseBlock.

        Args:
            x (Tensor): The input feature.

        Returns:
            Tensor: The output feature of BasicDenseBlock.
        """
        return self.layers(x)
