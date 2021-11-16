# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer
from mmcv.runner import BaseModule

from mmflow.ops import build_operators


class CorrBlock(BaseModule):
    """Basic Correlation Block.

    A block used to calculate correlation.

    Args:
        corr (dict): Config dict for build correlation operator.
        act_cfg (dict): Config dict for activation layer.
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Default: False.
    """

    def __init__(self,
                 corr_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 scaled: bool = False) -> None:
        super().__init__()
        corr = build_operators(corr_cfg)
        act = build_activation_layer(act_cfg)
        self.scaled = scaled

        self.kernel_size = corr.kernel_size
        self.corr_block = [corr, act]
        self.stride = corr_cfg.get('stride', 1)

    def forward(self, feat1: torch.Tensor,
                feat2: torch.Tensor) -> torch.Tensor:
        """Forward function for CorrBlock.

        Args:
            feat1 (Tensor): The feature from the first image.
            feat2 (Tensor): The feature from the second image.

        Returns:
            Tensor: The correlation between feature1 and feature2.
        """
        N, C, H, W = feat1.shape
        if self.scaled:
            corr = self.corr_block[0](feat1, feat2) / float(
                C * self.kernel_size**2)
        else:
            corr = self.corr_block[0](feat1, feat2)
        corr = corr.view(N, -1, H // self.stride, W // self.stride)
        out = self.corr_block[1](corr)

        return out

    def __repr__(self):
        s = super().__repr__()
        s += f'\nscaled={self.scaled}'
        return s


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
