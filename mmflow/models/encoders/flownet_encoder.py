# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
from mmcv.cnn.bricks.conv_module import ConvModule

from ..builder import ENCODERS
from ..utils import BasicEncoder, CorrBlock


@ENCODERS.register_module()
class FlowNetEncoder(BasicEncoder):
    """The feature extraction sub-module of FlowNetS and FlowNetC.

    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        num_convs (Sequence[int]): Numbers of conv layers for each
            pyramid level. Default: (1, 1, 2, 2, 2, 2).
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
            Default: (64, 128, 256, 512, 512, 1024).
        kernel_size (Sequence): List of kernel size of each feature level.
            Default: (7, 5, (5, 3), 3, 3, 3).
        strides (Sequence[int]): List of strides of each pyramid level.
            Default: (2, 2, 2, 2, 2, 2).
        dilations (Sequence[int]): List of dilation of each pyramid level.
            Default: (1, 1, 1, 1, 1, 1).
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
                 num_convs: Sequence[int] = (1, 1, 2, 2, 2, 2),
                 out_channels: Sequence[int] = (64, 128, 256, 512, 512, 1024),
                 kernel_size: Sequence = (7, 5, (5, 3), 3, 3, 3),
                 strides: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        super().__init__(
            in_channels=in_channels,
            pyramid_levels=pyramid_levels,
            num_convs=num_convs,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)


@ENCODERS.register_module()
class CorrEncoder(BasicEncoder):
    """The Correlation feature extraction sub-module of FlowNetC..

    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): Number of pyramid levels.
        kernel_sizes (Sequence[int]): List of numbers of kernel size of each
            block. Default: (3, 3, 3, 3).
        num_convs (Sequence[int]): List of number of convolution layers.
            Default: (1, 2, 2, 2).
        out_channels (Sequence[int]): List of numbers of output channels of
            each ConvModule. Default: (256, 512, 512, 1024).
        redir_in_channels (int): Number of input channels of
            redirection ConvModule. Default: 256.
        redir_channels (int): Number of output channels of redirection
            ConvModule. Default: 32.
        strides (Sequence[int]): List of numbers of strides of each block.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): List of numbers of dilations of each block.
            Default: (1, 1, 1, 1).
        corr_cfg (dict): Config dict for correlation layer.
            Default: dict(type='Correlation', kernel_size=1, max_displacement
            =10, stride=1, padding=0, dilation_patch=2)
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not. Default: False.
        act_cfg (dict): Config for each activation layer in ConvModule.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        conv_cfg (dict, optional): Config for convolution layers.
            Default: None.
        norm_cfg (dict, optional): Config for each normalization layer.
            Default: None.
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 pyramid_levels: Sequence[str],
                 kernel_size: Sequence[int] = (3, 3, 3, 3),
                 num_convs: Sequence[int] = (1, 2, 2, 2),
                 out_channels: Sequence[int] = (256, 512, 512, 1024),
                 redir_in_channels: int = 256,
                 redir_channels: int = 32,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 corr_cfg: dict = dict(
                     type='Correlation',
                     kernel_size=1,
                     max_displacement=10,
                     stride=1,
                     padding=0,
                     dilation_patch=2),
                 scaled: bool = False,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        super().__init__(
            in_channels=in_channels,
            pyramid_levels=pyramid_levels,
            num_convs=num_convs,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.corr = CorrBlock(corr_cfg, act_cfg, scaled=scaled)

        self.conv_redir = ConvModule(
            in_channels=redir_in_channels,
            out_channels=redir_channels,
            kernel_size=1,
            act_cfg=act_cfg)

    def forward(self, f1: torch.Tensor,
                f2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward function for CorrEncoder.

        Args:
            f1 (Tensor): The feature from the first input image.
            f2 (Tensor): The feature from the second input image.

        Returns:
            Dict[str, Tensor]: The feature pyramid for correlation.
        """

        corr_feat = self.corr(f1, f2)
        redir_feat = self.conv_redir(f1)

        x = torch.cat((redir_feat, corr_feat), dim=1)

        outs = dict()
        for i, convs_layer in enumerate(self.layers):
            x = convs_layer(x)
            # After correlation, the feature level starts at level3
            if 'level' + str(i + 3) in self.pyramid_levels:
                outs['level' + str(i + 3)] = x

        return outs


@ENCODERS.register_module()
class FlowNetSDEncoder(BasicEncoder):
    """The feature extraction sub-module of FlowNetSD.

    Args:
        in_channels (int): Number of input channels.
        plugin_channels (int): The output channels of plugin convolution layer.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        num_convs (Sequence[int]): Numbers of conv layers for each
            pyramid level. Default: (2, 2, 2, 2, 2, 2).
        out_channels (Sequence): List of numbers of output
            channels of each pyramid level.
            Default: ((64, 128), 128, 256, 512, 512, 1024).
        kernel_size (int): Kernel size of each feature level. Default: 3.
        strides (Sequence[int]): List of strides of each pyramid level.
            Default: (2, 2, 2, 2, 2, 2).
        dilations (Sequence[int]): List of dilation of each pyramid level.
            Default: (1, 1, 1, 1, 1, 1).
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
                 plugin_channels: int,
                 pyramid_levels: Sequence[str],
                 num_convs: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 out_channels: Sequence = ((64, 128), 128, 256, 512, 512,
                                           1024),
                 kernel_size: int = 3,
                 strides: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            plugin_channels,
            pyramid_levels,
            num_convs=num_convs,
            out_channels=out_channels,
            strides=strides,
            dilations=dilations,
            kernel_size=kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.plugin_layer = ConvModule(
            in_channels=in_channels,
            out_channels=plugin_channels,
            act_cfg=act_cfg,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward function for FlowNetSDEncoder.

        Args:
            imgs (Tensor): The concatenate images.

        Returns:
            Dict[str, Tensor]: The feature pyramid extracted from images.
        """
        outs = dict()
        x = self.plugin_layer(imgs)
        for i, convs_layer in enumerate(self.layers):
            x = convs_layer(x)
            if 'level' + str(i + 1) in self.pyramid_levels:
                outs['level' + str(i + 1)] = x

        return outs
