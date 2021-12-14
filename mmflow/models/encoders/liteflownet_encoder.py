# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from ..builder import ENCODERS
from ..utils import BasicEncoder


@ENCODERS.register_module()
class NetC(BasicEncoder):
    """The feature extraction sub-module in LiteFlowNet.

    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        num_convs (Sequence[int]): Numbers of conv layers for each
            pyramid level. Default: (1, 3, 2, 2, 1, 1).
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
            Default: (32, 32, 64, 96, 128, 192).
        strides (Sequence[int]): List of strides of each pyramid level.
            Default: (1, 2, 2, 2, 2, 2).
        kernel_size (Sequence[int]): List of kernel size of each feature level.
            Default: (7, 3, 3, 3, 3, 3).
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
                 out_channels: Sequence[int] = (32, 32, 64, 96, 128, 192),
                 strides: Sequence[int] = (1, 2, 2, 2, 2, 2),
                 num_convs: Sequence[int] = (1, 3, 2, 2, 1, 1),
                 kernel_size: Sequence[int] = (7, 3, 3, 3, 3, 3),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            in_channels=in_channels,
            pyramid_levels=pyramid_levels,
            num_convs=num_convs,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
