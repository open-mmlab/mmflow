# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from ..builder import ENCODERS
from ..utils import BasicEncoder


@ENCODERS.register_module()
class PWCNetEncoder(BasicEncoder):
    """The feature extraction sub-module in PWC-Net.

    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        net_type (str): The type of this sub-module, if net_type is Basic, the
            the number of convolution layers of each level is 3, if net_type is
            Small, the the number of convolution layers of each level is 2.
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
            Default: (16, 32, 64, 96, 128, 196).
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
        init_cfg (dict, optional): Config of weights initialization. Default:
            None.
    """
    _arch_settings = {'Basic': (3, 3, 3, 3, 3, 3), 'Small': (2, 2, 2, 2, 2, 2)}

    def __init__(self,
                 in_channels: int,
                 pyramid_levels: Sequence[str],
                 net_type: str = 'Basic',
                 out_channels: Sequence[int] = (16, 32, 64, 96, 128, 196),
                 strides: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        if net_type not in self._arch_settings:
            raise KeyError(f'invalid net type {net_type} for PWC-Net')

        num_convs = self._arch_settings[net_type]

        super().__init__(
            in_channels=in_channels,
            pyramid_levels=pyramid_levels,
            num_convs=num_convs,
            out_channels=out_channels,
            strides=strides,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
