# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt

import torch
from mmcv.cnn import build_activation_layer
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
        scale_mode (str): How to scale correlation. The value includes
        `'dimension'` and `'sqrt dimension'`, but it doesn't work when
        scaled = True. Default to `'dimension'`.
    """

    def __init__(self,
                 corr_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 scaled: bool = False,
                 scale_mode: str = 'dimension') -> None:
        super().__init__()

        assert scale_mode in ('dimension', 'sqrt dimension'), (
            'scale_mode must be \'dimension\' or \'sqrt dimension\' '
            f'but got {scale_mode}')

        corr = build_operators(corr_cfg)
        act = build_activation_layer(act_cfg)
        self.scaled = scaled
        self.scale_mode = scale_mode

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
        scale_factor = 1.

        if self.scaled:

            if 'sqrt' in self.scale_mode:
                scale_factor = sqrt(float(C * self.kernel_size**2))
            else:
                scale_factor = float(C * self.kernel_size**2)

        corr = self.corr_block[0](feat1, feat2) / scale_factor

        corr = corr.view(N, -1, H // self.stride, W // self.stride)

        out = self.corr_block[1](corr)

        return out

    def __repr__(self):
        s = super().__repr__()
        s += f'\nscaled={self.scaled}'
        s += f'\nscale_mode={self.scale_mode}'
        return s
