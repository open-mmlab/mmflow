# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcv.runner import BaseModule
from torch import Tensor


class Correlation1D(BaseModule):
    """Correlation1D Module.

    The neck of Flow1D, which calculates correlation tensor of input features
    with the method of 3D cost volume.
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                feat1: Tensor,
                feat2: Tensor,
                y_direction: bool = False) -> Tensor:
        """Forward function for Correlation1D.

        Args:
            feat1 (Tensor): The feature from first input image.
            feat2 (Tensor): The 1D cross attention feat2 on x or y direction.
            y_direction (bool): whether y direction or not.
        Returns:
            Tensor: Correlation of x correlation or y correlation.
        """
        b, c, h, w = feat1.shape
        scale_factor = c**0.5
        if y_direction:
            # y direction, corr shape is  [B, W, H, H]
            feat1 = feat1.permute(0, 3, 2, 1)
            feat2 = feat2.permute(0, 3, 1, 2)
        else:
            # x direction, corr shape is  [B, H, W, W]
            feat1 = feat1.permute(0, 2, 3, 1)
            feat2 = feat2.permute(0, 2, 1, 3)
        corr = torch.matmul(feat1, feat2) / scale_factor
        return corr
