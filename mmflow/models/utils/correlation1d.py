# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

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

    def forward(
        self,
        feat1: Tensor,
        feat2_x: Tensor,
        feat2_y: Tensor,
    ) -> Sequence[Tensor]:
        """Forward function for Correlation1D.

        Args:
            feat1 (Tensor): The feature from first input image.
            feat2_x (Tensor): The 1D cross attention feature2 on x direction.
            feat2_y (Tensor): The 1D cross attention feature2 on y direction.

        Returns:
            Sequence[Tensor]: Correlation list, include x correlation
            and y correlation.
        """
        corr_x = self.corr_x(feat1, feat2_x)
        corr_y = self.corr_y(feat1, feat2_y)
        corr = [corr_x, corr_y]
        return corr

    @staticmethod
    def corr_x(feature1: Tensor, feature2: Tensor) -> Tensor:
        """corr_x function for Correlation1D.

        Args:
            feature1 (Tensor): Input feature1.
            feature2 (Tensor): Input feature2.

        Returns:
            Tensor: x correlation.
        """
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c**0.5

        # x direction, corr shape is  [B, H, W, W]
        feature1 = feature1.permute(0, 2, 3, 1)
        feature2 = feature2.permute(0, 2, 1, 3)
        corr = torch.matmul(feature1, feature2)

        # reshape to [B*H*W, 1, 1, W]
        corr = corr.unsqueeze(3).unsqueeze(3)
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)

        return corr

    @staticmethod
    def corr_y(feature1: Tensor, feature2: Tensor) -> Tensor:
        """corr_y function for Correlation1D.

        Args:
            feature1 (Tensor): Input feature1.
            feature2 (Tensor): Input feature2.

        Returns:
            Tensor: y correlation.
        """
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c**0.5

        # y direction, corr shape is  [B, W, H, H]
        feature1 = feature1.permute(0, 3, 2, 1)
        feature2 = feature2.permute(0, 3, 1, 2)
        corr = torch.matmul(feature1, feature2)

        # reshape to [B*H*W, 1, H, 1]
        corr = corr.permute(0, 2, 1, 3).contiguous().view(b, h, w, 1, h, 1)
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)

        return corr
