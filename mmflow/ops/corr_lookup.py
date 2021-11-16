# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .builder import OPERATORS


def coords_grid(batch: int, xx: Tensor, yy: Tensor) -> Tensor:
    """Coordinate grid.

    Args:
        batch (int): The batch size of feature.
        xx (Tensor): 1-D tensor of size W with values from the interval
            [0, W-1].
        yy (Tensor): 1-D tensor of size H with values from the interval
            [0, H-1].

    Returns:
        Tensor: Tensor of shape (batch, 2, H, W) with values of items'
            coordinate.
    """
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)  # shape(batch, 2, H, W)


def bilinear_sample(feat: Tensor,
                    grid: Tensor,
                    mode: str = 'bilinear',
                    padding_mode: str = 'zeros',
                    align_corners: bool = False,
                    scale: bool = True) -> Tensor:
    """Computes the output using input feature values and pixel locations from
    grid.

    Args:
        feat (Tensor): The input feature.
        grid (Tensor): The coordinate grid or the scaled coordinate that has
            values in the range of [-1, 1].
        mode (str): Interpolation mode to calculate output values.
            Defaults to 'bilinear'.
        padding_mode (str): Padding mode for outside grid values.
            Defaults to 'zeros'.
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s corner
            pixels. If set to False, they are instead considered as referring
            to the corner points of the input’s corner pixels, making the
            sampling more resolution agnostic. Default to False.
        scale (bool): Whether scale the grid values in the range of [-1, 1].
            Defaults to True.

    Returns:
        Tensor: The output tensor using input feature values and pixel
            locations from grid
    """
    H, W = feat.shape[-2:]
    if grid.shape[-1] != 2:
        grid = grid.permute(0, 2, 3, 1)
    if scale:
        grid[:, :, :, 0] = grid[:, :, :, 0] * 2. / max(W - 1, 1) - 1.
        grid[:, :, :, 1] = grid[:, :, :, 1] * 2. / max(H - 1, 1) - 1.

    return F.grid_sample(feat, grid, mode, padding_mode, align_corners)


@OPERATORS.register_module()
class CorrLookup(nn.Module):
    """Correlation lookup operator.

    This operator is used in `RAFT<https://arxiv.org/pdf/2003.12039.pdf>`_

    Args:
        radius (int): the radius of the local neighborhood of the pixels.
            Default to 4.
        mode (str): interpolation mode to calculate output values 'bilinear'
            | 'nearest' | 'bicubic'. Default: 'bilinear' Note: mode='bicubic'
            supports only 4-D input.
        padding_mode (str): padding mode for outside grid values 'zeros' |
            'border' | 'reflection'. Default: 'zeros'
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s corner
            pixels. If set to False, they are instead considered as referring
            to the corner points of the input’s corner pixels, making the
            sampling more resolution agnostic. Default to True.
    """

    def __init__(self,
                 radius: int = 4,
                 mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = True) -> None:
        super().__init__()
        self.r = radius
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, corr_pyramid: Sequence[Tensor], flow: Tensor) -> Tensor:
        """Forward function of Correlation lookup.

        Args:
            corr_pyramid (Sequence[Tensor]): Correlation pyramid.
            flow (Tensor): Current estimated optical flow.

        Returns:
            Tensor: Feature map by indexing from the correlation pyramid.
        """
        B, _, H, W = flow.shape
        xx = torch.arange(0, W, device=flow.device)
        yy = torch.arange(0, H, device=flow.device)
        grid = coords_grid(B, xx, yy) + flow  # shape N, 2, H, W
        grid = grid.permute(0, 2, 3, 1)  # shape N, H, W, 2

        dx = torch.linspace(
            -self.r, self.r, 2 * self.r + 1, device=flow.device)
        dy = torch.linspace(
            -self.r, self.r, 2 * self.r + 1, device=flow.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta_lvl = delta.view(1, 2 * self.r + 1, 2 * self.r + 1, 2)

        out_corr_pyramid = []
        for i, corr in enumerate(corr_pyramid):
            centroid_lvl = grid.reshape(B * H * W, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sample(corr, coords_lvl, self.mode,
                                   self.padding_mode, self.align_corners)
            corr = corr.view(B, H, W, -1)
            out_corr_pyramid.append(corr)

        out = torch.cat(out_corr_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
