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


@OPERATORS.register_module()
class CorrLookupFlow1D(nn.Module):
    """Correlation lookup operator for Flow1D.

    This operator is used in `Flow1D<https://arxiv.org/pdf/2104.13918.pdf>`_

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

    def forward(self, corr: Sequence[Tensor], flow: Tensor) -> Tensor:
        """Forward function of Correlation lookup for Flow1D.

        Args:
            corr (Sequence[Tensor]): Correlation on x and y direction.
            flow (Tensor): Current estimated optical flow.

        Returns:
            Tensor: lookup cost volume on the correlation of x and y directions
             concatenate together.
        """
        B, _, H, W = flow.shape
        # reshape corr_x to [B*H*W, 1, 1, W]
        corr_x = corr[0].view(-1, 1, 1, W)
        # reshape corr_y to [B*H*W, 1, H, 1]
        corr_y = corr[1].permute(0, 2, 1, 3).contiguous().view(-1, 1, H, 1)

        # reshape flow to [B, H, W, 2]
        flow = flow.permute(0, 2, 3, 1)
        coords_x = flow[:, :, :, 0]
        coords_y = flow[:, :, :, 1]
        coords_x = torch.stack((coords_x, torch.zeros_like(coords_x)), dim=-1)
        coords_y = torch.stack((torch.zeros_like(coords_y), coords_y), dim=-1)
        centroid_x = coords_x.view(B * H * W, 1, 1, 2)
        centroid_y = coords_y.view(B * H * W, 1, 1, 2)

        dx = torch.linspace(
            -self.r, self.r, 2 * self.r + 1, device=flow.device)
        dy = torch.linspace(
            -self.r, self.r, 2 * self.r + 1, device=flow.device)

        delta_x = torch.stack((dx, torch.zeros_like(dx)), dim=-1)
        delta_y = torch.stack((torch.zeros_like(dy), dy), dim=-1)
        # [1, 2r+1, 1, 2]
        delta_y = delta_y.view(1, 2 * self.r + 1, 1, 2)

        coords_x = centroid_x + delta_x
        coords_y = centroid_y + delta_y

        corr_x = bilinear_sample(corr_x, coords_x, self.mode,
                                 self.padding_mode, self.align_corners)
        corr_y = bilinear_sample(corr_y, coords_y, self.mode,
                                 self.padding_mode, self.align_corners)

        # shape is [B, 2r+1, H, W]
        corr_x = corr_x.view(B, H, W, -1)
        corr_x = corr_x.permute(0, 3, 1, 2).contiguous()
        corr_y = corr_y.view(B, H, W, -1)
        corr_y = corr_y.permute(0, 3, 1, 2).contiguous()

        correlation = torch.cat((corr_x, corr_y), dim=1)

        return correlation
