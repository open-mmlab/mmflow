# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from mmflow.ops import build_operators


def flow_to_coords(flow: Tensor) -> Tensor:
    """Generate shifted coordinate grid based based input flow.

    Args:
        flow (Tensor): Estimated optical flow.

    Returns:
        Tensor: Coordinate that shifted by input flow with shape (B, 2, H, W).
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(B, 1, 1, 1) + flow
    return coords


def compute_range_map(flow: Tensor, **kwargs) -> Tensor:
    """Compute range map.

    Args:
        flow (Tensor): The backward flow with shape (N, 2, H, W)
        win_size (int, tuple): The window size for calculating range map.

    Return:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """
    win_size = kwargs['win_size']
    assert isinstance(win_size, (tuple, int)), \
        f'win_size must be a tuple or int, but got {type(win_size)}'

    if isinstance(win_size, int):
        win_size = (win_size, win_size)

    win_h, win_w = win_size

    N, _, H, W = flow

    coords = flow_to_coords(flow)

    # Split coordinates into an integer part and
    # a float offset for interpolation.
    coords_floor = torch.floor(coords)
    coords_offset = coords - coords_floor
    coords_floor = coords_floor.to(torch.int32)

    # Define a batch offset for flattened indexes into all pixels.
    batch_range = torch.arange(N).view(N, 1, 1)
    idx_batch_offset = batch_range.repeat(1, H, W) * H * W

    # Flatten everything.
    coords_floor_flattened = coords_floor.reshape(N, 2, -1)
    coords_offset_flattened = coords_offset.reshape(N, 2, -1)
    idx_batch_offset_flattened = idx_batch_offset.reshape(N, -1)

    # Initialize results.
    idxs_list = []
    weights_list = []

    # Loop over differences di and dj to the four neighboring pixels.
    for di in range(win_h):
        for dj in range(win_w):
            # Compute the neighboring pixel coordinates.
            idxs_i = coords_floor_flattened[:, 0, ...] + di
            idxs_j = coords_floor_flattened[:, 1, ...] + dj
            # Compute the flat index into all pixels.
            idxs = idx_batch_offset_flattened + idxs_i * W + idxs_j

            # Only count valid pixels.
            mask = torch.where(
                torch.logical_and(
                    torch.logical_and(idxs_i >= 0, idxs_i < H),
                    torch.logical_and(idxs_j >= 0, idxs_j < W))).reshape(-1)
            valid_idxs = torch.select(idxs, mask)
            valid_offsets = torch.select(coords_offset_flattened, mask)

            # Compute weights according to bilinear interpolation.
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0, ...]
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1, ...]
            weights = weights_i * weights_j

            # Append indices and weights to the corresponding list.
            idxs_list.append(valid_idxs)
            weights_list.append(weights)
    # Concatenate everything.
    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)

    # Sum up weights for each pixel and reshape the result.
    count_image = torch.zeros_like(weights)
    count_image.index_add_(
        dim=1, index=idxs, source=weights)[:N * H * W].reshape(N, 1, H, W)

    return count_image


def forward_backward_consistency(flow_fw: Tensor, flow_bw: Tensor,
                                 **kwarg) -> Tensor:
    """Occlusion mask from forward-backward consistency.

    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)

    Returns:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    warp = build_operators(kwarg['warp_cfg'])

    warped_flow_bw = warp(flow_bw)

    forward_backward_sq_diff = torch.sum(
        (flow_fw + warped_flow_bw)**2, dim=1, keepdim=True)
    forward_backward_sum_sq = torch.sum(
        flow_fw * 2 + warped_flow_bw**2, dim=1, keepdim=True)

    occ = (forward_backward_sq_diff <
           forward_backward_sum_sq * 0.01 + 0.5).to(flow_fw)
    return occ


def forward_backward_absdiff(flow_fw: Tensor, flow_bw: Tensor,
                             **kwarg) -> Tensor:
    """Occlusion mask from forward-backward consistency.

    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)

    Returns:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    warp = build_operators(kwarg['warp_cfg'])

    warped_flow_bw = warp(flow_bw)

    forward_backward_sq_diff = torch.sum(
        (flow_fw + warped_flow_bw)**2, dim=1, keepdim=True)

    occ = (forward_backward_sq_diff**0.5 < 1.5).to(flow_fw)

    return occ


def occlusion_estimation(flow_fw: Tensor,
                         flow_bw: Tensor,
                         mode: str = 'consistency',
                         **kwarg) -> Dict[str, Tensor]:
    """Occlusion estimation.

    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)
        mode (str): The method for occlusion estimation, which can be
            ``'consistency'``, ``'range_map'`` or ``'fb_abs'``.
        warp_cfg (dict, optional): _description_. Defaults to None.

    Returns:
        Dict[str,Tensor]: 1 denote non-occluded and 0 denote occluded
    """
    assert mode in ('consistency', 'range_map', 'fb_abs'), \
        'mode must be \'consistency\', \'range_map\' or \'fb_abs\', ' \
        f'but got {mode}'

    if mode == 'consistency':
        occ_fw = forward_backward_consistency(flow_fw, flow_bw, **kwarg)
        occ_bw = forward_backward_consistency(flow_bw, flow_fw, **kwarg)

    elif mode == 'range_map':
        occ_fw = compute_range_map(flow_bw)
        occ_bw = compute_range_map(flow_fw)

    elif mode == 'fb_abs':
        occ_fw = forward_backward_absdiff(flow_fw, flow_bw)
        occ_bw = forward_backward_absdiff(flow_bw, flow_fw)

    return dict(occ_fw=occ_fw, occ_bw=occ_bw)
