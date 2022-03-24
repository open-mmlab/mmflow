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


def compute_range_map(flow: Tensor) -> Tensor:
    """Compute range map.

    Args:
        flow (Tensor): The backward flow with shape (N, 2, H, W)

    Return:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    N, _, H, W = flow.shape

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
    coords_floor_flattened = coords_floor.permute(0, 2, 3, 1).reshape(-1, 2)
    coords_offset_flattened = coords_offset.permute(0, 2, 3, 1).reshape(-1, 2)
    idx_batch_offset_flattened = idx_batch_offset.reshape(-1)

    # Initialize results.
    idxs_list = []
    weights_list = []

    # Loop over differences di and dj to the four neighboring pixels.
    for di in range(2):
        for dj in range(2):
            # Compute the neighboring pixel coordinates.
            idxs_j = coords_floor_flattened[..., 0] + dj
            idxs_i = coords_floor_flattened[..., 1] + di
            # Compute the flat index into all pixels.
            idxs = idx_batch_offset_flattened + idxs_i * W + idxs_j

            # Only count valid pixels.
            mask = torch.logical_and(
                torch.logical_and(idxs_j >= 0, idxs_j < W),
                torch.logical_and(idxs_i >= 0, idxs_i < H))
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flattened[mask]

            # Compute weights according to bilinear interpolation.
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 0]
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 1]
            weights = weights_i * weights_j

            # Append indices and weights to the corresponding list.
            idxs_list.append(valid_idxs)
            weights_list.append(weights)
    # Concatenate everything.
    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)

    # Sum up weights for each pixel and reshape the result.
    count_image = torch.zeros(N * H * W)
    count_image = count_image.index_add_(
        dim=0, index=idxs, source=weights).reshape(N, H, W)
    occ = (count_image >= 1).to(flow)[:, None, ...]
    return occ


def forward_backward_consistency(
        flow_fw: Tensor,
        flow_bw: Tensor,
        warp_cfg: dict = dict(type='Warp', align_corners=True),
) -> Tensor:
    """Occlusion mask from forward-backward consistency.

    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)

    Returns:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    warp = build_operators(warp_cfg)

    warped_flow_bw = warp(flow_bw, flow_fw)

    forward_backward_sq_diff = torch.sum(
        (flow_fw + warped_flow_bw)**2, dim=1, keepdim=True)
    forward_backward_sum_sq = torch.sum(
        flow_fw * 2 + warped_flow_bw**2, dim=1, keepdim=True)

    occ = (forward_backward_sq_diff <
           forward_backward_sum_sq * 0.01 + 0.5).to(flow_fw)
    return occ


def forward_backward_absdiff(flow_fw: Tensor,
                             flow_bw: Tensor,
                             warp_cfg: dict = dict(
                                 type='Warp', align_corners=True),
                             diff: int = 1.5) -> Tensor:
    """Occlusion mask from forward-backward consistency.

    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)

    Returns:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    warp = build_operators(warp_cfg)

    warped_flow_bw = warp(flow_bw, flow_fw)

    forward_backward_sq_diff = torch.sum(
        (flow_fw + warped_flow_bw)**2, dim=1, keepdim=True)

    occ = (forward_backward_sq_diff**0.5 < diff).to(flow_fw)

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
        occ_fw = forward_backward_absdiff(flow_fw, flow_bw, **kwarg)
        occ_bw = forward_backward_absdiff(flow_bw, flow_fw, **kwarg)

    return dict(occ_fw=occ_fw, occ_bw=occ_bw)
