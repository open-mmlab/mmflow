# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..builder import LOSSES


def sequence_loss(preds, flow_gt, gamma, valid=None, max_flow=400):
    """Compute sequence loss between prediction and ground truth.

    Args:
        preds (list(torch.Tensor)): List of flow prediction from
            flow_estimator.
        flow_gt (torch.Tensor): Ground truth flow map.
        gamma (float): Scale factor gamma in loss calculation.
        valid (torch.Tensor, optional): Tensor Used to exclude invalid pixels.
            Default: None.
        max_flow (int, optional): Used to exclude extremely large
            displacements. Default: 400.

    Returns:
        flow_loss (float): Total sequence loss.
    """
    n_preds = len(preds)
    flow_loss = 0.
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid is None:
        valid = torch.ones(flow_gt[:, 0, :, :].shape).to(flow_gt)
    else:
        valid = ((valid >= 0.5) & (mag < max_flow)).to(flow_gt)

    for i, pred in enumerate(preds):
        i_weight = gamma**(n_preds - i - 1)
        i_loss = (pred - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
    return flow_loss


@LOSSES.register_module()
class SequenceLoss(nn.Module):
    """Sequence Loss for RAFT.

    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_flow (float): The maximum value of optical flow, if some pixel's
            flow of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, gamma: float = 0.8, max_flow: float = 400.) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self,
                flow_preds: Sequence[Tensor],
                flow_gt: Tensor,
                valid: Tensor = None) -> Tensor:
        """Forward function for MultiLevelEPE.

        Args:
            preds_dict Sequence[Tensor]: The list of predicted optical flow.
            target (Tensor): Ground truth of optical flow with shape
                (B, 2, H, W).
            valid (Tensor, optional): Valid mask for optical flow.
                Defaults to None.

        Returns:
            Tensor: value of pixel-wise end point error loss.
        """
        return sequence_loss(flow_preds, flow_gt, self.gamma, valid,
                             self.max_flow)
