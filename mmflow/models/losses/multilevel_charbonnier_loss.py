# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..builder import LOSSES
from .multilevel_flow_loss import multi_level_flow_loss


def charbonnier_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     q: float = 0.2,
                     eps: float = 0.01) -> torch.Tensor:
    """Generalized Charbonnier loss function between output and ground truth.

    The loss function is
    .. math::
      loss = ((u-u_gt)^2+(v-v_gt)^2+eps)^q

    Generalized Charbonnier loss was used in LiteFlowNet when fine tuning,
    with eps=0.01 q=0.2.

    Args:
        pred (torch.Tensor): output flow map from flow_estimator
            shape(B, 2, H, W).
        target (torch.Tensor): ground truth flow map shape(B, 2, H, W).
        q (float): the exponent in charbonnier loss.
        eps (float): small constant to numerical stability when
            fine-tuning model. Defaults to 0.01.

    Returns:
        Tensor: loss map with the shape (B, H, W).
    """

    assert pred.shape == target.shape, \
        (f'pred shape {pred.shape} does not match target '
         f'shape {target.shape}.')

    diff = torch.add(pred, -target)

    loss_map = (torch.sum(diff * diff, dim=1) + eps)**q  # shape (B, H, W).

    return loss_map


@LOSSES.register_module()
class MultiLevelCharbonnierLoss(nn.Module):
    """Multi-level Generalized Charbonnier loss.

    Args:

        q (float): the exponent in charbonnier loss.
        eps (float): small constant to numerical stability when
            fine-tuning model. Defaults to 0.01.
        weights (dict): manual rescaling weights given to the loss of flow map
            at each level, and the keys in weights must correspond to predicted
            dict. Defaults to dict(
            level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005).
        flow_div (float): the divisor used to scale down ground truth.
            Defaults to 20.
        max_flow (float): maximum value of optical flow, if some pixel's flow
            of target is larger than it, this pixel is not valid. Default to
            float('inf').
        resize_flow (str): mode for reszing flow: 'downsample' and 'upsample',
            as multi-level predicted outputs don't match the ground truth.
            If set to 'downsample', it will downsample the ground truth, and
            if set to 'upsample' it will upsample the predicted flow, and
            'upsample' is used for sparse flow map as no generic interpolation
            mode can resize a ground truth of sparse flow correctly.
            Default to 'downsample'.
        scale_as_level (bool): Whether flow for each level is at its native
            spatial resolution. If `'scale_as_level'` is True, the ground
            truth is scaled at different levels, if it is False, the ground
            truth will not be scaled. Default to False.
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.
    """

    def __init__(self,
                 q: float = 0.2,
                 eps: float = 0.01,
                 flow_div: float = 20.,
                 weights: Dict[str, float] = dict(
                     level6=0.32,
                     level5=0.08,
                     level4=0.02,
                     level3=0.01,
                     level2=0.005),
                 max_flow: float = float('inf'),
                 resize_flow: str = 'downsample',
                 scale_as_level: bool = False,
                 reduction: str = 'sum') -> None:
        super().__init__()

        assert isinstance(q, float) and q > 0.
        self.q = q

        assert isinstance(eps, float) and eps > 0.
        self.eps = eps

        assert flow_div > 0
        self.flow_div = flow_div

        assert isinstance(weights, dict)
        self.weights = weights

        assert max_flow > 0.
        self.max_flow = max_flow

        assert resize_flow in ('downsample', 'upsample')
        self.resize_flow = resize_flow

        assert isinstance(scale_as_level, bool)
        self.scale_as_level = scale_as_level

        assert reduction in ('mean', 'sum')
        self.reduction = reduction

    def forward(self,
                pred: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                target: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forwar function for MultiLevelCharbonnierLoss.

        Args:
            preds_dict (dict): Multi-level output of predicted optical flow,
                and the contain in dict is a Tensor or list of Tensor with
                shape (B, 1, H_l, W_l), where l indicates the level.
            target (Tensor): Ground truth of optical flow with shape
                (B, 2, H, W).
            valid (Tensor, optional): Valid mask for optical flow.
                Defaults to None.

        Returns:
            Tensor: value of pixel-wise generalized Charbonnier loss.
        """

        return multi_level_flow_loss(
            charbonnier_loss,
            pred,
            target,
            weights=self.weights,
            valid=valid,
            flow_div=self.flow_div,
            max_flow=self.max_flow,
            resize_flow=self.resize_flow,
            scale_as_level=self.scale_as_level,
            reduction=self.reduction,
            q=self.q,
            eps=self.eps,
        )

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__
        repr_str += (f'(resize_flow={self.resize_flow}, '
                     f'scale_as_level={self.scale_as_level}, '
                     f'flow_div={self.flow_div}, '
                     f'weights={self.weights}, '
                     f'q={self.q}, '
                     f'eps={self.eps}, '
                     f'reduction=\'{self.reduction}\')')

        return repr_str
