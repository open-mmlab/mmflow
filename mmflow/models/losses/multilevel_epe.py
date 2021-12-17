# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..builder import LOSSES
from .multilevel_flow_loss import multi_level_flow_loss


def endpoint_error(pred: torch.Tensor,
                   target: torch.Tensor,
                   p: int = 2,
                   q: Optional[float] = None,
                   eps: Optional[float] = None) -> torch.Tensor:
    r"""Calculate end point errors between prediction and ground truth.

    If not define q, the loss function is
    .. math::
      loss = \Vert \mathbf{u}-\mathbf{u_gt} \Vert^p

    otherwise,
    .. math::
      loss = (\Vert \mathbf{u}-\mathbf{u_gt} \Vert^p+eps)^q

    For PWC-Net L2 norm loss: p=2, for the robust loss function p=1, q=0.4,
    eps=0.01.

    Args:
        pred (torch.Tensor): output flow map from flow_estimator
            shape(B, 2, H, W).
        target (torch.Tensor): ground truth flow map shape(B, 2, H, W).
        p (int): norm degree for loss. Options are 1 or 2. Defaults to 2.
        q (float, optional): used to give less penalty to outliers when
            fine-tuning model. Defaults to 0.4.
        eps (float, optional): a small constant to numerical stability when
            fine-tuning model. Defaults to 0.01.

    Returns:
        Tensor: end-point error map with the shape (B, H, W).
    """

    assert pred.shape == target.shape, \
        (f'pred shape {pred.shape} does not match target '
         f'shape {target.shape}.')

    epe_map = torch.norm(pred - target, p, dim=1)  # shape (B, H, W).

    if q is not None and eps is not None:
        epe_map = (epe_map + eps)**q

    return epe_map


@LOSSES.register_module()
class MultiLevelEPE(nn.Module):
    """Multi-level end point error loss.

    Args:

        p (int): norm degree for loss. Options are 1 or 2. Defaults to 2.
        q (float): used to give less penalty to outliers when fine-tuning
            model. Defaults to None.
        eps (float): a small constant to numerical stability when fine-tuning
            model. Defaults to None.
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
                 p: int = 2,
                 q: Optional[float] = None,
                 eps: Optional[float] = None,
                 weights: Dict[str, float] = dict(
                     level6=0.32,
                     level5=0.08,
                     level4=0.02,
                     level3=0.01,
                     level2=0.005),
                 flow_div: float = 20.,
                 max_flow: float = float('inf'),
                 resize_flow: str = 'downsample',
                 scale_as_level: bool = False,
                 reduction: str = 'sum') -> None:

        super().__init__()

        assert p == 1 or p == 2
        self.p = p

        self.q = q
        if self.q is not None:
            assert self.q > 0

        self.eps = eps
        if self.eps is not None:
            assert eps > 0

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
                preds_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                target: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forwar function for MultiLevelEPE.

        Args:
            preds_dict (dict): Multi-level output of predicted optical flow,
                and the contain in dict is a Tensor or list of Tensor with
                shape (B, 1, H_l, W_l), where l indicates the level.
            target (Tensor): Ground truth of optical flow with shape
                (B, 2, H, W).
            valid (Tensor, optional): Valid mask for optical flow.
                Defaults to None.

        Returns:
            Tensor: value of pixel-wise end point error loss.
        """

        return multi_level_flow_loss(
            endpoint_error,
            preds_dict,
            target,
            weights=self.weights,
            valid=valid,
            flow_div=self.flow_div,
            max_flow=self.max_flow,
            resize_flow=self.resize_flow,
            scale_as_level=self.scale_as_level,
            reduction=self.reduction,
            p=self.p,
            q=self.q,
            eps=self.eps,
        )

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__
        repr_str += (f'(resize_flow={self.resize_flow}, '
                     f'scale_as_level={self.scale_as_level}, '
                     f'flow_div={self.flow_div}, '
                     f'weights={self.weights}, '
                     f'p={self.p}, '
                     f'q={self.q}, '
                     f'eps={self.eps}, '
                     f'reduction=\'{self.reduction}\')')

        return repr_str
