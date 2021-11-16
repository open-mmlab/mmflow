# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor,
                         balance: bool, reduction: str) -> torch.Tensor:
    r"""Calculate (weighted) binary cross entropy between occlusion prediction
    and occlusion ground truth.

    The loss function is
    .. math::
      loss = w \cdot o_pred\ln{o_gt}+\bar{w}(1-o_pred)\ln{(1-o_gt)}

    where the weights is:
    .. math::
      w=\frac{H \cdot W}{\sum{o_pred}+\sum{o_gt}}

    and
    .. math::
      \bar{w}=\frac{H \cdot W}{\sum{(1-o_pred)}+\sum{(1-o_gt)}}

    but from IRR released code, the weights were scaled by 0.5.

    Args:
        pred (Tensor): output predicted occ map from flow_estimator
            with shape(B, 1, H, W).
        target (Tensor): ground truth occ map with shape (B, 1, H, W).
        balance (bool): whether balance the class weights for irr models.
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.

    Return:
        Tensor: value of pixel-wise binary cross entropy loss.
    """

    assert pred.shape == target.shape, \
        (f'pred shape {pred.shape} does not match target shape '
         f'{target.shape}.')

    b = pred.shape[0]

    h, w = pred.shape[2:]

    # normalize pred using sigmoid
    pred = torch.sigmoid(pred)

    if balance:
        tp_weight = 0.5 * h * w / (
            torch.sum(target, dim=[1, 2, 3]) + torch.sum(pred, dim=[1, 2, 3]) +
            1e-8)

        fn_weight = 0.5 * h * w / (
            torch.sum(1 - target, dim=[1, 2, 3]) +
            torch.sum(1 - pred, dim=[1, 2, 3]) + 1e-8)

    else:
        tp_weight, fn_weight = torch.ones(b).to(target), torch.ones(b).to(
            target)

    tp = -target * torch.log(pred + 1e-8) * tp_weight.view(b, 1, 1, 1)

    fn = -(1 - target) * torch.log(1 - pred + 1e-8) * fn_weight.view(
        b, 1, 1, 1)

    bce_map = tp + fn

    if reduction == 'none':
        return torch.squeeze(bce_map)  # shape (B, H, W).

    elif reduction == 'mean':
        return torch.mean(bce_map)

    elif reduction == 'sum':
        return torch.sum(bce_map) / b


def multi_levels_binary_cross_entropy(
    preds_dict: Dict[str, Union[Sequence[torch.Tensor], torch.Tensor]],
    target: torch.Tensor,
    weights: Dict[str, float] = dict(
        level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005),
    balance: bool = False,
    reduction: str = 'sum',
) -> torch.Tensor:
    """Multi-level binary cross entropy function.

    Args:
        preds_dict (dict): multi-level output of predicted occlusion, and the
            contain in dict is a Tensor or list of Tensor with shape
            (B, 1, H_l, W_l), where l indicates the level.
        target (Tensor): ground truth of occlusion with shape (B, 1, H, W).
        weights (dict): manual rescaling weights given to the loss of occlusion
            at each level, and the keys in weights must correspond to predicted
            dict. Defaults to dict(
            level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005).
        balance (bool): whether balance true positive and false negative by
            predicted and ground truth labels. Defaults to False.
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.

    Returns:
        Tensor: value of pixel-wise binary cross entropy loss.
    """

    assert preds_dict.keys() == weights.keys(), \
        'Error: Keys of prediction do not match keys of weights!'

    loss = 0

    for k in weights.keys():
        # predict more than one flow map at one level
        cur_pred = preds_dict[k] if \
            isinstance(preds_dict[k], (tuple, list)) else [preds_dict[k]]

        num_preds = len(cur_pred)

        h, w = cur_pred[0].shape[2:]

        cur_weight = weights.get(k)

        cur_target = F.adaptive_avg_pool2d(target, [h, w])
        for pred in cur_pred:
            loss += binary_cross_entropy(pred, cur_target, balance,
                                         reduction) * cur_weight

    return loss / num_preds


@LOSSES.register_module()
class MultiLevelBCE(nn.Module):
    """Multi-level binary cross entropy.

    Args:
        weights (dict): manual rescaling weights given to the loss of occlusion
            at each level, and the keys in weights must correspond to predicted
            dict. Defaults to dict(
            level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005).
        balance (bool): whether balance true positive and false negative by
            predicted and ground truth labels. Defaults to False.
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.
    """

    def __init__(self,
                 weights: Dict[str, float] = dict(
                     level6=0.32,
                     level5=0.08,
                     level4=0.02,
                     level3=0.01,
                     level2=0.005),
                 balance: bool = False,
                 reduction: str = 'sum') -> None:

        super().__init__()

        assert isinstance(balance, bool)
        self.balance = balance

        assert isinstance(weights, dict)
        self.weights = weights

        assert reduction in ('mean', 'sum')
        self.reduction = reduction

    def forward(self, preds_dict: dict, target: torch.Tensor) -> torch.Tensor:
        """Forwar function for MultiLevelBCE.

        Args:
            preds_dict (dict): multi-level output of predicted occlusion, and
                the contain in dict is a Tensor or list of Tensor with shape
                (B, 1, H_l, W_l), where l indicates the level.
            target (Tensor): ground truth of occlusion with shape (B, 1, H, W).

        Returns:
            Tensor: value of pixel-wise binary cross entropy loss.
        """

        return multi_levels_binary_cross_entropy(
            preds_dict,
            target,
            weights=self.weights,
            balance=self.balance,
            reduction=self.reduction)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(balance={self.balance}, '
                     f'weights={self.weights}, '
                     f'reduction=\'{self.reduction}\')')
        return repr_str
