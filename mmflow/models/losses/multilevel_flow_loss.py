# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F


def multi_level_flow_loss(loss_function,
                          preds_dict: Dict[str, Union[torch.Tensor,
                                                      List[torch.Tensor]]],
                          target: torch.Tensor,
                          weights: Dict[str, float] = dict(
                              level6=0.32,
                              level5=0.08,
                              level4=0.02,
                              level3=0.01,
                              level2=0.005),
                          valid: Optional[torch.Tensor] = None,
                          flow_div: float = 20.,
                          max_flow: float = float('inf'),
                          resize_flow: str = 'downsample',
                          reduction: str = 'sum',
                          scale_as_level: bool = False,
                          **kwargs) -> torch.Tensor:
    """Multi-level endpoint error loss function.

    Args:
        loss_function: pixel-wise loss function for optical flow map.
        preds_dict (dict): multi-level output of predicted optical flow, and
            the contain in dict is a Tensor or list of Tensor with shape
            (B, 1, H_l, W_l), where l indicates the level.
        target (Tensor): ground truth of optical flow with shape (B, 2, H, W).
        weights (dict): manual rescaling weights given to the loss of flow map
            at each level, and the keys in weights must correspond to predicted
            dict. Defaults to dict(
            level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005).
        valid (Tensor, optional): valid mask for optical flow.
            Defaults to None.
        flow_div (float): the divisor used to scale down ground truth.
            Defaults to 20.
        max_flow (float): maximum value of optical flow, if some pixel's flow
            of target is larger than it, this pixel is not valid. Default to
            float('inf').
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.
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
        kwargs: arguments for loss_function.

    Returns:
        Tensor: end-point error loss.
    """

    assert isinstance(weights, dict)

    assert list(preds_dict.keys()).sort() == list(weights.keys()).sort(), \
        'Error: Keys of prediction do not match keys of weights!'

    mag = torch.sum(target**2, dim=1).sqrt()

    if valid is None:
        valid = torch.ones_like(target[:, 0, :, :])
    else:
        valid = ((valid >= 0.5) & (mag < max_flow)).to(target)

    target_div = target / flow_div

    c_org, h_org, w_org = target.shape[1:]
    assert c_org == 2, f'The channels ground truth must be 2, but got {c_org}'

    loss = 0

    for level in weights.keys():

        # predict more than one flow map at one level
        cur_pred = preds_dict[level] if isinstance(
            preds_dict[level], (tuple, list)) else [preds_dict[level]]

        num_preds = len(cur_pred)

        b, _, h, w = cur_pred[0].shape

        scale_factor = torch.Tensor([
            float(w / w_org), float(h / h_org)
        ]).to(target) if scale_as_level else torch.Tensor([1., 1.]).to(target)

        cur_weight = weights.get(level)

        if resize_flow == 'downsample':
            # down sample ground truth following irr solution
            # https://github.com/visinf/irr/blob/master/losses.py#L16
            cur_target = F.adaptive_avg_pool2d(target_div, [h, w])
            cur_valid = F.adaptive_max_pool2d(valid, [h, w])
        else:
            cur_target = target_div
            cur_valid = valid

        loss_map = torch.zeros_like(cur_target[:, 0, ...])

        for i_pred in cur_pred:

            if resize_flow == 'upsample':
                # up sample predicted flow following pwcnet and irr solution
                # https://github.com/visinf/irr/blob/master/losses.py#L20
                # when training sparse flow dataset, as no generic
                # interpolation mode can resize a ground truth of sparse flow
                # correctly.
                i_pred = F.interpolate(
                    i_pred,
                    size=cur_target.shape[2:],
                    mode='bilinear',
                    align_corners=False)

            cur_target = torch.einsum('b c h w, c -> b c h w', cur_target,
                                      scale_factor)

            loss_map += loss_function(i_pred, cur_target, **kwargs) * cur_valid

            if reduction == 'mean':
                loss += loss_map.sum() / (cur_valid.sum() + 1e-8) * cur_weight
            elif reduction == 'sum':
                loss += loss_map.sum() / b * cur_weight

    return loss / num_preds
