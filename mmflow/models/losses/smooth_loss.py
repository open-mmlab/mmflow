# Copyright (c) OpenMMLab. All rights reserved.

from typing import Tuple

import torch
from torch import Tensor


def gradient(data: Tensor, stride: int = 1) -> Tuple[Tensor]:
    """Calculate gradient in data.

    Args:
        data (Tensor): input data with shape (B, C, H, W).
        stride (int): stride for distance of calculating changing. Default to
            1.

    Returns:
        tuple(Tensor): partial derivative of data with respect to x, with shape
            (B, C, H-stride, W), and partial derivative of data with respect to
            y with shape (B, C, H, W-stride).
    """
    D_dy = data[:, :, stride:] - data[:, :, :-stride]
    D_dx = data[:, :, :, stride:] - data[:, :, :, :-stride]
    return D_dx / stride, D_dy / stride


def smooth_1st_loss(flow: Tensor,
                    image: Tensor,
                    alpha: float = 0.,
                    smooth_edge_weighting: str = 'exponential') -> Tensor:
    """The First order smoothness loss.

    Modified from
    https://github.com/lliuz/ARFlow/blob/master/losses/flow_loss.py
    licensed under MIT License,
    and https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py
    licensed under the Apache License, Version 2.0.

    Args:
        flow (Tensor): Input optical flow with shape (B, 2, H, W).
        image (Tensor): Input image with shape (B, 3, H, W).
        alpha (float): Weight for modulates edge weighting. Default to: 0.
        smooth_edge_weighting (str): Function for calculating abstract
            value of image gradient which can be a string {'exponential'
            'gaussian'}.

    Returns:
        Tensor: A scaler of the first order smoothness loss.
    """ # noqa E501
    assert smooth_edge_weighting in ('exponential', 'gaussian'), (
        'smooth edge function must be `exponential` or `gaussian`,'
        f'but got {smooth_edge_weighting}')
    # Compute image gradients and sum them up to match the receptive field
    img_dx, img_dy = gradient(image)

    abs_fn = None
    if smooth_edge_weighting == 'gaussian':
        abs_fn = torch.square
    elif smooth_edge_weighting == 'exponential':
        abs_fn = torch.abs

    weights_x = torch.exp(-torch.mean(abs_fn(img_dx * alpha), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(abs_fn(img_dy * alpha), 1, keepdim=True))

    dx, dy = gradient(flow)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() + loss_y.mean()


def smooth_2nd_loss(flow: Tensor,
                    image: Tensor,
                    alpha: float = 0.,
                    smooth_edge_weighting: str = 'exponential'):
    """The Second order smoothness loss.

    Modified from
    https://github.com/lliuz/ARFlow/blob/master/losses/flow_loss.py
    licensed under MIT License,
    and https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py
    licensed under the Apache License, Version 2.0.

    Args:
        flow (Tensor): Input optical flow with shape (B, 2, H, W).
        image (Tensor): Input image with shape (B, 3, H, W).
        alpha (float): Weight for modulates edge weighting. Default to: 0.
        smooth_edge_weighting (str): Function for calculating abstract
            value of image gradient which can be a string {'exponential'
            'gaussian'}.

    Returns:
        Tensor: A scaler of the first order smoothness loss.
    """ # noqa E501

    assert smooth_edge_weighting in ('exponential', 'gaussian'), (
        'smooth edge function must be `exponential` or `gaussian`,'
        f'but got {smooth_edge_weighting}')
    # Compute image gradients and sum them up to match the receptive field
    img_dx, img_dy = gradient(image, stride=2)

    abs_fn = None
    if smooth_edge_weighting == 'gaussian':
        abs_fn = torch.square
    elif smooth_edge_weighting == 'exponential':
        abs_fn = torch.abs

    # weights_x with shape (B, 1, H, W-2)
    # weights_y with shape (B, 1, H-2, W)
    weights_x = torch.exp(-torch.mean(abs_fn(img_dx * alpha), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(abs_fn(img_dy * alpha), 1, keepdim=True))

    dx, dy = gradient(flow)
    dx2, _ = gradient(dx)  # (B, 1, H, W-2)
    _, dy2 = gradient(dy)  # (B, 1, H-2, W)

    loss_x = weights_x * dx2.abs() / 2.
    loss_y = weights_y * dy2.abs() / 2.

    return loss_x.mean() + loss_y.mean()
