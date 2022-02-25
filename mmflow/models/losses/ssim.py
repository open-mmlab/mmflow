# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def weighted_ssim(x: Tensor,
                  y: Tensor,
                  weight: Optional[Tensor] = None,
                  c1=0.01**2,
                  c2=0.03**2,
                  weight_epsilon=0.01):
    """Computes a weighted structured image similarity.

    This function is modified from
    https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py
    Copyright 2022 The Google Research Authors.

    Args:
        x (Tensor): A Tensor representing a batch of images, of shape
            [B, C, H, W].
        y (Tensor): A Tensor representing a batch of images, of shape
            [B, C, H, W].
        weight (Tensor, optional): A Tensor of shape [H, W], representing
            the weight of each pixel in both images when we come to calculate
            moments (means and correlations). Defaults to None.
        c1 (float): A floating point number, regularizes division by zero of
            the means. Defaults to 0.01**2.
        c2 (float): A floating point number, regularizes division by zero of
            the means. Defaults to 0.03 ** 2.
        weight_epsilon (float): A floating point number, used to regularize
            division by the weight. Defaults to 0.01.
    Returns:
        A tuple of two Tensors. First, of shape [B, C, H-2, W-2], is scalar
        similarity loss oer pixel per channel. It is needed so that we know
        how much to weigh each pixel in the first tensor. For example, if
        ``weight`` was very small in some area of the images, the first tensor
        will still assign a loss to these pixels, but we shouldn't take the
        result too seriously.
    """
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError(
            'Both c1 and c2 are infinite, SSIM loss is zero. This is '
            'likely unintended.')
    _, _, H, W = x.shape

    if weight is None:
        weight = torch.ones((H, W)).to(x)
    else:
        assert weight.shape == (H, W), \
                f'image shape is {(H, W)}, but weight shape is {weight.shape}'
    weight = weight[None, None, ...]
    average_pooled_weight = F.avg_pool2d(weight, (3, 3), stride=(1, 1))
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (
        average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        weighted_avg = F.avg_pool2d(
            z * weight_plus_epsilon, (3, 3), stride=(1, 1))
        return weighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1)
