# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
# import torch.nn.functional as F
from torch import Tensor


def weighted_ssim(x: Tensor,
                  y: Tensor,
                  weight: Optional[Tensor] = None,
                  c1=float('inf'),
                  c2=9e-6,
                  weight_epsilon=0.01):
    """Computes a weighted structured image similarity.

    Args:
        x (Tensor): A Tensor representing a batch of images, of shape
            [B, C, H, W].
        y (Tensor): A Tensor representing a batch of images, of shape
            [B, C, H, W].
        weight (Tensor, optional): A Tensor of shape [H, W], representing
            the weight of each pixel in both images when we come to calculate
            moments (means and correlations). Defaults to None.
        c1 (float): A floating point number, regularizes division by zero of
            the means. Defaults to float('inf').
        c2 (float): A floating point number, regularizes division by zero of
            the means. Defaults to 9e-6.
        weight_epsilon (float): A floating point number, used to regularize
            division by the weight. Defaults to 0.01.
    """
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError(
            'Both c1 and c2 are infinite, SSIM loss is zero. This is '
            'likely unintended.')
    B, C, H, W = x.shape

    if weight is None:
        weight = torch.ones((H, W)).to(x)
    else:
        assert weight.shape == (H, W), \
                f'image shape is {(H, W)}, but weight shape is {weight.shape}'
    weight = weight[None, None, ...]
    # average_pooled_weight = F.avg_pool2d(weight, kernel=(3, 3))
