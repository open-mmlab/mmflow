# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor


def census_loss(image1: Tensor, image2: Tensor, patch_size: int = 3) -> Tensor:
    """The census loss.

    Modified from
    https://github.com/lliuz/ARFlow/blob/master/losses/flow_loss.py
    licensed under MIT License,

    Args:
        image1 (Tensor): Tensor of shape (B, 3, H, W)
        image2 (Tensor): Tensor of shape (B, 3, H, W)
        patch_size (int): The window size for census transform. Defaults to 3.

    Returns:
        Tensor: The scaler of census loss between image1 and image2.
    """

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale[:, None, ...]

    def _census_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        weight = torch.eye(out_channels).view(
            (out_channels, 1, patch_size, patch_size)).to(image)
        patches = F.conv2d(intensities, weight, padding=patch_size // 2)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = (t1 - t2)**2
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).to(t)
        # just mask the edge
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _census_transform(image1)
    t2 = _census_transform(image2)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(image1, patch_size // 2)

    return (dist * mask).sum() / mask.sum()
