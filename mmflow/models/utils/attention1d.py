# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmcv.runner import BaseModule
from torch import Tensor, nn


class AttentionLayer(BaseModule):
    """AttentionLayer on x or y direction, compute the self attention on y or x
    direction.

    Args:
        in_channels (int): Number of input channels.
        y_attention (bool): Whether calculate y axis's attention or not.
    """

    def __init__(self, in_channels: int, y_attention: bool = False) -> None:
        super().__init__()
        self.y_attention = y_attention

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature1: Tensor, feature2: Tensor, position: Tensor,
                value: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function for AttentionLayer.

        Query: feature1 + position
        Key: feature2 + position
        Value: feature2

        Args:
            feature1 (Tensor): The input feature1.
            feature2 (Tensor): The input feature2.
            position (Tensor): position encoding.
            value (Tensor): attention value.

        Returns:
            Tuple[Tensor, Tensor]: The output of attention layer
            and attention weights (scores).
        """
        b, c, h, w = feature1.size()

        query = feature1 + position if position is not None else feature1
        query = self.query_conv(query)

        key = feature2 + position if position is not None else feature2
        key = self.key_conv(key)

        value = feature2 if value is None else value
        scale_factor = c**0.5

        if self.y_attention:
            # multiple on H direction, feature shape is [B, W, H, C]
            query = query.permute(0, 3, 2, 1)
            key = key.permute(0, 3, 1, 2)
            value = value.permute(0, 3, 2, 1)
        else:  # x attention
            # multiple on W direction, feature shape is [B, W, H, C]
            query = query.permute(0, 2, 3, 1)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 3, 1)

        # the shape of attention is [B, W, H, H] or  [B, H, W, W]
        scores = torch.matmul(query, key) / scale_factor
        scores = torch.softmax(scores, dim=-1)

        out = torch.matmul(scores, value)

        # the shape of output is [B, C, H, W]
        if self.y_attention:
            out = out.permute(0, 3, 2, 1).contiguous()
        else:
            out = out.permute(0, 3, 1, 2).contiguous()

        return out, scores


class Attention1D(BaseModule):
    """Cross-Attention on x or y direction, without multi-head and dropout
    support for faster speed First compute y or x direction self attention,
    then compute x or y direction cross attention.

    Args:
        in_channels (int): Number of input channels.
        y_attention (bool): Whether calculate y axis's attention or not
        double_cross_attn (bool): Whether calculate self attention or not
    """

    def __init__(self,
                 in_channels: int,
                 y_attention: bool = False,
                 double_cross_attn: bool = True) -> None:
        super().__init__()
        self.y_attention = y_attention
        self.double_cross_attn = double_cross_attn
        if double_cross_attn:
            self.self_attn = AttentionLayer(in_channels, not y_attention)
        self.cross_attn = AttentionLayer(in_channels, y_attention)

    def forward(self, feature1: Tensor, feature2: Tensor, position: Tensor,
                value: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function for Attention1D.

        Args:
            feature1 (Tensor): The input feature1.
            feature2 (Tensor): The input feature2.
            position (Tensor): position encoding.
            value (Tensor): attention value.

        Returns:
            Tuple[Tensor, Tensor]: The output of attention layer
            and attention weights (scores).
        """
        if self.double_cross_attn:
            feature1 = self.self_attn(feature1, feature1, position, value)[0]
        return self.cross_attn(feature1, feature2, position, value)
