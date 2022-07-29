# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcv.runner import BaseModule
from torch import nn


class AttentionLayer(BaseModule):
    """AttentionLayer on x or y direction, compute the self attention on y or x
    direction.

    Args:
        in_channels (int): Number of input channels.
        y_attention (bool): Whether calculate y axis's attention or not.
    """

    def __init__(self, in_channels: int, y_attention: bool = False):
        super().__init__()
        self.y_attention = y_attention

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature1, feature2, position, value):
        """Forward function for AttentionLayer.

        Key: feature2 + position
        Value: feature2
        Query: feature1 + position

        Args:
            feature1 (Tensor): The input feature1.
            feature2 (Tensor): The input feature2.
            position (Tensor): position encoding.
            value (Tensor): attention value.
        Returns:

            out (Tensor): attention layer output
            attention (Tensor): attention of key and query
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
        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, value)

        # the shape of output is [B, C, H, W]
        if self.y_attention:
            out = out.permute(0, 3, 2, 1).contiguous()  # [B, C, H, W]
        else:
            out = out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return out, attention


class Attention1D(BaseModule):
    """Cross-Attention on x or y direction, without multi-head and dropout
    support for faster speed First compute y or x direction self attention,
    then compute x or y direction cross attention.

    Args:
        in_channels (int): Number of input channels.
        y_attention (bool): Whether y axis's attention or not
    """

    def __init__(self, in_channels: int, y_attention: bool = False):
        super(Attention1D, self).__init__()
        self.y_attention = y_attention

        self.self_attn = AttentionLayer(in_channels, not y_attention)
        self.cross_attn = AttentionLayer(in_channels, y_attention)

    def forward(self, feature1, feature2, position, value):
        """Forward function for Attention1D.

        Args:
            feature1 (Tensor): The input feature1.
            feature2 (Tensor): The input feature2.
            position (Tensor): position encoding.
            value (Tensor): attention value.

        Returns:
            out (Tensor): cross attention output
            attention (Tensor):  cross attention of key and query
        """
        feature1 = self.self_attn(feature1, feature1, position, value)[0]
        return self.cross_attn(feature1, feature2, position, value)
