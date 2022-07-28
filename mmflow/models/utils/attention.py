# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcv.runner import BaseModule
from torch import nn


class AttentionLayer(BaseModule):
    """AttentionLayer on x or y direction, compute the self attention on y or x
    direction.

    Args:
        in_channels (int): Number of input channels.
        y_attention (bool): Whether y axis's attention or not
    """

    def __init__(
        self,
        in_channels: int,
        y_attention: bool = False,
    ):
        super(AttentionLayer, self).__init__()
        self.y_attention = y_attention

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature1, position, value):
        """Forward function for AttentionLayer.
        Key: feature1 + position
        Value: feature1
        Query: feature1 + position

                Args:
                    feature1 (Tensor): The input feature1.
                    position (Tensor): position encoding.
                    value (Tensor): attention value.
                Returns:

                """
        b, c, h, w = feature1.size()

        query = feature1 + position if position is not None else feature1
        query = self.query_conv(query)

        key = feature1 + position if position is not None else feature1
        key = self.key_conv(key)

        value = feature1 if value is None else value
        scale_factor = c**0.5

        if self.y_attention:
            # multiple on H direction
            query = query.permute(0, 3, 2, 1)  # [B, W, H, C]
            key = key.permute(0, 3, 1, 2)  # [B, W, C, H]
            value = value.permute(0, 3, 2, 1)  # [B, W, H, C]
        else:  # x attention
            # multiple on W direction
            query = query.permute(0, 2, 3, 1)  # [B, H, W, C]
            key = key.permute(0, 2, 1, 3)  # [B, H, C, W]
            value = value.permute(0, 2, 3, 1)  # [B, H, W, C]

        scores = torch.matmul(
            query, key) / scale_factor  # [B, W, H, H] or  [B, H, W, W]

        attention = torch.softmax(
            scores, dim=-1)  # [B, W, H, H] or  [B, H, W, W]

        out = torch.matmul(attention, value)  # [B, W, H, H] or  [B, H, W, W]

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
        double_cross_attention (bool): Whether double cross attention or not
    """

    def __init__(
        self,
        in_channels: int,
        y_attention: bool = False,
    ):
        super(Attention1D, self).__init__()
        self.y_attention = y_attention

        self.self_attn = AttentionLayer(in_channels, not y_attention)

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature1, feature2, position, value):
        """Forward function for Attention1D.
        Key: feature2 + position
        Value: feature2
        Query: self attention feature1 + position

                Args:
                    feature1 (Tensor): The input feature1.
                    feature2 (Tensor): The input feature2.
                    position (Tensor): position encoding.
                    value (Tensor): attention value.
                Returns:

                """
        b, c, h, w = feature1.size()

        feature1 = self.self_attn(feature1, position, value)[0]

        query = feature1 + position if position is not None else feature1
        query = self.query_conv(query)

        key = feature2 + position if position is not None else feature2
        key = self.key_conv(key)

        value = feature2 if value is None else value
        scale_factor = c**0.5

        if self.y_attention:
            # multiple on H direction
            query = query.permute(0, 3, 2, 1)  # [B, W, H, C]
            key = key.permute(0, 3, 1, 2)  # [B, W, C, H]
            value = value.permute(0, 3, 2, 1)  # [B, W, H, C]
        else:  # x attention
            # multiple on W direction
            query = query.permute(0, 2, 3, 1)  # [B, H, W, C]
            key = key.permute(0, 2, 1, 3)  # [B, H, C, W]
            value = value.permute(0, 2, 3, 1)  # [B, H, W, C]

        scores = torch.matmul(
            query, key) / scale_factor  # [B, W, H, H] or  [B, H, W, W]

        attention = torch.softmax(
            scores, dim=-1)  # [B, W, H, H] or  [B, H, W, W]

        out = torch.matmul(attention, value)  # [B, W, H, H] or  [B, H, W, W]

        if self.y_attention:
            out = out.permute(0, 3, 2, 1).contiguous()  # [B, C, H, W]
        else:
            out = out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return out, attention
