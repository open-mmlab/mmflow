# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import DECODERS
from .raft_decoder import ConvGRU, RAFTDecoder


class RelPosEmb(BaseModule):
    """Relative position embedding.

    Separate embedding vectors are learned for the vertical and horizontal
    offset and are added on query tensor.

    Args:
        max_pos_size (int): maximum position length.
        head_channels (int): The channels of head feature.
    """

    def __init__(self, max_pos_size: int, head_channels: int):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, head_channels)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, head_channels)
        deltas = torch.arange(max_pos_size).view(
            1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q: torch.Tensor):
        """forward function for relative position embedding.

        Args:
            q (torch.Tensor): query tensor in attention.

        Returns:
            torch.Tensor: Position score with shape (B, heads, H * W, H * W).
        """
        B, heads, H, W, head_channels = q.shape

        height_emb = self.rel_height(self.rel_ind[:H, :H].reshape(-1)).reshape(
            H, H, -1, head_channels)
        width_emb = self.rel_width(self.rel_ind[:W, :W].reshape(-1)).reshape(
            W, -1, W, head_channels)

        height_score = torch.einsum('b h x y d, x u v d -> b h x y u v', q,
                                    height_emb)
        width_score = torch.einsum('b h x y d, y u v d -> b h x y u v', q,
                                   width_emb)

        return (height_score + width_score).reshape(B, heads, H * W, H * W)


class Attention(BaseModule):
    """Compute 4D attention matrix encodes self-similarity in appearance
    feature space by using context features.

    Args:
        in_channels (int): The channels of input context features
        heads (int): The number of parallel attention heads.
        head_channels (int): The channels of head feature.
        position_only (bool): Whether use position-only attention. Default to
            False.
        max_pos_size (int, optional): The max size of positional embedding
            vectors. If max_pos_size is None, attention is content-only
            self-similarity attention model. If position_only=True,
            max_pos_size must be defined. Default to None.

        Note:
            Attention module has 3 mode based on different `position_only` and
            `max_pos_size`:
            1. If `position_only` is True and `max_pos_size` is defined,
            it denotes the position-only attention module.
            2. If `position_only` is False and `max_pos_size` is defined,
            it denotes the joint position and content-wise attention module.
            3. If `postion_only` is False and `max_pos_size` is not defined,
            it denotes the content-only self-similarity attention module.
    """

    def __init__(self,
                 in_channels: int,
                 heads: int,
                 head_channels: int,
                 position_only: bool = False,
                 max_pos_size: Optional[int] = None) -> None:
        super().__init__()
        if position_only and max_pos_size is None:
            raise RuntimeError(
                'Must define `max_pos_size`, if position_only=True')

        self.in_channels = in_channels
        self.heads = heads
        self.head_channels = head_channels
        self.position_only = position_only
        self.max_pos_size = max_pos_size

        self.scale = head_channels**-0.5
        self.to_qk = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.heads * self.head_channels * 2,
            kernel_size=1,
            bias=False)

        if self.max_pos_size is not None:
            self.pos_emb = RelPosEmb(
                max_pos_size=self.max_pos_size, head_channels=head_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for computing self-similarity with context
        features.

        Args:
            x (torch.Tensor): The context features.

        Returns:
            torch.Tensor: self-similarity for context features.
        """
        B, _, H, W = x.shape
        q, k = torch.split(
            self.to_qk(x),
            [self.heads * self.head_channels, self.heads * self.head_channels],
            dim=1)

        # q shape is (B, heads, HxW, head_channels)
        q = q.view(B, self.heads, self.head_channels, H,
                   W).permute(0, 1, 3, 4, 2)
        # k shape is (B, heads, head_channels, HxW)
        k = k.view(B, self.heads, self.head_channels, -1)

        self_similarity = torch.zeros((B, self.heads, H * W, H * W)).to(q)

        if self.max_pos_size is not None:
            self_similarity += self.pos_emb(q)

        if not self.position_only:
            # self_similarity shape is (B, heads, HxW, HxW)
            self_similarity += torch.matmul(
                q.view(B, self.heads, -1, self.head_channels),
                k.view(B, self.heads, self.head_channels, -1)) * self.scale

        attn = self_similarity.softmax(dim=-1)

        return attn


class Aggregate(BaseModule):
    """Computing aggregated global motion features.

    Args:
        in_channels (int): The channels of motion features.
        heads (int): The number of parallel heads.
        head_channels (int): The channels of head feature.
    """

    def __init__(self, in_channels: int, heads: int,
                 head_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.heads = heads
        self.head_channels = head_channels

        self.scale = head_channels**-0.5
        self.to_v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.heads * self.head_channels,
            kernel_size=1,
            bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        if in_channels != heads * head_channels:
            self.project = nn.Conv2d(
                in_channels=heads * head_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False)
        else:
            self.project = nn.Sequential()

    def forward(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward function to compute aggregated global motion features.

        Args:
            attn (torch.Tensor): The attention features
            x (torch.Tensor): The motion features

        Returns:
            torch.Tensor: The aggregated global motion features
        """
        B, _, H, W = x.shape

        # v shape : B, Heads, Head_channels, HxW
        v = self.to_v(x).view(B, self.heads, self.head_channels, -1)

        # attn shape: B, Heads, HxW, HxW
        # out shape: B, Heads, HxW, Head_channels
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))

        # out shape: B, in_channels, H, W
        out = self.project(out.permute(0, 1, 3, 2).reshape(B, -1, H, W))

        out = x + self.gamma * out

        return out


@DECODERS.register_module()
class GMADecoder(RAFTDecoder):
    """The decoder of GMA.

    Args:
        heads (int): The number of parallel attention heads.
        motion_channels (int): The channels of motion channels.
        position_only (bool): Whether use position-only attention. Default to
            False.
        max_pos_size (int, optional): The max size of positional embedding
            vectors. If max_pos_size is None, attention is content-only
            self-similarity attention model. If position_only=True,
            max_pos_size must be defined. Default to None.

        Note:
            Attention module has 3 mode based on different `position_only` and
            `max_pos_size`:
            1. If `position_only` is True and `max_pos_size` is defined,
            it denotes the position-only attention module.
            2. If `position_only` is False and `max_pos_size` is defined,
            it denotes the joint position and content-wise attention module.
            3. If `postion_only` is False and `max_pos_size` is not defined,
            it denotes the content-only self-similarity attention module.
    """

    def __init__(self,
                 *args,
                 heads: int,
                 motion_channels: int,
                 position_only: bool = False,
                 max_pos_size: Optional[int] = None,
                 **kwargs) -> None:
        self.heads = heads
        self.motion_channels = motion_channels
        self.position_only = position_only
        self.max_pos_size = max_pos_size

        super().__init__(*args, **kwargs)
        self.attn = Attention(
            in_channels=self.cxt_channels,
            heads=heads,
            head_channels=self.cxt_channels,
            position_only=self.position_only,
            max_pos_size=self.max_pos_size)

        self.aggregator = Aggregate(
            in_channels=motion_channels,
            heads=heads,
            head_channels=motion_channels)

    def make_gru_block(self) -> torch.nn.Module:
        return ConvGRU(
            self.h_channels,
            self.cxt_channels + self.motion_channels * 2,
            net_type=self.gru_type)

    def forward(self, feat1, feat2, flow, h, cxt_feat):
        """Forward function for RAFTDecoder.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The initialized flow when warm start.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.

        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """

        corr_pyramid = self.corr_block(feat1, feat2)
        upflow_preds = []
        delta_flow = torch.zeros_like(flow)

        attention = self.attn(cxt_feat)

        for _ in range(self.iters):
            flow = flow.detach()
            corr = self.corr_lookup(corr_pyramid, flow)
            motion_feat = self.encoder(corr, flow)
            motion_features_global = self.aggregator(attention, motion_feat)
            x = torch.cat([cxt_feat, motion_feat, motion_features_global],
                          dim=1)
            h = self.gru(h, x)

            delta_flow = self.flow_pred(h)
            flow = flow + delta_flow

            if hasattr(self, 'mask_pred'):
                mask = .25 * self.mask_pred(h)
            else:
                mask = None

            upflow = self._upsample(flow, mask)
            upflow_preds.append(upflow)

        return upflow_preds
