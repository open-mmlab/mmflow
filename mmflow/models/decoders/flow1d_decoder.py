# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmflow.registry import MODELS
from ..utils.attention1d import Attention1D
from ..utils.correlation1d import Correlation1D
from .raft_decoder import MotionEncoder, RAFTDecoder, XHead


class MotionEncoderFlow1D(MotionEncoder):
    """The module of motion encoder for Flow1D.

    An encoder which consists of several convolution layers and outputs
    features as GRU's input.

    Args:
        num_levels (int): Number of levels used when calculating correlation
            tensor. Default: 32.
        radius (int): Radius used when calculating correlation tensor.
            Default: 4.
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
            Default: 'Basic'.
    """

    def __init__(self,
                 radius: int = 32,
                 net_type: str = 'Basic',
                 **kwargs) -> None:
        super().__init__(radius=radius, net_type=net_type, **kwargs)
        corr_channels = self._corr_channels.get(net_type) if isinstance(
            self._corr_channels[net_type],
            (tuple, list)) else [self._corr_channels[net_type]]
        corr_kernel = self._corr_kernel.get(net_type) if isinstance(
            self._corr_kernel.get(net_type),
            (tuple, list)) else [self._corr_kernel.get(net_type)]
        corr_padding = self._corr_padding.get(net_type) if isinstance(
            self._corr_padding.get(net_type),
            (tuple, list)) else [self._corr_padding.get(net_type)]

        corr_inch = 2 * (2 * radius + 1)
        corr_net = self._make_encoder(corr_inch, corr_channels, corr_kernel,
                                      corr_padding, **kwargs)
        self.corr_net = nn.Sequential(*corr_net)


class PositionEmbeddingSine(nn.Module):
    """refer to the standard version of position embedding used by the
    Attention is all you need paper, generalized to work on images.

    https://github.com/facebookresearch/detr/blob/main/models/position_encod
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=True,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


@MODELS.register_module()
class Flow1DDecoder(RAFTDecoder):
    """The decoder of Flow1D Net.

    The decoder of Flow1D Net, which outputs list of upsampled flow estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """
    _h_channels = {'Basic': 128, 'Small': 96}
    _cxt_channels = {'Basic': 128, 'Small': 64}

    def __init__(self,
                 net_type: str,
                 radius: int,
                 corr_op_cfg: dict = dict(
                     type='CorrLookupFlow1D', align_corners=True),
                 feat_channels: Union[int, Sequence[int]] = 256,
                 mask_channels: int = 64,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(
            net_type=net_type,
            num_levels=4,
            radius=radius,
            corr_op_cfg=corr_op_cfg,
            feat_channels=feat_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.attn_block_x = Attention1D(
            in_channels=feat_channels,
            y_attention=False,
            double_cross_attn=True)
        self.attn_block_y = Attention1D(
            in_channels=feat_channels,
            y_attention=True,
            double_cross_attn=True)
        self.corr_block = Correlation1D()
        self.feat_channels = feat_channels if isinstance(
            tuple, list) else [feat_channels]

        self.encoder = MotionEncoderFlow1D(
            radius=radius,
            net_type=net_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]
        self.mask_channels = mask_channels * 9
        if net_type == 'Basic':
            self.mask_pred = XHead(
                self.h_channels, feat_channels, self.mask_channels, x='mask')

    def _upsample(self,
                  flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex
        combination.

        Args:
            flow (Tensor): The optical flow with the shape [N, 2, H/8, W/8].
            mask (Tensor, optional): The learnable mask with shape
                [N, grid_size x scale x scale, H/8, H/8].

        Returns:
            Tensor: The output optical flow with the shape [N, 2, H, W].
        """
        scale = 8
        grid_size = 9
        grid_side = int(math.sqrt(grid_size))
        N, _, H, W = flow.shape
        if mask is None:
            new_size = (scale * H, scale * W)
            return scale * F.interpolate(
                flow, size=new_size, mode='bilinear', align_corners=True)
        # predict a (Nx8×8×9xHxW) mask
        mask = mask.view(N, 1, grid_size, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        # extract local grid with 3x3 side  padding = grid_side//2
        upflow = F.unfold(scale * flow, [grid_side, grid_side], padding=1)
        # upflow with shape N, 2, 9, 1, 1, H, W
        upflow = upflow.view(N, 2, grid_size, 1, 1, H, W)

        # take a weighted combination over the neighborhood grid 3x3
        # upflow with shape N, 2, 8, 8, H, W
        upflow = torch.sum(mask * upflow, dim=2)
        upflow = upflow.permute(0, 1, 4, 2, 5, 3)
        return upflow.reshape(N, 2, scale * H, scale * W)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                flow: torch.Tensor, h_feat: torch.Tensor,
                cxt_feat: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward function for Flow1D.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The initialized flow when warm start.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.

        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """
        pos_encoding = PositionEmbeddingSine(self.feat_channels[0] // 2)
        position = pos_encoding(feat1)

        # attention
        feat2_x, _ = self.attn_block_x(feat1, feat2, position, None)
        feat2_y, _ = self.attn_block_y(feat1, feat2, position, None)

        correlation_x = self.corr_block(feat1, feat2_y, x_correlation=True)
        correlation_y = self.corr_block(feat1, feat2_x, x_correlation=False)

        corrleation1d = [correlation_x, correlation_y]
        upflow_preds = []
        delta_flow = torch.zeros_like(flow)
        for _ in range(self.iters):
            flow = flow.detach()
            corr = self.corr_lookup(corrleation1d, flow)
            motion_feat = self.encoder(corr, flow)
            x = torch.cat([cxt_feat, motion_feat], dim=1)
            h_feat = self.gru(h_feat, x)

            delta_flow = self.flow_pred(h_feat)
            flow = flow + delta_flow

            if hasattr(self, 'mask_pred'):
                mask = .25 * self.mask_pred(h_feat)
            else:
                mask = None

            upflow = self._upsample(flow, mask)
            upflow_preds.append(upflow)

        return upflow_preds
