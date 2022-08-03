# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmflow.ops import build_operators
from ..builder import DECODERS, build_loss
from .base_decoder import BaseDecoder


class CorrelationPyramid(BaseModule):
    """Pyramid Correlation Module.

    The neck of RAFT-Net, which calculates correlation tensor of input features
    with the method of 4D Correlation Pyramid mentioned in RAFT-Net.

    Args:
        num_levels (int): Number of levels in the module.
            Default: 4.
    """

    def __init__(self, num_levels: int = 4) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.num_levels = num_levels

    def forward(self, feat1: torch.Tensor,
                feat2: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward function for Correlation pyramid.

        Args:
            feat1 (Tensor): The feature from first input image.
            feat2 (Tensor): The feature from second input image.

        Returns:
            Sequence[Tensor]: The list of correlation which is pooled using
                average pooling with kernel sizes {1, 2, 4, 8}.
        """
        N, C, H, W = feat1.shape
        corr = torch.matmul(
            feat1.view(N, C, -1).permute(0, 2, 1),
            feat2.view(N, C, -1)).view(N, H, W, H, W)
        corr = corr.reshape(N * H * W, 1, H, W) / torch.sqrt(
            torch.tensor(C).float())
        corr_pyramid = [corr]
        for _ in range(self.num_levels - 1):
            _corr = self.pool(corr_pyramid[-1])
            corr_pyramid.append(_corr)

        return corr_pyramid


class MotionEncoder(BaseModule):
    """The module of motion encoder.

    An encoder which consists of several convolution layers and outputs
    features as GRU's input.

    Args:
        num_levels (int): Number of levels used when calculating correlation
            tensor. Default: 4.
        radius (int): Radius used when calculating correlation tensor.
            Default: 4.
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
            Default: 'Basic'.
    """
    _corr_channels = {'Basic': (256, 192), 'Small': 96}
    _corr_kernel = {'Basic': (1, 3), 'Small': 1}
    _corr_padding = {'Basic': (0, 1), 'Small': 0}

    _flow_channels = {'Basic': (128, 64), 'Small': (64, 32)}
    _flow_kernel = {'Basic': (7, 3), 'Small': (7, 3)}
    _flow_padding = {'Basic': (3, 1), 'Small': (3, 1)}

    _out_channels = {'Basic': 126, 'Small': 80}
    _out_kernel = {'Basic': 3, 'Small': 3}
    _out_padding = {'Basic': 1, 'Small': 1}

    def __init__(self,
                 num_levels: int = 4,
                 radius: int = 4,
                 net_type: str = 'Basic',
                 **kwargs) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Small']
        corr_channels = self._corr_channels.get(net_type) if isinstance(
            self._corr_channels[net_type],
            (tuple, list)) else [self._corr_channels[net_type]]
        corr_kernel = self._corr_kernel.get(net_type) if isinstance(
            self._corr_kernel.get(net_type),
            (tuple, list)) else [self._corr_kernel.get(net_type)]
        corr_padding = self._corr_padding.get(net_type) if isinstance(
            self._corr_padding.get(net_type),
            (tuple, list)) else [self._corr_padding.get(net_type)]

        flow_channels = self._flow_channels.get(net_type)
        flow_kernel = self._flow_kernel.get(net_type)
        flow_padding = self._flow_padding.get(net_type)

        self.out_channels = self._out_channels.get(net_type) if isinstance(
            self._out_channels.get(net_type),
            (tuple, list)) else [self._out_channels.get(net_type)]
        out_kernel = self._out_kernel.get(net_type) if isinstance(
            self._out_kernel.get(net_type),
            (tuple, list)) else [self._out_kernel.get(net_type)]
        out_padding = self._out_padding.get(net_type) if isinstance(
            self._out_padding.get(net_type),
            (tuple, list)) else [self._out_padding.get(net_type)]

        corr_inch = num_levels * (2 * radius + 1)**2
        corr_net = self._make_encoder(corr_inch, corr_channels, corr_kernel,
                                      corr_padding, **kwargs)
        self.corr_net = nn.Sequential(*corr_net)

        flow_inch = 2
        flow_net = self._make_encoder(flow_inch, flow_channels, flow_kernel,
                                      flow_padding, **kwargs)
        self.flow_net = nn.Sequential(*flow_net)

        out_inch = corr_channels[-1] + flow_channels[-1]
        out_net = self._make_encoder(out_inch, self.out_channels, out_kernel,
                                     out_padding, **kwargs)
        self.out_net = nn.Sequential(*out_net)

    def _make_encoder(self, in_channel: int, channels: int, kernels: int,
                      paddings: int, conv_cfg: dict, norm_cfg: dict,
                      act_cfg: dict) -> None:
        encoder = []

        for ch, k, p in zip(channels, kernels, paddings):

            encoder.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=ch,
                    kernel_size=k,
                    padding=p,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channel = ch
        return encoder

    def forward(self, corr: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Forward function for MotionEncoder.

        Args:
            corr (Tensor): The correlation feature.
            flow (Tensor): The last estimated optical flow.

        Returns:
            Tensor: The output feature of motion encoder.
        """
        corr_feat = self.corr_net(corr)
        flow_feat = self.flow_net(flow)

        out = self.out_net(torch.cat([corr_feat, flow_feat], dim=1))
        return torch.cat([out, flow], dim=1)


class ConvGRU(BaseModule):
    """GRU with convolution layers.

    GRU cell with fully connected layers replaced with convolutions.

    Args:
        h_channels (int): Number of channels of hidden feature.
        x_channels (int): Number of channels of the concatenation of motion
            feature and context features.
        net_type (str):  Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
    """
    _kernel = {'Conv': 3, 'SeqConv': ((1, 5), (5, 1))}
    _padding = {'Conv': 1, 'SeqConv': ((0, 2), (2, 0))}

    def __init__(self,
                 h_channels: int,
                 x_channels: int,
                 net_type: str = 'SeqConv') -> None:
        super().__init__()
        assert net_type in ['Conv', 'SeqConv']
        kernel_size = self._kernel.get(net_type) if isinstance(
            self._kernel.get(net_type),
            (tuple, list)) else [self._kernel.get(net_type)]
        padding = self._padding.get(net_type) if isinstance(
            self._padding.get(net_type),
            (tuple, list)) else [self._padding.get(net_type)]

        conv_z = []
        conv_r = []
        conv_q = []

        for k, p in zip(kernel_size, padding):
            conv_z.append(
                ConvModule(
                    in_channels=h_channels + x_channels,
                    out_channels=h_channels,
                    kernel_size=k,
                    padding=p,
                    act_cfg=dict(type='Sigmoid')))
            conv_r.append(
                ConvModule(
                    in_channels=h_channels + x_channels,
                    out_channels=h_channels,
                    kernel_size=k,
                    padding=p,
                    act_cfg=dict(type='Sigmoid')))
            conv_q.append(
                ConvModule(
                    in_channels=h_channels + x_channels,
                    out_channels=h_channels,
                    kernel_size=k,
                    padding=p,
                    act_cfg=dict(type='Tanh')))
        self.conv_z = nn.ModuleList(conv_z)
        self.conv_r = nn.ModuleList(conv_r)
        self.conv_q = nn.ModuleList(conv_q)

    def init_weights(self) -> None:

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.orthogonal_(m.weight)

        self.apply(weights_init)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward function for ConvGRU.

        Args:
            h (Tensor): The last hidden state for GRU block.
            x (Tensor): The current input feature for GRU block

        Returns:
            Tensor: The current hidden state.
        """
        for conv_z, conv_r, conv_q in zip(self.conv_z, self.conv_r,
                                          self.conv_q):

            hx = torch.cat([h, x], dim=1)
            z = conv_z(hx)
            r = conv_r(hx)
            q = conv_q(torch.cat([r * h, x], dim=1))
            h = (1 - z) * h + z * q
        return h


class XHead(BaseModule):
    """A module for flow or mask prediction.

    Args:
        in_channels (int): Input channels of first convolution layer.
        feat_channels (Sequence(int)): List of features channels of different
            convolution layers.
        x_channels (int): Final output channels of predict layer.
        x (str): Type of predict layer. Choice: ['flow', 'mask']
    """

    def __init__(self, in_channels: int, feat_channels: Sequence[int],
                 x_channels: int, x: str) -> None:
        super().__init__()
        conv_layers = []
        for ch in feat_channels:
            conv_layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=ch,
                    kernel_size=3,
                    padding=1))
            in_channels = ch
        self.layers = nn.Sequential(*conv_layers)
        if x == 'flow':
            self.predict_layer = nn.Conv2d(
                feat_channels[-1], x_channels, kernel_size=3, padding=1)
        elif x == 'mask':
            self.predict_layer = nn.Conv2d(
                feat_channels[-1], x_channels, kernel_size=1, padding=0)
        else:
            raise ValueError(f'x must be \'flow\' or \'mask\', but got {x}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.predict_layer(x)


@DECODERS.register_module()
class RAFTDecoder(BaseDecoder):
    """The decoder of RAFT Net.

    The decoder of RAFT Net, which outputs list of upsampled flow estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
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

    def __init__(
        self,
        net_type: str,
        num_levels: int,
        radius: int,
        iters: int,
        corr_op_cfg: dict = dict(type='CorrLookup', align_corners=True),
        gru_type: str = 'SeqConv',
        feat_channels: Union[int, Sequence[int]] = 256,
        mask_channels: int = 64,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        flow_loss: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Small']
        assert type(feat_channels) in (int, tuple, list)
        self.corr_block = CorrelationPyramid(num_levels=num_levels)

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]
        self.net_type = net_type
        self.num_levels = num_levels
        self.radius = radius
        self.h_channels = self._h_channels.get(net_type)
        self.cxt_channels = self._cxt_channels.get(net_type)
        self.iters = iters
        self.mask_channels = mask_channels * (2 * radius + 1)
        corr_op_cfg['radius'] = radius
        self.corr_lookup = build_operators(corr_op_cfg)
        self.encoder = MotionEncoder(
            num_levels=num_levels,
            radius=radius,
            net_type=net_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gru_type = gru_type
        self.gru = self.make_gru_block()
        self.flow_pred = XHead(self.h_channels, feat_channels, 2, x='flow')

        if net_type == 'Basic':
            self.mask_pred = XHead(
                self.h_channels, feat_channels, self.mask_channels, x='mask')

        if flow_loss is not None:
            self.flow_loss = build_loss(flow_loss)

    def make_gru_block(self):
        return ConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 2 + self.cxt_channels,
            net_type=self.gru_type)

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
        scale = 2**(self.num_levels - 1)
        grid_size = self.radius * 2 + 1
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
                flow: torch.Tensor, h: torch.Tensor,
                cxt_feat: torch.Tensor) -> Sequence[torch.Tensor]:
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
        for _ in range(self.iters):
            flow = flow.detach()
            corr = self.corr_lookup(corr_pyramid, flow)
            motion_feat = self.encoder(corr, flow)
            x = torch.cat([cxt_feat, motion_feat], dim=1)
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

    def forward_train(
            self,
            feat1: torch.Tensor,
            feat2: torch.Tensor,
            flow: torch.Tensor,
            h_feat: torch.Tensor,
            cxt_feat: torch.Tensor,
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The last estimated flow from GRU cell.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of model.
        """

        flow_pred = self.forward(feat1, feat2, flow, h_feat, cxt_feat)

        return self.losses(flow_pred, flow_gt, valid=valid)

    def forward_test(self,
                     feat1: torch.Tensor,
                     feat2: torch.Tensor,
                     flow: torch.Tensor,
                     h_feat: torch.Tensor,
                     cxt_feat: torch.Tensor,
                     img_metas=None) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The last estimated flow from GRU cell.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """
        flow_pred = self.forward(feat1, feat2, flow, h_feat, cxt_feat)

        flow_result = flow_pred[-1]
        # flow maps with the shape [H, W, 2]
        flow_result = flow_result.permute(0, 2, 3, 1).cpu().data.numpy()
        # unravel batch dim
        flow_result = list(flow_result)
        flow_result = [dict(flow=f) for f in flow_result]
        return self.get_flow(flow_result, img_metas=img_metas)

    def losses(self,
               flow_pred: Sequence[torch.Tensor],
               flow_gt: torch.Tensor,
               valid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute optical flow loss.

        Args:
            flow_pred (Sequence[Tensor]): The list of predicted optical flow.
            flow_gt (Tensor): The ground truth of optical flow.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        loss = dict()
        loss['loss_flow'] = self.flow_loss(flow_pred, flow_gt, valid)
        return loss
