# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmflow.ops.builder import build_operators
from ..builder import DECODERS, build_components, build_loss
from ..utils import BasicDenseBlock, CorrBlock
from .base_decoder import BaseDecoder


class PWCModule(BaseModule):
    """Basic module of PWC-Net decoder.

    Args:
        in_channels (int): Input channels of basic dense block.
        up_flow (bool, optional): Whether to calculate upsampling flow and
            feature or not. Default: True.
        densefeat_channels (Sequence[int]): Number of output channels for
            dense layers. Default: (128, 128, 96, 64, 32).
        conv_cfg (dict, optional): Config dict of convolution layer in module.
            Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in module.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in module.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, optional): Config dict of initialization of BaseModule.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 up_flow: bool = True,
                 densefeat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.up_flow = up_flow
        self.dense_net = BasicDenseBlock(in_channels, densefeat_channels,
                                         conv_cfg, norm_cfg, act_cfg)
        self.last_channels = in_channels + sum(densefeat_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._make_predict_layer()
        self._make_upsample_layer()

    def _make_predict_layer(self) -> torch.nn.Module:
        self.predict_layer = nn.Conv2d(
            self.last_channels, 2, kernel_size=3, padding=1)

    def _make_upsample_layer(self) -> torch.nn.Module:

        if self.up_flow:
            self.upflow_layer = nn.ConvTranspose2d(
                2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat_layer = nn.ConvTranspose2d(
                self.last_channels, 2, kernel_size=4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function for PWCModule.

        Args:
            x (Tensor): The input feature.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The predicted optical flow,
                the feature to predict flow, the upsampled flow from the last
                level, and the upsampled feature.
        """
        feat = self.dense_net(x)
        flow = self.predict_layer(feat)
        upflow = None
        upfeat = None
        if self.up_flow:
            upflow = self.upflow_layer(flow)
            upfeat = self.upfeat_layer(feat)
        return flow, feat, upflow, upfeat


@DECODERS.register_module()
class PWCNetDecoder(BaseDecoder):
    """The Decoder of PWC-Net.

    The decoder of PWC-Net which outputs flow predictions and features.

    Args:
        in_channels (dict): Dict of number of input channels for each level.
        densefeat_channels (Sequence[int]): Number of output channels for
            dense layers. Default: (128, 128, 96, 64, 32).
        flow_div (float): The divisor works for scaling the ground truth.
            Default: 20.
        corr_cfg (dict): Config for correlation layer.
            Defaults to dict(type='Correlation', max_displacement=4).
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Defaults to False.
        warp_cfg (dict): Config for warp operation. Defaults to
            dict(type='Warp', align_corners=True).
        conv_cfg (dict, optional): Config of convolution layer in module.
            Default: None.
        norm_cfg (dict, optional): Config of norm layer in module.
            Default: None.
        act_cfg (dict, optional): Config of activation layer in module.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        post_processor (dict, optional): Config of flow post process module.
            Default: None
        flow_loss: Config of loss function of optical flow. Default: None.
        init_cfg (dict, list, optional): Config of dict weights initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: Dict[str, int],
                 densefeat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 flow_div: float = 20.,
                 corr_cfg: dict = dict(type='Correlation', max_displacement=4),
                 scaled: bool = False,
                 warp_cfg: dict = dict(type='Warp', align_corners=True),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 post_processor: dict = None,
                 flow_loss: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        assert isinstance(in_channels, dict)

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.densefeat_channels = densefeat_channels
        self.flow_div = flow_div

        self.flow_levels = list(in_channels.keys())
        self.flow_levels.sort()
        self.start_level = self.flow_levels[-1]
        self.end_level = self.flow_levels[0]

        self.corr_cfg = corr_cfg
        self.scaled = scaled
        self.warp_cfg = warp_cfg

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if flow_loss is not None:
            self.flow_loss = build_loss(flow_loss)

        self._make_corr_block(self.corr_cfg, self.act_cfg, self.scaled)

        if warp_cfg is not None:
            self._make_warp(self.warp_cfg)

        self.multiplier = dict()
        for level in self.flow_levels:
            self.multiplier[level] = self.flow_div * 2**(-int(level[-1]))

        self.post_processor = (
            build_components(post_processor)
            if post_processor is not None else None)

        self._make_layers()

    def _make_layers(self) -> None:
        """Build sub-modules of this decoder."""
        layers = []

        for level in self.flow_levels:
            up_flow = (level != self.end_level)
            layers.append([
                level,
                self._make_layer(self.in_channels[level], up_flow=up_flow)
            ])
        self.decoders = nn.ModuleDict(layers)

    def _make_layer(self,
                    in_channels: int,
                    up_flow: bool = True) -> torch.nn.Module:
        """Build module at each level of this decoder.

        Args:
            in_channels (int): The channels of input feature
            up_sample (bool): Whether upsample flow for the next level.
                Defaults to True.

        Returns:
            torch.nn.Module: The sub-module for this decoder.
        """

        return PWCModule(
            in_channels,
            up_flow,
            self.densefeat_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _make_corr_block(self, corr_cfg: dict, act_cfg: dict,
                         scaled: bool) -> None:
        """Make correlation.

        Args:
            corr_cfg (dict): Config for correlation layer.
            act_cfg (dict): Config of activation layer in module.
            scaled (bool): Whether to use scaled correlation by the number of
                elements involved to calculate correlation or not.
        """
        self.corr_block = CorrBlock(
            corr_cfg=corr_cfg, act_cfg=act_cfg, scaled=scaled)

    def _make_warp(self, warp_cfg: dict) -> None:
        """Build warp operator.

        Args:
            warp_cfg (dict): Config for warp operation.
        """
        self.warp = build_operators(warp_cfg)

    def forward(self, feat1: Dict[str, torch.Tensor],
                feat2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function for PWCNetDecoder.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.

        Returns:
            Dict[str, Tensor]: The predicted multi-levels optical flow.
        """

        flow_pred = dict()
        flow = None
        upflow = None
        upfeat = None

        for level in self.flow_levels[::-1]:
            _feat1, _feat2 = feat1[level], feat2[level]

            if level == self.start_level:
                corr_feat = self.corr_block(_feat1, _feat2)
            else:
                warp_feat = self.warp(_feat2, upflow * self.multiplier[level])
                corr_feat_ = self.corr_block(_feat1, warp_feat)
                corr_feat = torch.cat((corr_feat_, _feat1, upflow, upfeat),
                                      dim=1)

            flow, feat, upflow, upfeat = self.decoders[level](corr_feat)

            flow_pred[level] = flow

        if self.post_processor is not None:
            post_flow = self.post_processor(feat)
            flow_pred[self.end_level] = flow_pred[self.end_level] + post_flow

        return flow_pred

    def forward_train(
            self,
            feat1: Dict[str, torch.Tensor],
            feat2: Dict[str, torch.Tensor],
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
            flow_gt (Tensor): The ground truth of optical flow from image1 to
                image2.
            valid (Tensor, optional): The valid mask of optical flow ground
                truth. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        flow_pred = self.forward(feat1, feat2)
        return self.losses(flow_pred, flow_gt, valid=valid)

    def forward_test(
        self,
        feat1: Dict[str, torch.Tensor],
        feat2: Dict[str, torch.Tensor],
        H: int,
        W: int,
        img_metas: Optional[Sequence[dict]] = None
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model testing.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
            H (int): The height of images after data augmentation.
            W (int): The width of images after data augmentation.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """

        flow_pred = self.forward(feat1, feat2)
        flow_result = flow_pred[self.end_level]

        # resize flow to the size of images after augmentation.
        flow_result = F.interpolate(
            flow_result, size=(H, W), mode='bilinear', align_corners=False)
        # reshape [2, H, W] to [H, W, 2]
        flow_result = flow_result.permute(0, 2, 3,
                                          1).cpu().data.numpy() * self.flow_div

        # unravel batch dim,
        flow_result = list(flow_result)
        flow_result = [dict(flow=f) for f in flow_result]

        return self.get_flow(flow_result, img_metas=img_metas)

    def losses(
            self,
            flow_pred: Dict[str, torch.Tensor],
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute optical flow loss.

        Args:
            flow_pred (Dict[str, Tensor]): multi-level predicted optical flow.
            flow_gt (Tensor): The ground truth of optical flow.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        loss = dict()
        loss['loss_flow'] = self.flow_loss(flow_pred, flow_gt, valid)
        return loss
