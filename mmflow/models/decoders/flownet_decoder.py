# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule

from ..builder import DECODERS, build_loss
from .base_decoder import BaseDecoder


class DeconvModule(BaseModule):
    """Basic deconvolution module for FlowNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, tuple): Size of deconvolution kernel. Default: 4.
        stride (int, tuple): Whether use stride deconvolution to upsample.
            Default: 2.
        padding (int, tuple): Padding size of deconvolution. Default: 1.
        dilation (int, tuple): Dilation of deconvolution. Default: 1.
        bias (bool): Whether add bias in deconvolution layers. Default: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...] = 4,
                 stride: Tuple[int, ...] = 2,
                 padding: Tuple[int, ...] = 1,
                 dilation: Tuple[int, ...] = 1,
                 bias: bool = False,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None) -> None:
        super().__init__()

        deconvs = []

        deconvs.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias))

        if norm_cfg is not None:
            deconvs.append(build_norm_layer(norm_cfg, out_channels))

        if act_cfg is not None:
            deconvs.append(build_activation_layer(act_cfg))

        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for deconvolution module.

        Args:
            x (Tensor): Input feature.

        Returns:
            Tensor: Output feature of deconvolution module.
        """
        out = self.deconvs(x)
        return out


class BasicBlock(BaseModule):
    """Basic block of FlowNetDecoder.

    The block includes one convolution for prediction and one
    deconvolution for next block.

    Args:
        in_channels (int): Number of input channels.
        pred_channels (int): Number of prediction channels. If 1, predict
            occlusion map. If 2, predict flow map.
        out_channels (int): Number of output channels of DeconvModule.
        inter_channels (int, Optional): Number of output channels of inter
            convolution, which is the same as the number of input channels for
            flow prediction convolution. If None, there is no inter convolution
            between input and flow prediction convolution and the input channel
            of flow prediction convolution is the same as in_channels.
            Default: None.
        deconv_bias (bool): Whether add bias term in deconvolution module.
            Default: False.
        pred_bias (bool): Whether add bias term in prediction module.
            Default: False.
        upsample_bias (bool): Whether add bias term in upsample module.
            Default: False.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for each activation layer in
            ConvModule. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 pred_channels: int,
                 out_channels: Optional[int] = None,
                 inter_channels: int = None,
                 deconv_bias: bool = False,
                 pred_bias: bool = False,
                 upsample_bias: bool = False,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None) -> None:
        super().__init__()
        if inter_channels is None:
            self.pred_out = nn.Conv2d(
                in_channels=in_channels,
                out_channels=pred_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=pred_bias)
        else:
            pred_out = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=pred_bias),
                nn.Conv2d(
                    in_channels=inter_channels,
                    out_channels=pred_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=pred_bias)
            ]
            self.pred_out = nn.Sequential(*pred_out)
        self.up_sample = out_channels is not None
        if self.up_sample:
            self.deconv_out = DeconvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=deconv_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.upsample_pred = DeconvModule(
                in_channels=pred_channels,
                out_channels=pred_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=upsample_bias)

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function for basic block of FlowNetDecoder.

        Args:
            x (Tensor): Input feature.

        Returns:
            tuple: Predicted optical flow, upsampled optical flow and
                upsampled input feature.
        """
        upflow = None
        upfeat = None
        flow = self.pred_out(x)
        if self.up_sample:
            upfeat = self.deconv_out(x)
            upflow = self.upsample_pred(flow)

        return flow, upflow, upfeat


@DECODERS.register_module()
class FlowNetSDecoder(BaseDecoder):
    """The decoder of FlowNetS.

    This module works for predicting flow and computing loss.

    Args:
        in_channels (dict): Dict of number of input channels for each level.
        out_channels (dict): Dict of number of output channels of deconvolution
            module for each level.
        pred_channels (int): Number of prediction channels. If 1, predict
            occlusion map. If 2, predict flow map. Default: 2.
        flow_div (float): The divisor works for scaling the ground truth.
            Default: 20.
        inter_channels (Sequence[int], optional): List of numbers of output
            channels of inter convolution in each BasicBlock. Check the
            description of this argument in BasicBlock for more details.
            Default: None.
        deconv_bias:  Whether add bias term in deconvolution module.
            Default: False.
        pred_bias (bool): Whether add bias term in prediction module.
            Default: False.
        upsample_bias (bool): Whether add bias term in upsample module.
            Default: False.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for each activation layer in
            ConvModule. Default: dict(type='LeakyReLU', negative_slope=0.1).
        flow_loss: Config of loss function of optical flow. Default: None.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: Dict[str, int],
                 out_channels: Dict[str, int],
                 pred_channels: int = 2,
                 flow_div: float = 20.,
                 inter_channels: Optional[Sequence] = None,
                 deconv_bias: bool = False,
                 pred_bias: bool = False,
                 upsample_bias: bool = False,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 flow_loss: Optional[dict] = None,
                 init_cfg=None) -> None:

        super().__init__(init_cfg)

        self.decoders = nn.ModuleDict()
        self.flow_levels = list(in_channels.keys())
        self.flow_levels.sort()
        self.start_level = self.flow_levels[-1]
        self.end_level = self.flow_levels[0]
        self.flow_div = flow_div

        if flow_loss is not None:
            self.flow_loss = build_loss(flow_loss)

        layers = []
        for level in self.flow_levels:
            inter_ch = inter_channels.get(level) if isinstance(
                inter_channels, dict) else inter_channels
            layers.append([
                level,
                BasicBlock(
                    in_channels=in_channels[level],
                    pred_channels=pred_channels,
                    inter_channels=inter_ch,
                    out_channels=out_channels.get(level, None),
                    deconv_bias=deconv_bias,
                    pred_bias=pred_bias,
                    upsample_bias=upsample_bias,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ])
        self.decoders = nn.ModuleDict(layers)

    def forward(self, feat: Dict[str,
                                 torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function for decoder of FlowNetS.

        Args:
            feat (Dict[str, Tensor]): Input feature pyramid which is encoded
                images.

        Returns:
            Dict[str, Tensor]: Multi-level predicted optical flow.
        """

        flow_pred = dict()
        upfeat = None
        upflow = None

        for level in self.flow_levels[::-1]:

            if level == self.start_level:
                feat_ = feat[level]
            else:
                feat_ = torch.cat((feat[level], upfeat, upflow), dim=1)

            flow, upflow, upfeat = self.decoders[level](feat_)
            flow_pred[level] = flow

        return flow_pred

    def forward_train(
            self,
            *args,
            flow_gt: Optional[torch.Tensor] = None,
            valid: torch.Tensor = None,
            return_multi_level_flow: bool = False) -> Dict[str, torch.Tensor]:
        """Forward function for decoder of FlowNetS when model training.

        Args:
            flow_gt (Tensor, optional): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            return_multi_level_flow (bool, optional): The flag to control
                whether do not calculate the loss and  only return the
                multi-level optical flow from forward function. If set True,
                this model is a sub-model in Flownet2 and do not calculate the
                loss. Defaults to False.

        Returns:
            Dict[str, Tensor]: The losses of output or multi-level predicted
                optical flow.
        """

        flow_pred = self.forward(*args)

        if return_multi_level_flow:
            return flow_pred

        return self.losses(flow_pred, flow_gt, valid)

    def forward_test(
        self,
        *args,
        H: int,
        W: int,
        return_multi_level_flow: bool = False,
        img_metas: Optional[Sequence[dict]] = None
    ) -> Union[Dict[str, torch.Tensor], Sequence[Dict[str, np.ndarray]]]:
        """Forward function for decoder of FlowNetS when model testint.

        Args:
            H (int): The height of images after data augmentation.
            W (int): The width of images after data augmentation.
            return_multi_level_flow (bool, optional): The flag to control
                whether do not calculate the loss and  only return the
                multi-level optical flow from forward function. If set True,
                this model is a sub-model in Flownet2 and do not calculate the
                loss. Defaults to False.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Union[Dict[str, Tensor], Sequence[Dict[str, ndarray]]]: multi-level
                predicted optical flow or the predicted optical flow with the
                same size of images before augmentation.
        """

        flow_pred = self.forward(*args)

        # it must be one of estimator in flownet2
        if return_multi_level_flow:
            return flow_pred

        flow_result = flow_pred[self.end_level]
        # resize flow to the size of images after augmentation.
        flow_result = F.interpolate(
            flow_result, size=(H, W), mode='bilinear', align_corners=False)
        # reshape [2, H, W] to [H, W, 2]
        flow_result = flow_result.permute(0, 2, 3,
                                          1).cpu().data.numpy() * self.flow_div
        # unravel batch dim
        flow_result = list(flow_result)
        flow_result = [dict(flow=f) for f in flow_result]

        return self.get_flow(flow_result, img_metas=img_metas)

    def losses(self,
               flow_pred: Dict[str, torch.Tensor],
               flow_gt: torch.Tensor,
               valid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """The loss function for Flownet.

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


@DECODERS.register_module()
class FlowNetCDecoder(FlowNetSDecoder):
    """The decoder of FlowNetS."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, feat1: Dict[str, torch.Tensor],
                corr_feat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function for decoder of FlowNetS.

        Args:
            feat1 (Dict[str, Tensor]): Input feature pyramid which is encoded
                image1.
            corr_feat (Dict[str, Tensor]): Input feature pyramid which is
                encoded correlation of feature1 and feature2 that are the third
                -level feature of image1 and image2.

        Returns:
            Dict[str, Tensor]: Multi-level predicted optical flow.
        """

        flow_pred = dict()
        upflow = None
        upfeat = None
        for level in self.flow_levels[::-1]:
            if level == self.start_level:
                feat = corr_feat[level]
            else:
                if corr_feat.get(level) is None:
                    feat = torch.cat((feat1[level], upfeat, upflow), dim=1)
                else:
                    feat = torch.cat((corr_feat[level], upfeat, upflow), dim=1)

            flow, upflow, upfeat = self.decoders[level](feat)

            flow_pred[level] = flow

        return flow_pred
