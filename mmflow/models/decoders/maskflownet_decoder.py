# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.ops import DeformConv2d
from mmcv.runner import BaseModule

from ..builder import DECODERS
from ..utils import CorrBlock
from .pwcnet_decoder import PWCModule, PWCNetDecoder


def Upsample(img, factor) -> torch.Tensor:
    """Upsampling function.

    Args:
        img (tensor(B, C, H, W)): Input image for upsampling.
        factor (int): Upsampling factor.

    Returns:
        Upsampled image.
    """
    if factor == 1:
        return img
    _, _, H, W = img.shape
    img = F.pad(img, [0, 1, 0, 1], mode='replicate')
    upsamp_img = F.interpolate(
        img, (H * factor + 1, W * factor + 1),
        mode='bilinear',
        align_corners=True)
    upsamp_img = upsamp_img[:, :, :-1, :-1]

    return upsamp_img


class BasicDeformWarpBlock(BaseModule):
    """Basic Deformable Warp Block.

    Args:
        channels (int): Input channels of deformable convolution.
        act_cfg (dict): Config dict of activation layer.
        with_deform_bias (bool): Whether to use bias in deformable convolution
            or not. Default: True.
        init_cfg (dict, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 channels: int,
                 act_cfg: dict,
                 with_deform_bias: bool = True,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(init_cfg)
        self.channels = channels
        self.deconv = DeformConv2d(channels, channels, 3, padding=1)
        self.act = build_activation_layer(act_cfg)
        self.with_deform_bias = with_deform_bias
        if self.with_deform_bias:
            self.deconv_bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, feat2: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Forward function for BasicDeformWarpBlock.

        Args:
            feat2 (Tensor): The feature from the second image.
            flow (Tensor): The optical flow used for DCN offset.

        Returns:
            Tensor: The output feature from BasicDeformWarpBlock.
        """
        B, _, H, W = flow.shape
        conv_offset = torch.repeat_interleave(
            flow[:, None], repeats=9, dim=1).reshape(B, -1, H, W)
        if self.with_deform_bias:
            deform_feat = self.deconv(
                feat2, conv_offset) + self.deconv_bias.expand(B, -1, -1, -1)
        else:
            deform_feat = self.deconv(feat2, conv_offset)
        return self.act(deform_feat)


class DeformWarpBlock(BaseModule):
    """Deformable Warp Block.

    Args:
        channels (int): Input channels of deformable convolution.
        up_channels (int): Input channels of trade-off convolution layer for
            upsampled feat.
        act_cfg (dict): Config of activation layer.
        with_deform_bias (bool, optional): Whether to use bias in
            deformable convolution or not. Default: True.
        init_cfg (dict, optional): Config dict for initialization of module.
            Default: None.
    """

    def __init__(self,
                 channels: int,
                 up_channels: int,
                 act_cfg: dict,
                 with_deform_bias: bool = True,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(init_cfg)
        self.channels = channels
        self.deconv = DeformConv2d(channels, channels, 3, padding=1)
        self.tradeoff_conv = nn.Conv2d(up_channels, channels, 3, padding=1)
        self.act = build_activation_layer(act_cfg)
        self.with_deform_bias = with_deform_bias
        if self.with_deform_bias:
            self.deconv_bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, feat2: torch.Tensor, flow: torch.Tensor,
                mask_feat: torch.Tensor,
                up_feat: torch.Tensor) -> torch.Tensor:
        """Forward function for DeformWarpBlock.

        Args:
            feat2 (Tensor): The feature from the second image.
            flow (Tensor): The optical flow used for DCN offset.
            mask_feat (Tensor): The learnable occlusion mask.
            up_feat (Tensor): The upsampled feature from the last level.

        Returns:
            Tensor: The output feature from DeformWarpBlock.
        """

        assert mask_feat.shape[1] == 1
        B, _, H, W = flow.shape
        # repeat = kernel_size*kernel_size
        conv_offset = torch.repeat_interleave(
            flow[:, None], repeats=9, dim=1).reshape(B, -1, H, W)
        if self.with_deform_bias:
            deform_feat = self.deconv(
                feat2, conv_offset) + self.deconv_bias.expand(B, -1, -1, -1)
        else:
            deform_feat = self.deconv(feat2, conv_offset)
        tradeoff_feat = self.tradeoff_conv(up_feat)
        mask_feat = torch.squeeze(torch.sigmoid(mask_feat), dim=1)
        warp_feat = torch.einsum('ijkl, ikl->ijkl', deform_feat,
                                 mask_feat) + tradeoff_feat
        return self.act(warp_feat)


class WarpCorrBlock(BaseModule):
    """Warp Correlation Block.

    Args:
        channels (int): Input channels of DeformWarpBlock.
        corr_cfg (dict): Config of BasicCorrBlock.
        up_channels (int, optional): Input channels of trade-off convolution
            layer for upsampled feat, if set to None, the warp_type must be
            'Basic'. Default: None.
        warp_type (str): Type of Warp block, it has 2 options 'Basic' and
            'AsyOFMM'. Default: "Basic".
        act_cfg (dict): Config dict of activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Defaults to False.
        with_deform_bias (bool): Whether to use bias in DeformConv2d or not.
            Default: True.
        init_cfg (dict, optional): Config dict for initialization of module.
            Default: None.
    """

    def __init__(self,
                 channels: int,
                 corr_cfg: dict,
                 up_channels: Optional[int] = None,
                 warp_type: str = 'Basic',
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 scaled: bool = False,
                 with_deform_bias: bool = True,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        assert warp_type in ('Basic', 'AsymOFMM')

        self.channels = channels
        self.up_channels = up_channels
        self.corr_cfg = corr_cfg
        self.warp_type = warp_type
        self.act_cfg = act_cfg
        self.scaled = scaled
        self.with_deform_bias = with_deform_bias
        self._make_layers()

    def _make_layers(self) -> None:
        """Build warp and correlation block as the given type."""
        if self.warp_type == 'AsymOFMM':
            self.warp = DeformWarpBlock(
                channels=self.channels,
                up_channels=self.up_channels,
                act_cfg=self.act_cfg,
                with_deform_bias=self.with_deform_bias)
        elif self.warp_type == 'Basic':
            self.warp = BasicDeformWarpBlock(
                channels=self.channels,
                act_cfg=self.act_cfg,
                with_deform_bias=self.with_deform_bias)
        self.corr = CorrBlock(self.corr_cfg, self.act_cfg, scaled=self.scaled)

    def forward(self,
                feat1: torch.Tensor,
                feat2: torch.Tensor,
                up_flow: torch.Tensor,
                up_mask: Optional[torch.Tensor] = None,
                up_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function for WarpCorrBlock.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            up_flow (Tensor):  The upsampled optical flow from the last level.
            up_mask (Tensor, optional): The upsampled learnable occlusion mask
                from the last level. Defaults to None.
            up_feat (Tensor, optional): The upsampled feature from the last
                level.

        Returns:
            Tensor: The correlation between feat1 and the warped feat2.
        """
        if self.warp_type == 'AsymOFMM':
            warp_feat = self.warp(feat2, up_flow, up_mask, up_feat)
        elif self.warp_type == 'Basic':
            warp_feat = self.warp(feat2, up_flow)
        corr_feat = self.corr(feat1, warp_feat)

        return corr_feat


class MaskModule(PWCModule):
    """Basic module of Mask-FlowNet.

    Args:
        up_channels (int): Output channels of feature upsampling layer.
        with_mask (bool): Whether to predict mask or not.
    """

    def __init__(self, up_channels: int, with_mask: bool, *args,
                 **kwargs) -> torch.Tensor:
        self.up_channels = up_channels
        self.with_mask = with_mask
        super().__init__(*args, **kwargs)

    def _make_predict_layer(self) -> None:

        self.predict_flow = nn.Conv2d(
            self.last_channels, 2, kernel_size=3, padding=1)

        if self.with_mask:
            self.predict_mask = nn.Conv2d(
                self.last_channels, 1, kernel_size=3, padding=1)

    def _make_upsample_layer(self) -> None:
        if self.up_flow:
            self.upfeat_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    self.last_channels,
                    out_channels=self.up_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1), build_activation_layer(self.act_cfg))

    def forward(
        self, x: torch.Tensor, upflow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Forward function for MaskModule.

        Args:
            x (Tensor): The input feature.
            upflow (Tensor): The upsampled optcal flow from the last level.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: The
                predicted optical flow, the predicted occlusion mask,
                the feature to predict optical flow, the upsample flow from the
                last level, the upsampled occlusion mask from the last level,
                and upsampled feature from the last level.
        """
        feat = self.dense_net(x)
        flow = self.predict_flow(feat) + upflow

        mask = None
        upflow = None
        upmask = None
        upfeat = None
        if self.with_mask:
            mask = self.predict_mask(feat)
            if self.up_flow:
                upmask = Upsample(mask, factor=2)
        if self.up_flow:
            upflow = Upsample(flow, factor=2)
            # upfeat will used to make trade-off feat.
            upfeat = self.upfeat_layer(feat)

        return flow, mask, feat, upflow, upmask, upfeat


@DECODERS.register_module()
class MaskFlowNetSDecoder(PWCNetDecoder):
    """The decoder module of MaskFlowNetS.

    Args:
        warp_in_channels (dict): Dict input channels of warp block.
        up_channels (dict): Dict input channels of feat upsampling layer at
            each level.
        with_mask (bool, optional): Whether to predict mask or not.
            If not define, decoder predicts mask depending on whether upsample
            feat from the last level. Default: None.
        warp_type (str): Type of Warp block, it has 2 options 'Basic' and
            'AsyOFMM'. Default: "Basic".
        with_deform_bias (bool): Whether to use bias in DeformConv2d or not.
            Default: True.
    """

    def __init__(self,
                 warp_in_channels: Dict[str, int],
                 up_channels: Dict[str, int],
                 with_mask: Optional[bool] = None,
                 warp_type: str = 'AsymOFMM',
                 with_deform_bias: bool = True,
                 *args,
                 **kwargs) -> None:

        assert isinstance(warp_in_channels, dict)
        assert isinstance(up_channels, dict)

        self.warp_in_channels = warp_in_channels
        self.up_channels = up_channels
        self.with_mask = with_mask
        self.warp_type = warp_type
        self.with_deform_bias = with_deform_bias

        super().__init__(*args, **kwargs)

    def _make_layers(self) -> None:
        """Build sub-modules of this decoder."""
        layers = []
        for level in self.flow_levels:
            up_sample = level != self.end_level
            # predicts mask depending on whether upsample
            with_mask = up_sample if self.with_mask is None else self.with_mask
            up_channels = self.up_channels.get(level)

            layers.append([
                level,
                self._make_layer(self.in_channels[level], up_channels,
                                 up_sample, with_mask)
            ])
            self.decoders = nn.ModuleDict(layers)

    def _make_layer(self,
                    in_channels: int,
                    up_channels: Optional[int] = None,
                    up_sample: bool = True,
                    with_mask: bool = True) -> torch.nn.Module:
        """Build module at each level of this decoder.

        Args:
            in_channels (int): The channels of input feature
            up_channels (int, optional): The channels of upsampled features.
                Defaults to None.
            up_sample (bool): Whether upsample feature for the next level.
                Defaults to True.
            with_mask (bool): Whether predict occlusion mask.
                Defaults to True.

        Returns:
            torch.nn.Module: The sub-module for this decoder.
        """
        return MaskModule(
            up_channels,
            with_mask,
            in_channels,
            up_sample,
            self.densefeat_channels,
            act_cfg=self.act_cfg)

    def _make_corr_block(self, corr_cfg: dict, act_cfg: dict,
                         scaled: bool) -> None:
        """Make correlation block for different optical flow.

        Args:
            corr_cfg (dict): Config for correlation layer.
            act_cfg (dict): Config of activation layer in module.
            scaled (bool): Whether to use scaled correlation by the number of
                elements involved to calculate correlation or not.
        """
        self.corr_block = nn.ModuleDict()

        for level in self.flow_levels:
            if level == self.start_level:
                self.corr_block[level] = CorrBlock(corr_cfg, act_cfg, scaled)
            else:
                self.corr_block[level] = WarpCorrBlock(
                    channels=self.warp_in_channels[level],
                    corr_cfg=corr_cfg,
                    up_channels=self.up_channels[level],
                    warp_type=self.warp_type,
                    act_cfg=act_cfg,
                    scaled=scaled,
                    with_deform_bias=self.with_deform_bias)

    def forward(
        self,
        feat1: Dict[str, torch.Tensor],
        feat2: Dict[str, torch.Tensor],
        return_mask: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor],
                                              torch.Tensor]]:
        """Forward function for MaskFlownet decoder.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.

        Returns:
            Dict[str, Tensor]: The predicted multi-levels optical flow.
        """
        minH, minW = feat1[self.start_level].shape[2:]
        flow_pred = dict()
        flow = None
        upmask = None
        upfeat = None
        upflow = torch.zeros(1, 2, minH, minW).to(feat1[self.start_level])

        for level in self.flow_levels[::-1]:
            _feat1, _feat2 = feat1[level], feat2[level]

            if level == self.start_level:
                corr_feat = self.corr_block[level](_feat1, _feat2)
            else:

                corr_feat_ = self.corr_block[level](
                    _feat1, _feat2, upflow * self.multiplier[level], upmask,
                    upfeat)
                corr_feat = torch.cat((corr_feat_, _feat1, upfeat, upflow),
                                      dim=1)
            flow, _, feat, upflow, upmask, upfeat = self.decoders[level](
                corr_feat, upflow)

            if level == 'level3':
                # the upsampled mask in level3 is the last mask in stage1
                # of MaskFlowNet.
                last_mask = upmask

            # offset in dcn is (y0,x0,...) but in value in flow map is (u, v)
            flow_pred[level] = flow.flip(1)

        if self.post_processor is not None:
            post_flow = self.post_processor(feat)
            flow_pred[self.end_level] = flow_pred[
                self.flow_levels[0]] + post_flow.flip(1)

        if return_mask:
            # Stage2 of MaskFlowNet need input mask.
            return flow_pred, Upsample(last_mask, 4)
        else:
            return flow_pred


@DECODERS.register_module()
class MaskFlowNetDecoder(MaskFlowNetSDecoder):
    """The decoder module of MaskFlowNet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_corr_block(self, corr_cfg: dict, act_cfg: dict,
                         scaled: bool) -> None:
        self.corr_layer = CorrBlock(
            self.corr_cfg, self.act_cfg, scaled=self.scaled)
        self.corr_block = nn.ModuleDict()

        for level in self.flow_levels:
            self.corr_block[level] = WarpCorrBlock(
                channels=self.warp_in_channels[level],
                corr_cfg=corr_cfg,
                up_channels=self.up_channels[level],
                warp_type=self.warp_type,
                act_cfg=act_cfg,
                scaled=scaled,
                with_deform_bias=self.with_deform_bias)

    def forward(
        self,
        feat1: Dict[str, torch.Tensor],
        feat2: Dict[str, torch.Tensor],
        feat3: Dict[str, torch.Tensor],
        feat4: Dict[str, torch.Tensor],
        flows_stage1: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward function for MaskFlowNetDecoder.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image from stage1 of MaskFlowNet.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image from stage1 of MaskFlowNet.
            feat3 (Dict[str, Tensor]): The feature pyramid from the first
                image from stage2 of MaskFlowNet.
            feat4 (Dict[str, Tensor]): The feature pyramid from the second
                image from stage2 of MaskFlowNet.
            flows_stage1 (Dict[str, Tensor]): Estimated multi-level flow from
                the stage1.

        Returns:
            Dict[str, Tensor]: The predicted multi-levels optical flow.
        """

        flows_pred = dict()
        upfeat = None
        upflow = None

        for level in self.flow_levels[::-1]:
            i_feat1, i_feat2, i_feat3, i_feat4, i_flow = feat1[level], feat2[
                level], feat3[level], feat4[level], flows_stage1[level]

            if level == self.start_level:
                upflow = i_flow

            corr_feat1 = self.corr_block[level](
                i_feat1, i_feat2, self.multiplier[level] * upflow)
            corr_feat2 = self.corr_layer(i_feat3, i_feat4)
            corr_feat = torch.cat((corr_feat1, corr_feat2), dim=1)

            if upfeat is None:
                flow, _, feat, upflow, _, upfeat = self.decoders[level](
                    torch.cat((corr_feat, upflow), dim=1), upflow)
            else:
                flow, _, feat, upflow, _, upfeat = self.decoders[level](
                    torch.cat((i_feat1, upfeat, corr_feat, upflow, i_flow),
                              dim=1), upflow)

            flows_pred[level] = flow.flip(1)

        if self.post_processor is not None:
            post_flow = self.post_processor(feat)
            flows_pred[self.end_level] = flows_pred[
                self.flow_levels[0]] + post_flow.flip(1)

        return flows_pred

    def forward_train(
            self,
            feat1: Dict[str, torch.Tensor],
            feat2: Dict[str, torch.Tensor],
            feat3: Dict[str, torch.Tensor],
            feat4: Dict[str, torch.Tensor],
            flows_stage1: Dict[str, torch.Tensor],
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image from stage1 of MaskFlowNet.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image from stage1 of MaskFlowNet.
            feat3 (Dict[str, Tensor]): The feature pyramid from the first
                image from stage2 of MaskFlowNet.
            feat4 (Dict[str, Tensor]): The feature pyramid from the second
                image from stage2 of MaskFlowNet.
            flows_stage1 (Dict[str, Tensor]): Estimated multi-level flow from
                the stage1.
            flow_gt (Tensor): The ground truth of optical flow from image1 to
                image2.
            valid (Tensor, optional): The valid mask of optical flow ground
                truth. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        flow_pred = self.forward(feat1, feat2, feat3, feat4, flows_stage1)
        return self.losses(flow_pred, flow_gt, valid=valid)

    def forward_test(
        self,
        feat1: Dict[str, torch.Tensor],
        feat2: Dict[str, torch.Tensor],
        feat3: Dict[str, torch.Tensor],
        feat4: Dict[str, torch.Tensor],
        flows_stage1: Dict[str, torch.Tensor],
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

        flow_pred = self.forward(feat1, feat2, feat3, feat4, flows_stage1)
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
