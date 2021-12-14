# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import BaseModule

from mmflow.ops import build_operators
from ..builder import DECODERS, build_loss
from ..utils import CorrBlock
from .base_decoder import BaseDecoder


class Upsample(nn.Module):
    """Upsampling module.

    Args:
        scale_factor (int): Scale factor of upsampling.
        channels (int): Number of channels of conv_transpose2d.
    """

    def __init__(self, scale_factor: int, channels: int) -> None:
        super().__init__()
        self.kernel_size = 2 * scale_factor - scale_factor % 2
        self.stride = scale_factor
        self.pad = math.ceil((scale_factor - 1) / 2.)
        self.channels = channels
        self.register_buffer('weight', self.bilinear_upsampling_filter())

    # caffe::BilinearFilter
    def bilinear_upsampling_filter(self) -> torch.Tensor:
        """Generate the weights for caffe::BilinearFilter.

        Returns:
            Tensor: The weights for caffe::BilinearFilter
        """
        f = math.ceil(self.kernel_size / 2.)
        c = (2 * f - 1 - f % 2) / 2. / f
        weight = torch.zeros(self.kernel_size**2)
        for i in range(self.kernel_size**2):
            x = i % self.kernel_size
            y = (i / self.kernel_size) % self.kernel_size
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return weight.view(1, 1, self.kernel_size,
                           self.kernel_size).repeat(self.channels, 1, 1, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward function for upsample.

        Args:
            data (Tensor): The input data.

        Returns:
            Tensor: The upsampled data.
        """
        return F.conv_transpose2d(
            data,
            self.weight,
            stride=self.stride,
            padding=self.pad,
            groups=self.channels)


class BasicBlock(BaseModule):
    """Basic convolution block for LiteFlowNet decoder modules.

    Args:
        in_channels (int): Input channels of the first convolution layer.
        feat_channels (Sequence[int]): Output channels of convolution layers in
            the block.
        conv_cfg (dict): Config of convolution layers.
        norm_cfg (dict): Config of normalization layers.
        act_cfg (dict): Config of activation layers.
        init_cfg (dict): Config of weights initialization.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: Sequence[int],
        conv_cfg: dict,
        norm_cfg: dict,
        act_cfg: dict,
        init_cfg: dict,
    ) -> None:

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        layers = []
        for ch in self.feat_channels:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=ch,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = ch
        self.feat_out_channels = feat_channels[-1]
        self.layers = nn.Sequential(*layers)


class MatchingBlock(BasicBlock):
    """Matching Block.

    Args:
        corr_cfg (dict): Config for building correlation operator.
        warp_cfg (dict): Config for building warp operator.
        last_kernel_size (int): Kernel size of the last convolution layer.
        scaled_corr (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
    """

    def __init__(
        self,
        *args,
        corr_cfg: dict,
        warp_cfg: dict,
        last_kernel_size: int,
        scaled_corr: bool,
    ) -> None:

        super().__init__(*args)
        self.corr_cfg = corr_cfg
        self.warp_op = build_operators(warp_cfg)
        self.pred_flow = nn.Conv2d(
            self.feat_out_channels,
            2,
            kernel_size=last_kernel_size,
            stride=1,
            padding=last_kernel_size // 2)

        self.corr = CorrBlock(self.corr_cfg, self.act_cfg, scaled_corr)
        if corr_cfg.get('stride', 1) > 1:
            self.corr_up = Upsample(
                scale_factor=2,
                channels=(corr_cfg.get('max_displacement') * 2 + 1)**2)

        else:
            self.corr_up = nn.Sequential()

    def forward(self,
                feat1: torch.Tensor,
                feat2: torch.Tensor,
                upflow: Optional[torch.Tensor] = None,
                multiplier: float = 1.) -> torch.Tensor:
        """Forward function for MatchingBlock.

        Args:
            feat1 (Tensor): The feature from the first image.
            feat2 (Tensor): The feature from the second image.
            upflow (Optional[Tensor], optional): The upsampled optical flow
                predicted from the last level, but for level6, there is not
                optical flow from last level. Defaults to None.
            multiplier (float, optional): The constant multiplier to scale
                upsampled flow from the last level. Defaults to 1.

        Returns:
            Tensor: The optical flow predicted from MatchingBlock.
        """

        if upflow is None:
            warp_feat = feat2
            upflow = torch.zeros_like(feat1)
        else:
            warp_feat = self.warp_op(feat2, upflow * multiplier)

        corr_feat = self.corr(feat1, warp_feat)
        corr_feat = self.corr_up(corr_feat)
        feat = self.layers(corr_feat)
        res_flow = self.pred_flow(feat)

        return upflow[:, :2, ...] + res_flow


class SubpixelBlock(BasicBlock):
    """Subpixel Block.

    Args:
        warp_cfg (dict): Config dict for building warp operator.
        last_kernel_size (int): Kernel size of the last convolution layer.
    """

    def __init__(self, *args, warp_cfg: dict, last_kernel_size: int) -> None:

        super().__init__(*args)

        self.warp_op = build_operators(warp_cfg)
        self.pred_flow = nn.Conv2d(
            self.feat_out_channels,
            2,
            kernel_size=last_kernel_size,
            stride=1,
            padding=last_kernel_size // 2)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                flow: torch.Tensor, multiplier: float) -> torch.Tensor:
        """Forward function for SubpixelBlock.

        Args:
            feat1 (Tensor): The feature from the first image.
            feat2 (Tensor): The feature from the second image.
            flow (Tensor): The optical flow predicted from MatchingBlock.
            multiplier (float): The constant multiplier to scale optical flow.
                Defaults to 1.

        Returns:
            Tensor:  The optical flow predicted from SubpixelBlock.
        """

        warp_feat = self.warp_op(feat2, flow * multiplier)
        feat = torch.cat((feat1, warp_feat, flow), dim=1)
        feat = self.layers(feat)
        res_flow = self.pred_flow(feat)

        return flow + res_flow


class RegularizationBlock(BasicBlock):
    """Regularization Block.

    Args:
        last_kernel_size (int, list, tuple): (List or tuple of) kernel size of
            convolution layers.
        out_channels (int): Number of output channels of convolution layers.
        warp_cfg (dict): Config dict for building warp operator.
    """

    def __init__(self, *args, last_kernel_size: Union[int, Sequence[int]],
                 out_channels: int, warp_cfg: dict) -> None:
        super().__init__(*args)

        self.warp_op = build_operators(warp_cfg)
        in_channels = self.feat_out_channels

        if isinstance(last_kernel_size, (list, tuple)):
            dist_layers = []
            for ks in last_kernel_size:
                padding = [ks_ // 2 for ks_ in ks] \
                               if isinstance(ks, (tuple, list)) else ks // 2
                dist_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=ks,
                        padding=padding,
                        stride=1))
                in_channels = out_channels
            self.dist_layer = nn.Sequential(*dist_layers)
        else:
            self.dist_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=last_kernel_size,
                padding=last_kernel_size // 2)
        self.patch_size = int(float(out_channels)**(0.5))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor,
                feat: torch.Tensor, flow: torch.Tensor,
                multiplier: float) -> torch.Tensor:
        """Forward function for RegularizationBlock.

        Args:
            img1 (Tensor): The input first image.
            img2 (Tensor): The input first image.
            feat (Tensor): The input feature for RegularizationBlock.
            flow (Tensor): The optical flow predicted from RegularizationBlock.
            multiplier (float): The constant multiplier to scale optical flow.
                Defaults to 1.

        Returns:
            Tensor: The optical flow regularized by RegularizationBlock.
        """

        B = img1.shape[0]
        warp_img2 = self.warp_op(img2, flow * multiplier)
        diff_img = torch.norm((img1 - warp_img2), dim=1, p=2, keepdim=True)
        nomean_flow = flow - flow.view(B, 2, -1).mean(2, True).view(B, 2, 1, 1)
        feat = torch.cat((diff_img, nomean_flow, feat), dim=1)
        feat = self.layers(feat)

        # feature-driven distance metric
        dist_feat = self.dist_layer(feat)
        dist_feat = -dist_feat**2
        dist_feat = F.softmax(dist_feat, dim=1)

        flow_x_unfold = F.unfold(
            input=flow[:, 0:1, ...],
            kernel_size=self.patch_size,
            padding=self.patch_size // 2).view_as((dist_feat))
        flow_y_unfold = F.unfold(
            input=flow[:, 1:2, ...],
            kernel_size=self.patch_size,
            padding=self.patch_size // 2).view_as(dist_feat)

        flow_x = torch.sum(dist_feat * flow_x_unfold, dim=1, keepdim=True)
        flow_y = torch.sum(dist_feat * flow_y_unfold, dim=1, keepdim=True)

        return torch.cat((flow_x, flow_y), dim=1)


@DECODERS.register_module()
class NetE(BaseDecoder):
    """NetE of LiteFlowNet.

    A sub-network structure of LiteFlowNet for cascaded flow inference and
    flow regularization.

    Args:
        in_channels (dict): Dict of input channels of different levels of
            convolution layers in feature layer.
            Default: dict(level2=32, level3=64, level4=96, level5=128,
            level6=192).
        corr_channels (dict): Dict of input channels of different levels of
            MatchingBlock (NetM).
            Default: dict(level2=144, level3=144, level4=49, level5=49,
            level6=49).
        sin_channels (dict): Dict of input channels of different levels of
            SubpixelBlock (NetS).
            Default: dict(level2=130, level3=130, level4=194, level5=258,
            level6=386).
        rin_channels (dict): Dict of input channels of different levels of
            RegularizationBlock (NetR).
            Default: dict(level2=131, level3=131, level4=131, level5=131,
            level6=195).
        feat_channels (int): Number of output channels of convolution layers in
            feature layer. Default: 64.
        mfeat_channels (tuple(int)): Tuple of output channels of convolution
            layers in MatchingBlock (NetM). Default: (128, 64, 32).
        sfeat_channels (tuple(int)): Tuple of output channels of convolution
            layers in SubpixelBlock (NetS). Default: (128, 64, 32).
        rfeat_channels (tuple(int)): Tuple of output channels of convolution
            layers in RegularizationBlock (NetR).
            Default: (128, 128, 64, 64, 32, 32).
        patch_size (dict): Dict of last kernel size of different levels
            in MatchingBlock and SubpixelBlock.
            Default: dict(level2=7, level3=5, level4=5, level5=3, level6=3).
        corr_cfg (dict): Config dict of correlation of different levels in
            MatchingBlock.
            Default: dict(
                level2=dict(type='Correlation', max_displacement=6),
                level3=dict(type='Correlation', max_displacement=6),
                level4=dict(type='Correlation', max_displacement=3),
                level5=dict(type='Correlation', max_displacement=3),
                level6=dict(type='Correlation', max_displacement=3)).
        warp_cfg (dict): Config dict for building warp operator.
            Default: dict(type='Warp').
        conv_cfg (dict, optional): Config dict of convolution layers in
            MatchingBlock, SubpixelBlock and RegularizationBlock.
            Default: None.
        norm_cfg (dict, optional): Config dict of normalization layers in
            MatchingBlock, SubpixelBlock and RegularizationBlock.
            Default: None.
        act_cfg (dict): Config dict of activation layers in MatchingBlock
            SubpixelBlock and RegularizationBlock.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        scaled_corr (bool):  Whether to use scaled correlation by the number
            of elements involved to calculate correlation or not.
            Default: True.
        regularized_flow (bool): Whether to use RegularizationBlock to
            regularize flow or not. Default: True.
        flow_loss: Config of loss function of optical flow. Default: None.
        extra_training_loss (bool): Whether to calculate the extra train loss
            for upsampled flow to the same resolution as images pair.
            Default to False.
        init_cfg (dict, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: Dict[str, int] = dict(
                     level2=32, level3=64, level4=96, level5=128, level6=192),
                 corr_channels: Dict[str, int] = dict(
                     level2=144, level3=144, level4=49, level5=49, level6=49),
                 sin_channels: Dict[str, int] = dict(
                     level2=130,
                     level3=130,
                     level4=194,
                     level5=258,
                     level6=386),
                 rin_channels: Dict[str, int] = dict(
                     level2=131,
                     level3=131,
                     level4=131,
                     level5=131,
                     level6=195),
                 feat_channels: int = 64,
                 mfeat_channels: Sequence[int] = (128, 64, 32),
                 sfeat_channels: Sequence[int] = (128, 64, 32),
                 rfeat_channels: Sequence[int] = (128, 128, 64, 64, 32, 32),
                 patch_size: Dict[str, int] = dict(
                     level2=7, level3=5, level4=5, level5=3, level6=3),
                 corr_cfg: Dict[str, dict] = dict(
                     level2=dict(type='Correlation', max_displacement=6),
                     level3=dict(type='Correlation', max_displacement=6),
                     level4=dict(type='Correlation', max_displacement=3),
                     level5=dict(type='Correlation', max_displacement=3),
                     level6=dict(type='Correlation', max_displacement=3)),
                 warp_cfg: dict = dict(type='Warp'),
                 flow_div: float = 20.,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 scaled_corr: bool = True,
                 regularized_flow: bool = True,
                 flow_loss: Optional[dict] = None,
                 extra_training_loss: bool = False,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.corr_channels = corr_channels
        self.sin_channels = sin_channels
        self.rin_channels = rin_channels

        self.feat_channels = feat_channels
        self.flow_levels = list(in_channels.keys())
        self.flow_levels.sort()

        self.flow_div = flow_div
        self.multiplier = dict()
        for level in self.flow_levels:
            self.multiplier[level] = self.flow_div * 2**(-int(level[-1]) + 1)

        self.start_level = self.flow_levels[-1]
        self.end_level = self.flow_levels[0]

        self.mfeat_channels = mfeat_channels
        self.sfeat_channels = sfeat_channels
        self.rfeat_channels = rfeat_channels
        self.patch_size = patch_size

        self.corr_cfg = corr_cfg
        self.warp_cfg = warp_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.scaled_corr = scaled_corr
        self.regularized_flow = regularized_flow

        if flow_loss is not None:
            self.flow_loss = build_loss(flow_loss)

        self.extra_training_loss = extra_training_loss

        self._make_layers()

    def _make_layers(self) -> None:
        """Add layers for LiteFlowNet decoder."""
        layers = []
        for level in self.flow_levels:
            in_ch = self.in_channels[level]
            corr_ch = self.corr_channels[level]
            sin_ch = self.sin_channels[level]
            rin_ch = self.rin_channels[level]
            corr_cfg = self.corr_cfg[level]
            patch_size = self.patch_size[level]

            layers.append([
                level,
                self._make_layer(level, in_ch, corr_ch, sin_ch, rin_ch,
                                 corr_cfg, patch_size)
            ])
        self.decoders = nn.ModuleDict(layers)

    def _make_layer(
        self,
        level: str,
        in_ch: int,
        corr_ch: int,
        sin_ch: int,
        rin_ch: int,
        corr_cfg: dict,
        patch_size: int,
    ) -> nn.ModuleDict:
        """Add layers at each level of LiteFlowNet decoder.

        Args:
            level (str): The level of this submodule.
            in_ch (int): The channels of input feature.
            corr_ch (int): The channels of correlation feature.
            sin_ch (int): The input channels for NetS.
            rin_ch (int): The input channels for NetR.
            corr_cfg (dict): The Config for building correlation operator.
            patch_size (int): The patch size for feature-driven local
                convolution.

        Returns:
            nn.ModuleDict: The submodule for some stage of LiteFlowNet decoder.
        """

        if in_ch < self.feat_channels:
            feat_layer = ConvModule(
                in_channels=in_ch,
                out_channels=self.feat_channels,
                kernel_size=1,
                stride=1,
                act_cfg=self.act_cfg)
        else:
            feat_layer = nn.Sequential()

        NetM = MatchingBlock(
            corr_ch,
            self.mfeat_channels,
            self.conv_cfg,
            self.norm_cfg,
            self.act_cfg,
            self.init_cfg,
            corr_cfg=corr_cfg,
            warp_cfg=self.warp_cfg,
            last_kernel_size=patch_size,
            scaled_corr=self.scaled_corr)

        NetS = SubpixelBlock(
            sin_ch,
            self.sfeat_channels,
            self.conv_cfg,
            self.norm_cfg,
            self.act_cfg,
            self.init_cfg,
            last_kernel_size=patch_size,
            warp_cfg=self.warp_cfg)
        if patch_size > 3:
            patch_kernel = [(patch_size, 1), (1, patch_size)]
        else:
            patch_kernel = patch_size

        if level != self.flow_levels[0]:
            upflow_layer = Upsample(scale_factor=2, channels=2)
        else:
            upflow_layer = nn.Sequential()

        if level != self.flow_levels[0] or self.regularized_flow:
            if in_ch < self.rfeat_channels[0]:
                rfeat_layer = ConvModule(
                    in_channels=in_ch,
                    out_channels=self.rfeat_channels[0],
                    kernel_size=1,
                    stride=1,
                    act_cfg=self.act_cfg)
            else:
                rfeat_layer = nn.Sequential()

            NetR = RegularizationBlock(
                rin_ch,
                self.rfeat_channels,
                self.conv_cfg,
                self.norm_cfg,
                self.act_cfg,
                self.init_cfg,
                last_kernel_size=patch_kernel,
                out_channels=patch_size**2,
                warp_cfg=self.warp_cfg)

            return nn.ModuleDict({
                'feat_layer': feat_layer,
                'rfeat_layer': rfeat_layer,
                'NetM': NetM,
                'NetS': NetS,
                'NetR': NetR,
                'upflow_layer': upflow_layer
            })
        else:
            return nn.ModuleDict({
                'feat_layer': feat_layer,
                'NetM': NetM,
                'NetS': NetS,
                'upflow_layer': upflow_layer
            })

    def forward(self, img1: torch.Tensor, img2: torch.Tensor,
                feat1: Dict[str, torch.Tensor],
                feat2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function for LiteFlownet Decoder.

        Args:
            img1 (Tensor): The first input image.
            img2 (Tensor): The second input image.
            feat1 (Dict[str, Tensor]): The feature pyramid from first image.
            feat2 (Dict[str, Tensor]): The feature pyramid from second image.

        Returns:
            Dict[str, Tensor]: The predicted multi-level optical flow.
        """

        flow_pred = dict()

        upflow = None
        for level in self.flow_levels[::-1]:

            _feat1 = self.decoders[level]['feat_layer'](feat1[level])
            _feat2 = self.decoders[level]['feat_layer'](feat2[level])
            h, w = _feat1.shape[2:]
            _img1, _img2 = self._scale_img(img1, h,
                                           w), self._scale_img(img2, h, w)

            flowM = self.decoders[level]['NetM'](
                _feat1,
                _feat2,
                upflow,
                self.multiplier[level],
            )
            flowS = self.decoders[level]['NetS'](
                _feat1,
                _feat2,
                flowM,
                self.multiplier[level],
            )
            if level == self.end_level and not self.regularized_flow:
                upflow = self.decoders[level]['upflow_layer'](flowS)
                flow_pred[level] = flowS
            else:

                rfeat = self.decoders[level]['rfeat_layer'](feat1[level])
                flowR = self.decoders[level]['NetR'](
                    _img1,
                    _img2,
                    rfeat,
                    flowS,
                    self.multiplier[level],
                )
                upflow = self.decoders[level]['upflow_layer'](flowR)
                flow_pred[level] = flowR

        return flow_pred

    @staticmethod
    def _scale_img(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return F.interpolate(
            img, size=(h, w), mode='bilinear', align_corners=False)

    def forward_train(
            self,
            img1: torch.Tensor,
            img2: torch.Tensor,
            feat1: Dict[str, torch.Tensor],
            feat2: Dict[str, torch.Tensor],
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            img1 (Tensor): The first input image.
            img2 (Tensor): The second input image.
            feat1 (Dict[str, Tensor]): The feature pyramid from first image.
            feat2 (Dict[str, Tensor]): The feature pyramid from second image.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of model.
        """

        H, W = img1.shape[2:]

        flow_pred = self.forward(
            img1=img1, img2=img2, feat1=feat1, feat2=feat2)

        if self.extra_training_loss:
            flow_pred['level0'] = self._scale_img(flow_pred[self.end_level], H,
                                                  W)
        return self.losses(flow_pred=flow_pred, flow_gt=flow_gt, valid=valid)

    def forward_test(
            self,
            img1: torch.Tensor,
            img2: torch.Tensor,
            feat1: Dict[str, torch.Tensor],
            feat2: Dict[str, torch.Tensor],
            img_metas: Optional[dict] = None
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model testing.

        Args:
            img1 (Tensor): The first input image.
            img2 (Tensor): The second input image.
            feat1 (Dict[str, Tensor]): The feature pyramid from first image.
            feat2 (Dict[str, Tensor]): The feature pyramid from second image.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """

        H, W = img1.shape[2:]

        flow_pred = self.forward(
            img1=img1, img2=img2, feat1=feat1, feat2=feat2)

        flow_result = flow_pred[self.end_level]
        # flow to the size of images after augmentation.
        flow_result = F.interpolate(
            flow_result, size=(H, W), mode='bilinear', align_corners=False)

        # unravel batch dim, reshape [2, H, W] to [H, W, 2], and resize
        flow_result = flow_result.permute(0, 2, 3,
                                          1).cpu().data.numpy() * self.flow_div

        flow_result = list(flow_result)
        flow_result = [dict(flow=f) for f in flow_result]

        return self.get_flow(flow_result, img_metas=img_metas)

    def losses(self,
               flow_pred: Dict[str, torch.Tensor],
               flow_gt: torch.Tensor,
               valid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
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
