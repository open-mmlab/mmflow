# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmflow.ops import build_operators
from ..builder import COMPONENTS


@COMPONENTS.register_module()
class FlowRefine(BaseModule):
    """Bilateral refinement module for flow in IRR.

    Use a feature-driven local convolution to regularize flow for smoothing
    flow field.

    Args:
        in_channels (int): Number of input channels.
        feat_channels (Sequence[int]): List of numbers of outputs feature
            channels. Default: (128, 128, 96, 64, 64, 32, 32).
        patch_size (int): The size of regularization filter that works for
            feature-driven local convolution. Default: 3.
        warp_cfg (dict): Config for warp Operation. Default: dict(type='Warp').
        conv_cfg (dict , optional): Config for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config for activation layer in ConvModule.
            sDefault: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: Sequence[int] = (128, 128, 96, 64, 64, 32, 32),
                 patch_size: int = 3,
                 warp_cfg: dict = dict(type='Warp', align_corners=True),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.patch_size = patch_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.warp_op = build_operators(warp_cfg)

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
        layers.append(
            ConvModule(
                in_channels=self.feat_out_channels,
                # patch_size × patch_size regularization filter
                out_channels=patch_size * patch_size,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        feat: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function for IRR-PWC.

        Args:
            img1 (Tensor): The first input image1.
            img2 (Tensor): The second input image2.
            feat (Tensor): The input feature from 1x1 convolution layer in
                Correlation block.
            flow (Tensor): The flow needed to be refined.

        Returns:
            Tensor: The refined optical flow.
        """
        B = img1.shape[0]

        warp_img2 = self.warp_op(img2, flow)
        diff_img = torch.norm((img1 - warp_img2), dim=1, p=2, keepdim=True)
        nomean_flow = flow - flow.view(B, 2, -1).mean(2, True).view(B, 2, 1, 1)
        feat = torch.cat((nomean_flow, diff_img, feat), dim=1)

        # shape (B, patch_size × patch_size, H, W)
        feat = self.layers(feat)
        # a feature-driven CNN distance metric normalized with softmax
        feat = F.softmax(-feat**2, dim=1)

        # flow[:, 0, ...].shape is [B, H, W]
        # flow[:, 0:1, ...].shape is [B, 1, H, W]

        # Extracts sliding local blocks with shape
        # (B, patch_size × patch_size, H, W)
        # from horizontal flow map with shape(B, 1, H, W)
        flow_x_unfold = F.unfold(
            F.pad(
                flow[:, 0:1, ...],
                pad=[self.patch_size // 2] * 4,
                mode='replicate'),
            kernel_size=self.patch_size).view_as((feat))

        flow_y_unfold = F.unfold(
            F.pad(
                flow[:, 1:2, ...],
                pad=[self.patch_size // 2] * 4,
                mode='replicate'),
            kernel_size=self.patch_size).view_as(feat)

        # local convolution
        flow_x = torch.sum(feat * flow_x_unfold, dim=1, keepdim=True)
        flow_y = torch.sum(feat * flow_y_unfold, dim=1, keepdim=True)

        return torch.cat((flow_x, flow_y), dim=1)


@COMPONENTS.register_module()
class OccRefine(FlowRefine):
    """Bilateral refinement module for occlusion in IRR.

    Args:
        in_channels (int): Number of input channels.
        feat_channels (Sequence[int]): List of numbers of outputs feature
            channels. Default: (128, 128, 96, 64, 64, 32, 32).
        patch_size (int): The size of regularization filter that works for
            feature-driven local convolution. Default: 3.
        warp_cfg (dict): Config for warp Operation. Default: dict(type='Warp').
        conv_cfg (dict , optional): Config for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config for activation layer in ConvModule.
            sDefault: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: Sequence[int] = (128, 128, 96, 64, 64, 32, 32),
                 patch_size: int = 3,
                 warp_cfg=dict(type='Warp'),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            patch_size=patch_size,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            warp_cfg=warp_cfg,
            init_cfg=init_cfg)

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        occ: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of OccRefine.

        Args:
            feat1 (Tensor): Input feature extracted from the first image.
            feat2 (Tensor): Input feature extracted from the second image.
            occ (Tensor): Current estimated occlusion mask.
            flow (Tensor): Current estimated optical flow.

        Returns:
            Tensor: The occlusion mask after refining.
        """

        warp_feat2 = self.warp_op(feat2, flow)
        diff_feat = feat1 - warp_feat2

        feat = torch.cat((occ, feat1, diff_feat), dim=1)
        feat = self.layers(feat)
        feat = F.softmax(-feat**2, dim=1)

        occ_unfold = F.unfold(
            F.pad(occ, pad=[self.patch_size // 2] * 4, mode='replicate'),
            kernel_size=self.patch_size).view_as((feat))

        occ = torch.sum(feat * occ_unfold, dim=1, keepdim=True)

        return occ


@COMPONENTS.register_module()
class OccShuffleUpsample(BaseModule):
    """Refine module for upsampled occlusion output.

    Args:
        in_channels (int): Number of input channels.
        feat_channels (int): Number of feature channels in residual block.
        infeat_channels (int): Number of input feature channels for conv1x1.
            For reusing this module for different levels, it needs a same input
            channels, and conv1x1 works for modify the input channels to 3.
        out_channels (int): Number of output channels.
        warp_cfg (dict): Config for warp Operation. Default: dict(type='Warp').
        conv_cfg (dict , optional): Config for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config for activation layer in ConvModule.
            sDefault: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 infeat_channels: int,
                 out_channels: int,
                 warp_cfg: dict = dict(type='Warp'),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.init_conv = ConvModule(
            in_channels=in_channels,
            out_channels=feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.res_conv = nn.Sequential(
            ConvModule(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None))
        self.res_end_conv = ConvModule(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.out_conv = ConvModule(
            in_channels=feat_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mul_const = 0.1
        self.warp_op = build_operators(warp_cfg)
        self.conv_1x1 = ConvModule(
            in_channels=infeat_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(
        self,
        occ: torch.Tensor,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        flow_f: torch.Tensor,
        flow_b: torch.Tensor,
        flow_div: torch.Tensor,
        H_img: torch.Tensor,
        W_img: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of OccShuffleUpsample.

        Args:
            occ (Tensor): Current estimated occlusion mask.
            feat1 (Tensor): Input feature extracted from the first image.
            feat2 (Tensor): Input feature extracted from the second image.
            flow_f (Tensor): Current estimated optical flow from the first
                image to the second image.
            flow_f (Tensor): Current estimated optical flow from the second
                image to the first image.
            flow_div (Tensor): The divisor to scale optical flow.
            H_img (Tensor): The height of input images.
            W_img (Tensor): The width of input images.

        Returns:
            Tensor: Occlusion mask after refining,
        """

        feat2_warp = self.warp_op(feat2, flow_f)

        h_org, w_org = flow_f.shape[2:]
        u_scale = float(W_img) / float(w_org)
        v_scale = float(H_img) / float(h_org)

        flow_b_ = torch.zeros_like(flow_b)
        flow_b_[:, 0, ...] = flow_b[:, 0, ...] * u_scale / flow_div
        flow_b_[:, 1, ...] = flow_b[:, 1, ...] * v_scale / flow_div

        flow_b_warp = self.warp_op(flow_b_, flow_f)

        if feat1.shape[1] > 3:
            feat1 = self.conv_1x1(feat1)
            feat2_warp = self.conv_1x1(feat2_warp)

        flow_f_ = torch.zeros_like(flow_f)
        flow_f_[:, 0, ...] = flow_f[:, 0, ...] * u_scale / flow_div
        flow_f_[:, 1, ...] = flow_f[:, 1, ...] * v_scale / flow_div

        # feed a occlusion map, a feature map from the encoder,
        # a warped feature map from the other temporal direction,
        # flow, and warped flow.
        feat = torch.cat((occ, feat1, feat2_warp, flow_f_, flow_b_warp), dim=1)
        feat_init = self.init_conv(feat)
        feat_res = feat_init
        feat_res = feat_res + self.res_conv(feat_res) * self.mul_const
        feat_res = feat_res + self.res_conv(feat_res) * self.mul_const
        feat_res = feat_res + self.res_conv(feat_res) * self.mul_const
        feat_init = feat_init + self.res_end_conv(feat_res)

        return self.out_conv(feat_init) + occ
