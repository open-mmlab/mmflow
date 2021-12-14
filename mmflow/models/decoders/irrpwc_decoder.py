# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmflow.models.decoders.base_decoder import BaseDecoder
from mmflow.ops import build_operators
from ..builder import DECODERS, build_components, build_loss
from ..utils import BasicDenseBlock, CorrBlock


class IRRCorrBlock(BaseModule):
    """The correlation block of IRRPWC Net.

    Args:
        in_channels (int): Number of input channels of convolution layers.
        out_channels (int): Number of output channels of convolution layers.
        corr_cfg (dict): Config for correlation layer.
        scaled (bool): Whether to use scaled correlation by the
            number of elements involved to calculate correlation or not.
        warp_cfg(dict): Config for warp operation.
        act_cfg (dict, optional): Config dict for each activation layer in
            ConvModule.
        init_cfg (dict, list, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 corr_cfg: dict,
                 scaled: bool,
                 warp_cfg: dict,
                 act_cfg: dict,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(init_cfg)

        if in_channels == out_channels:
            self.conv_1x1 = nn.Sequential()
        else:
            self.conv_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                act_cfg=act_cfg)

        self.corr = CorrBlock(
            corr_cfg=corr_cfg, act_cfg=act_cfg, scaled=scaled)

        self.warp = build_operators(warp_cfg)

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        flow_f: Optional[torch.Tensor] = None,
        flow_b: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function for IRRCorrBlock.

        Args:
            feat1 (Tensor): The feature from image1.
            feat2 (Tensor): The feature from image2.
            flow_f (Optional[Tensor], optional): The optical flow from image1
                to image2, but for level6 there is not predicted flow.
                Defaults to None.
            flow_b (Optional[Tensor], optional): The optical flow from image2
                to image1, but for level6 there is not predicted flow.
                Defaults to None.

        Returns:
            Sequence[Tensor]: correlation between feature1 and warped feature2,
                feature1 after 1x1 convolution layer to make the number of
                channel equal, correlation between feature2 and warped
                feature1, and feature2 after 1x1 convolution layer.
        """

        if flow_f is None and flow_b is None:
            feat1_warp = feat1
            feat2_warp = feat2
        else:
            assert flow_f is not None and flow_b is not None

            feat1_warp = self.warp(feat1, flow_b)
            feat2_warp = self.warp(feat2, flow_f)

        corr_f = self.corr(feat1, feat2_warp)
        corr_b = self.corr(feat2, feat1_warp)

        feat1 = self.conv_1x1(feat1)
        feat2 = self.conv_1x1(feat2)

        return corr_f, feat1, corr_b, feat2


class IRRFlowDecoder(BasicDenseBlock):
    """The decoder works for estimating flow map in IRRPWC.

    Args:
        in_channels (int): Number of input channels per level.
        feat_channels (Sequence[int]): Output channels of convolution module
            in dense layers. Default: (128, 128, 96, 64, 32).
        conv_cfg (dict): Config of convolution layer in module.
        norm_cfg (dict): Config of norm layer in module.
        act_cfg (dict): Config of activation layer in module.
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: Sequence[int],
                 conv_cfg: dict,
                 norm_cfg: dict,
                 act_cfg: dict,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            in_channels,
            feat_channels=feat_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.last_channels = self.in_channels + sum(self.feat_channels)
        self.predict_layer = nn.Conv2d(
            self.last_channels, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for IRRFlowDecoder.

        Args:
            x (Tensor): The input feature.

        Returns:
            Tensor: The predicted optical flow.
        """
        feat = self.layers(x)
        return feat, self.predict_layer(feat)


class IRROccDecoder(BasicDenseBlock):
    """The decoder works for estimating occlusion map in IRRPWC.

    Args:
        in_channels (int): Number of input channels per level.
        feat_channels (Sequence[int]): Output channels of convolution module
            in dense layers. Default: (128, 128, 96, 64, 32).
        conv_cfg (dict): Config of convolution layer in module.
        norm_cfg (dict): Config of norm layer in module.
        act_cfg (dict): Config of activation layer in module.
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: Sequence[int],
                 conv_cfg: dict,
                 norm_cfg: dict,
                 act_cfg: dict,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            in_channels,
            feat_channels=feat_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.last_channels = self.in_channels + sum(self.feat_channels)
        self.predict_layer = nn.Conv2d(
            self.last_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for IRROccDecoder.

        Args:
            x (Tensor): The input feature.

        Returns:
            Tensor: The predicted occlusion mask.
        """
        feat = self.layers(x)
        return feat, self.predict_layer(feat)


@DECODERS.register_module()
class IRRPWCDecoder(BaseDecoder):
    """The decoder module of IRRPWC.

    Args:
        flow_levels (Sequence[str]): The list of output. For multi-levels
            outputs, the flow_levels indicates which levels will be
            predicted.
        corr_in_channels (dict): Dict of input channels of different
            levels of feature pyramid. In IRR-PWC, for reusing correlation
            block for different levels, it needs the same input channels,
            corr_in_channels works for conv1x1 which makes the input
            channels be equal.
        corr_feat_channels (int): Number of output feature channels for
            conv1x1 in corr_block.
        flow_decoder_in_channels (int): Number of input channels that will
            be feed in the flow decoder.
        occ_decoder_in_channels (int): Number of input channels that will
            be feed in the occlusion decoder.
        corr_cfg (dict): Config for correlation layer.
            Defaults to dict(type='Correlation', max_displacement=4).
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Defaults to True.
        warp_cfg (dict): Config for warp operation. Defaults to
            dict(type='Warp', align_corners=True) that are same to the official
            implementation of IRRPWC.
        densefeat_channels (Sequence[int]): Number of output channels for
            dense layers. Defaults to (128, 128, 96, 64, 32).
        flow_post_processor (dict, optional): Config of flow post process
            module. Default: None
        flow_refine (dict, optional): Config of flow refine module.
            Defaults to None.
        occ_post_processor (dict, optional): Config of occlusion post
            process module. Default: None
        occ_refine (dict, optional): Config of occlusion refine module.
            Defaults to None.
        occ_refined_levels (Sequence[str]): List of levels of occlusion
            output that will be refine by `OccShuffleUpsample` module.
            Defaults to ['level0', 'level1'].
        occ_upsample (dict, optional): Config of `OccShuffleUpsample`
            module. Defaults to None.
        flow_div (float): The divisor works for scaling the ground truth.
            Default: 20.
        upsample_cfg (dict): Config dict of interpolate in PyTorch.
            Default: dict(mode='bilinear', align_corners=True)
        conv_cfg (dict, optional): Config dict of convolution layer in
            module. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in module.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in
            module. Default: dict(type='LeakyReLU', negative_slope=0.1).
        flow_loss: Config of loss function of optical flow. Default: None.
        occ_loss: Config of loss function of occlusion mask. Default: None.
        init_cfg (dict, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 flow_levels: Sequence[str],
                 corr_in_channels: Dict[str, int],
                 corr_feat_channels: int,
                 flow_decoder_in_channels: int,
                 occ_decoder_in_channels: int,
                 corr_cfg: dict = dict(type='Correlation', max_displacement=4),
                 scaled: bool = True,
                 warp_cfg: dict = dict(type='Warp', align_corners=True),
                 densefeat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 flow_post_processor: dict = None,
                 flow_refine: dict = None,
                 occ_post_processor: dict = None,
                 occ_refine: dict = None,
                 occ_refined_levels: Sequence[str] = ['level0', 'level1'],
                 occ_upsample: dict = None,
                 flow_div: float = 20.,
                 upsample_cfg: dict = dict(
                     mode='bilinear', align_corners=True),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 flow_loss: Optional[dict] = None,
                 occ_loss: Optional[dict] = None,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.flow_levels = flow_levels
        self.flow_levels.sort()
        self.start_level = self.flow_levels[-1]
        self.end_level = self.flow_levels[0]

        if flow_loss is not None:
            self.flow_loss = build_loss(flow_loss)
        if occ_loss is not None:
            self.occ_loss = build_loss(occ_loss)

        self._make_corr_block(
            corr_in_channels=corr_in_channels,
            corr_feat_channels=corr_feat_channels,
            corr_cfg=corr_cfg,
            act_cfg=act_cfg,
            scaled=scaled,
            warp_cfg=warp_cfg)

        self.flow_decoders = IRRFlowDecoder(
            in_channels=flow_decoder_in_channels,
            feat_channels=densefeat_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.occ_decoders = IRROccDecoder(
            in_channels=occ_decoder_in_channels,
            feat_channels=densefeat_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.upsample_cfg = upsample_cfg

        self.flow_refine = build_components(flow_refine)
        self.flow_post_processor = build_components(flow_post_processor)

        self.occ_refined_levels = occ_refined_levels
        self.occ_post_processor = build_components(occ_post_processor)
        self.occ_refine = build_components(occ_refine)
        self.occ_shuffle_upsample = build_components(occ_upsample)

        self.flow_div = flow_div

    def _make_corr_block(self, corr_in_channels: Dict[str, int],
                         corr_feat_channels: int, corr_cfg: dict,
                         act_cfg: dict, scaled: bool, warp_cfg: dict) -> None:
        """Make correlation block.

        Args:
            corr_in_channels (dict): Dict of input channels of different
                levels of feature pyramid. In IRR-PWC, for reusing correlation
                block for different levels, it needs the same input channels,
                corr_in_channels works for conv1x1 which makes the input
                channels be equal.
            corr_feat_channels (int): Number of output feature channels for
                conv1x1 in corr_block.
            corr_cfg (dict): Config for correlation layer.
            act_cfg (dict, optional): Config dict of activation layer in
                module.
            scaled (bool): Whether to use scaled correlation by the number of
                elements involved to calculate correlation or not.
                Defaults to True.
            warp_cfg (dict): Config for warp operation.
        """

        layers = []
        for level, channels in corr_in_channels.items():
            layers.append([
                level,
                IRRCorrBlock(
                    in_channels=channels,
                    out_channels=corr_feat_channels,
                    corr_cfg=corr_cfg,
                    scaled=scaled,
                    warp_cfg=warp_cfg,
                    act_cfg=act_cfg,
                )
            ])
        self.corr_block = nn.ModuleDict(layers)

    def forward(
        self,
        feat1: Dict[str, torch.Tensor],
        feat2: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, Dict[str, List[torch.Tensor]]], Dict[str, Dict[
            str, List[torch.Tensor]]]]:
        """Forward function for IRR-PWC decoder.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
        Returns:
            Tuple[Dict[str, Dict[str, List[Tensor]]], Dict[str, Dict[str,
                List[Tensor]]]] : The predicted multi-level optical flow and
                the predicted multi-level occlusion mask.
        """

        img1 = feat1['level0']
        img2 = feat2['level0']

        B, _, H_img, W_img = img1.shape

        # flow_preds is the output of optical flow, includes keys:
        #   - flow_fw
        #       - levelx: [Tensor]
        #   - flow_bw
        #       - levelx: [Tensor]

        # occ_preds is the output of occlusion, includes keys:
        #   - flow_fw
        #       - levelx: [Tensor]
        #   - flow_bw
        #       - levelx: [Tensor]

        flow_preds = dict(flow_fw=dict(), flow_bw=dict())
        occ_preds = dict(occ_fw=dict(), occ_bw=dict())

        for level in self.flow_levels[::-1]:

            _feat1, _feat2 = feat1.get(level), feat2.get(level)

            B, _, h, w = _feat1.shape

            if level == self.start_level:

                flow_fw = torch.zeros((B, 2, h, w),
                                      requires_grad=False).to(_feat1)
                flow_bw = torch.zeros((B, 2, h, w),
                                      requires_grad=False).to(_feat1)
                occ_fw = torch.zeros((B, 1, h, w),
                                     requires_grad=False).to(_feat1)
                occ_bw = torch.zeros((B, 1, h, w),
                                     requires_grad=False).to(_feat1)

            else:

                flow_fw = self._scale_flow(flow_fw, h, w)
                flow_bw = self._scale_flow(flow_bw, h, w)
                if level not in self.occ_refined_levels:
                    occ_fw = self._scale_img(occ_fw, h, w)
                    occ_bw = self._scale_img(occ_bw, h, w)

            if level in self.occ_refined_levels:

                flow_preds['flow_fw'][level] = [
                    self._scale_flow_as_gt(flow_fw, H_img=H_img, W_img=W_img)
                ]
                flow_preds['flow_bw'][level] = [
                    self._scale_flow_as_gt(flow_bw, H_img=H_img, W_img=W_img)
                ]

                occ_fw = F.interpolate(occ_fw, scale_factor=2, mode='nearest')
                occ_bw = F.interpolate(occ_bw, scale_factor=2, mode='nearest')

                occ_fw = self.occ_shuffle_upsample(
                    occ_fw,
                    _feat1,
                    _feat2,
                    flow_fw,
                    flow_bw,
                    self.flow_div,
                    H_img,
                    W_img,
                )
                occ_bw = self.occ_shuffle_upsample(
                    occ_bw,
                    _feat2,
                    _feat1,
                    flow_bw,
                    flow_fw,
                    self.flow_div,
                    H_img,
                    W_img,
                )
                occ_preds['occ_fw'][level] = [occ_fw]
                occ_preds['occ_bw'][level] = [occ_bw]

            else:
                corr_f, feat1_1by1, corr_b, feat2_1by1 = self.corr_block[
                    level](_feat1, _feat2, flow_fw, flow_bw)

                # predict flow for each level on the original scale
                feat_flow_f, flow_res_f = self.flow_decoders(
                    torch.cat((corr_f, feat1_1by1, flow_fw), dim=1))
                feat_flow_b, flow_res_b = self.flow_decoders(
                    torch.cat((corr_b, feat2_1by1, flow_bw), dim=1))

                flow_fw = flow_res_f + flow_fw
                flow_bw = flow_res_b + flow_bw

                flow_fw = flow_fw + self.flow_post_processor(
                    torch.cat((feat_flow_f, flow_fw), dim=1))
                flow_bw = flow_bw + self.flow_post_processor(
                    torch.cat((feat_flow_b, flow_bw), dim=1))

                # predict occ
                feat_occ_f, occ_res_f = self.occ_decoders(
                    torch.cat((corr_f, feat1_1by1, occ_fw), dim=1))
                feat_occ_b, occ_res_b = self.occ_decoders(
                    torch.cat((corr_b, feat2_1by1, occ_bw), dim=1))
                occ_fw = occ_res_f + occ_fw
                occ_bw = occ_res_b + occ_bw
                occ_fw = occ_fw + self.occ_post_processor(
                    torch.cat((feat_occ_f, occ_fw), dim=1))
                occ_bw = occ_bw + self.occ_post_processor(
                    torch.cat((feat_occ_b, occ_bw), dim=1))

                # refine flow
                scale_img1 = self._scale_img(img1, h, w)
                scale_img2 = self._scale_img(img2, h, w)

                flow_refined_f = self.flow_refine(scale_img1, scale_img2,
                                                  feat1_1by1, flow_fw.detach())
                flow_refined_b = self.flow_refine(scale_img2, scale_img1,
                                                  feat2_1by1, flow_bw.detach())

                # refine occ
                occ_refined_f = self.occ_refine(feat1_1by1, feat2_1by1,
                                                occ_fw.detach(),
                                                flow_refined_f)
                occ_refined_b = self.occ_refine(feat2_1by1, feat1_1by1,
                                                occ_bw.detach(),
                                                flow_refined_b)

                # scale flow as gt scale and divided by flow_div
                # just rescale flow not reshape map
                flow_preds['flow_fw'][level] = [
                    self._scale_flow_as_gt(flow_fw, H_img, W_img),
                    self._scale_flow_as_gt(flow_refined_f, H_img, W_img)
                ]
                flow_preds['flow_bw'][level] = [
                    self._scale_flow_as_gt(flow_bw, H_img, W_img),
                    self._scale_flow_as_gt(flow_refined_b, H_img, W_img)
                ]
                occ_preds['occ_fw'][level] = [occ_fw, occ_refined_f]
                occ_preds['occ_bw'][level] = [occ_bw, occ_refined_b]

                flow_fw = flow_refined_f
                flow_bw = flow_refined_b
                occ_fw = occ_refined_f
                occ_bw = occ_refined_b

        return flow_preds, occ_preds

    def _scale_img(self, img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Scale image function.

        Args:
            img (Tensor): the input image.
            h (int): the height of output.
            w (int): the width of output.

        Returns:
            Tensor: The output image.
        """
        return F.interpolate(img, size=(h, w), **self.upsample_cfg)

    def _scale_flow(self, flow, h, w):
        """Scale flow function.

        Args:
            flow (Tensor): the input flow.
            h (int): the height of output.
            w (int): the width of output.

        Returns:
            Tensor: The output optical flow.
        """
        h_org, w_org = flow.shape[2:]
        scale = torch.Tensor([float(w / w_org), float(h / h_org)]).to(flow)
        flow = torch.einsum('b c h w, c -> b c h w', flow, scale)

        return F.interpolate(flow, size=(h, w), **self.upsample_cfg)

    def _scale_flow_as_gt(self, flow: torch.Tensor, H_img: int,
                          W_img: int) -> torch.Tensor:
        """Scale optical flow as ground truth.

        Args:
            flow (Tensor): the input flow.
            H_img (int): the height of input images.
            W_img (int): the width of input images.
        Returns:
            Tensor: The output optical flow.
        """
        h_org, w_org = flow.shape[2:]
        scale = torch.Tensor([float(W_img / w_org),
                              float(H_img / h_org)]).to(flow) / self.flow_div
        return torch.einsum('b c h w, c -> b c h w', flow, scale)

    def forward_train(
            self,
            feat1: Dict[str, torch.Tensor],
            feat2: Dict[str, torch.Tensor],
            flow_fw_gt: Optional[torch.Tensor] = None,
            flow_bw_gt: Optional[torch.Tensor] = None,
            occ_fw_gt: Optional[torch.Tensor] = None,
            occ_bw_gt: Optional[torch.Tensor] = None,
            flow_gt: Optional[torch.Tensor] = None,
            occ_gt: Optional[torch.Tensor] = None,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
            flow_fw_gt (Tensor, optional): The ground truth of optical flow
                from image1 to image2. Defaults to None.
            flow_bw_gt (Tensor, optional): The ground truth of optical flow
                from image2 to image1. Defaults to None.
            occ_fw_gt (Tensor, optional): The ground truth of occlusion mask
                from image1 to image2. Defaults to None.
            occ_bw_gt (Tensor, optional): The ground truth of occlusion mask
                from image2 to image1. Defaults to None.
            flow_gt (Tensor, optional): The ground truth of optical flow
                from image1 to image2. Defaults to None.
            occ_gt (Tensor, optional): The ground truth of occlusion mask from
                image1 to image2. Defaults to None.
            valid (Tensor, optional): The valid mask of optical flow ground
                truth. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        flow_preds, occ_preds = self.forward(feat1, feat2)

        flow_fw = flow_preds['flow_fw']
        flow_bw = flow_preds['flow_bw']
        occ_fw = occ_preds['occ_fw']
        occ_bw = occ_preds['occ_bw']

        losses = dict()

        if (flow_fw_gt is not None and flow_bw_gt is not None
                and occ_fw_gt is not None and occ_bw_gt is not None):

            loss_fw = self.losses(
                flow_pred=flow_fw,
                flow_gt=flow_fw_gt,
                occ_pred=occ_fw,
                occ_gt=occ_fw_gt,
                valid=valid)
            loss_bw = self.losses(
                flow_pred=flow_bw,
                flow_gt=flow_bw_gt,
                occ_pred=occ_bw,
                occ_gt=occ_bw_gt,
                valid=valid)

            losses['loss_flow'] = (loss_fw['loss_flow'] +
                                   loss_bw['loss_flow']) / 2
            losses['loss_occ'] = (loss_fw['loss_occ'] +
                                  loss_bw['loss_occ']) / 2

        elif (flow_gt is not None and occ_gt is not None):

            losses = self.losses(
                flow_pred=flow_fw,
                flow_gt=flow_gt,
                occ_pred=occ_fw,
                occ_gt=occ_gt,
                valid=valid)

            self._detach_unused_preds(flow_bw)
            self._detach_unused_preds(occ_bw)

        elif (flow_fw_gt is not None and flow_bw_gt is not None
              and occ_fw_gt is None and occ_bw_gt is None and flow_gt is None
              and occ_gt is None):

            loss_fw = self.losses(
                flow_pred=flow_fw, flow_gt=flow_fw_gt, valid=valid)
            loss_bw = self.losses(
                flow_pred=flow_bw, flow_gt=flow_bw_gt, valid=valid)
            losses['loss_flow'] = (loss_fw['loss_flow'] +
                                   loss_bw['loss_flow']) / 2

            self._detach_unused_preds(occ_fw)
            self._detach_unused_preds(occ_bw)

        elif (flow_fw_gt is None and flow_bw_gt is None and occ_fw_gt is None
              and occ_bw_gt is None and occ_gt is None
              and flow_gt is not None):
            losses = self.losses(
                flow_pred=flow_fw, flow_gt=flow_gt, valid=valid)

            self._detach_unused_preds(flow_bw)
            self._detach_unused_preds(occ_fw)
            self._detach_unused_preds(occ_bw)

        return losses

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

        flow_result = []
        flow_preds, _ = self.forward(feat1, feat2)

        flow_result_fw = flow_preds['flow_fw'][self.end_level][-1]

        flow_result_fw = F.interpolate(
            flow_result_fw, size=(H, W), mode='bilinear', align_corners=False)
        flow_result_fw = flow_result_fw.permute(
            0, 2, 3, 1).cpu().data.numpy() * self.flow_div
        # unravel batch dim
        flow_result_fw = list(flow_result_fw)

        flow_result_bw = flow_preds['flow_bw'][self.end_level][-1]

        flow_result_bw = F.interpolate(
            flow_result_bw, size=(H, W), mode='bilinear', align_corners=False)
        flow_result_bw = flow_result_bw.permute(
            0, 2, 3, 1).cpu().data.numpy() * self.flow_div

        # unravel batch dim
        flow_result_bw = list(flow_result_bw)

        # collect forward and backward flow
        flow_result = [
            dict(flow_fw=flow_fw, flow_bw=flow_bw)
            for flow_fw, flow_bw in zip(flow_result_fw, flow_result_bw)
        ]

        return self.get_flow(flow_result, img_metas=img_metas)

    def losses(self,
               flow_pred: Dict[str, Sequence[torch.Tensor]],
               flow_gt: torch.Tensor,
               occ_pred: Optional[Dict[str, Sequence[torch.Tensor]]] = None,
               occ_gt: torch.Tensor = None,
               valid: Optional[torch.Tensor] = None):
        """Compute optical flow loss and occlusion mask loss.

        Args:
            flow_pred (dict): multi-level predicted optical flow.
            flow_gt (Tensor): The ground truth of optical flow.
            occ_pred (dict): multi-level predicted occlusion mask.
            occ_gt (Tensor): The ground truth of occlusion mask.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        loss = dict()
        loss['loss_flow'] = self.flow_loss(flow_pred, flow_gt, valid)
        if (occ_pred is not None and occ_gt is not None
                and self.occ_loss is not None):
            loss['loss_occ'] = self.occ_loss(occ_pred, occ_gt)
        return loss

    @staticmethod
    def _detach_unused_preds(preds: Dict[str, torch.Tensor]) -> None:
        """Detach unused predicted output.

        As some datasets do not include ground truth of backward optical flow
        and occlusion, this function detaches the prediction that don't need
        to calculate the loss.

        Args:
            preds (Dict[str, Tensor]): The prediction without corresponding
                target.
        """
        for k in preds.keys():
            for i in range(len(preds[k])):
                preds[k][i] = preds[k][i].detach()
