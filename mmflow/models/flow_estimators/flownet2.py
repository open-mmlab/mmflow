# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
from numpy import ndarray

from ..builder import FLOW_ESTIMATORS, build_flow_estimator
from ..utils import BasicLink
from .base import FlowEstimator


@FLOW_ESTIMATORS.register_module()
class FlowNetCSS(FlowEstimator):
    """FlowNet2CSS model.

    Args:
        flownetC (dict): The config for FlownetC estimator.
        flownetS1 (dict): The first config for FlownetS estimator.
        flownetS2 (dict, optional): The second config for FlownetS estimator.
        link_cfg (dict): The config dict of link used to connect the
            estimators. Default to dict(scale_factor=4, mode='bilinear').
        flow_div (float): The divisor used to scale down ground truth.
            Defaults to 20.
        out_level (str): The level of output flow. Default to 'level2'.
        init_cfg (dict, list, optional): Config of dict weights initialization.
            Default: None.
    """

    def __init__(self,
                 flownetC: dict,
                 flownetS1: dict,
                 flownetS2: Optional[dict] = None,
                 link_cfg: dict = dict(scale_factor=4, mode='bilinear'),
                 flow_div: float = 20.,
                 out_level: str = 'level2',
                 init_cfg: Optional[Union[list, dict]] = None,
                 **kwargs) -> None:

        super().__init__(init_cfg=init_cfg, **kwargs)

        self.flownetC = build_flow_estimator(flownetC)
        self.flownetS1 = build_flow_estimator(flownetS1)
        self.link = BasicLink(**link_cfg)
        self.flow_div = flow_div

        if flownetS2 is not None:
            self.flownetS2 = build_flow_estimator(flownetS2)

        self.out_level = out_level

    def forward_train(
            self,
            imgs: torch.Tensor,
            flow_gt: Optional[torch.Tensor] = None,
            valid: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward function for Flownet2CSS when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        flowc = self.flownetC.decoder(
            *self.flownetC.extract_feat(imgs), )[self.out_level]

        img_channels = imgs.shape[1] // 2
        img1 = imgs[:, :img_channels, ...]
        img2 = imgs[:, img_channels:, ...]

        link_output1 = self.link(img1, img2, flowc, self.flow_div)

        concat1 = torch.cat((
            img1,
            img2,
            link_output1.warped_img2,
            link_output1.upsample_flow,
            link_output1.brightness_err,
        ),
                            dim=1)

        # Train FlowNetCS before FlowNetCSS
        if hasattr(self, 'flownetS2'):
            flows1 = self.flownetS1.decoder(
                self.flownetS1.encoder(concat1))[self.out_level]

            link_output2 = self.link(img1, img2, flows1, self.flow_div)
            concat2 = torch.cat((
                img1,
                img2,
                link_output2.warped_img2,
                link_output2.upsample_flow,
                link_output2.brightness_err,
            ),
                                dim=1)
            # when flownetS2 does not have flow_loss, loss if the multi-levels
            # flow_pred
            loss = self.flownetS2(
                concat2, flow_gt=flow_gt, valid=valid, test_mode=False)
        else:
            loss = self.flownetS1(
                concat1, flow_gt=flow_gt, valid=valid, test_mode=False)

        return loss

    def forward_test(
            self,
            imgs: torch.Tensor,
            img_metas: Optional[Sequence[dict]] = None) -> Sequence[ndarray]:
        """Forward function for Flownet2CSS when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        flowc = self.flownetC.decoder(
            *self.flownetC.extract_feat(imgs), )[self.out_level]

        img_channels = imgs.shape[1] // 2
        img1 = imgs[:, :img_channels, ...]
        img2 = imgs[:, img_channels:, ...]

        link_output1 = self.link(img1, img2, flowc, self.flow_div)

        concat1 = torch.cat((
            img1,
            img2,
            link_output1.warped_img2,
            link_output1.upsample_flow,
            link_output1.brightness_err,
        ),
                            dim=1)

        # Train FlowNetCS before FlowNetCSS
        if hasattr(self, 'flownetS2'):
            flows1 = self.flownetS1.decoder(
                self.flownetS1.encoder(concat1))[self.out_level]

            link_output2 = self.link(img1, img2, flows1, self.flow_div)
            concat2 = torch.cat((
                img1,
                img2,
                link_output2.warped_img2,
                link_output2.upsample_flow,
                link_output2.brightness_err,
            ),
                                dim=1)
            flow_result = self.flownetS2(
                concat2, img_metas=img_metas, test_mode=True)
        else:
            flow_result = self.flownetS1(
                concat1, img_metas=img_metas, test_mode=True)

        return flow_result

    def _forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward function for Flownet2CSS.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Dict[str, Tensor]: The predicted multi-level optical flow.
        """
        flowc = self.flownetC.decoder(
            *self.flownetC.extract_feat(imgs), )[self.out_level]

        img_channels = imgs.shape[1] // 2
        img1 = imgs[:, :img_channels, ...]
        img2 = imgs[:, img_channels:, ...]

        link_output1 = self.link(img1, img2, flowc, self.flow_div)

        concat1 = torch.cat((
            img1,
            img2,
            link_output1.warped_img2,
            link_output1.upsample_flow,
            link_output1.brightness_err,
        ),
                            dim=1)

        flows1 = self.flownetS1.decoder(
            self.flownetS1.encoder(concat1))[self.out_level]

        link_output2 = self.link(img1, img2, flows1, self.flow_div)
        concat2 = torch.cat((
            img1,
            img2,
            link_output2.warped_img2,
            link_output2.upsample_flow,
            link_output2.brightness_err,
        ),
                            dim=1)
        return self.flownetS2.decoder(self.flownetS2.encoder(concat2))


@FLOW_ESTIMATORS.register_module()
class FlowNet2(FlowEstimator):
    """FlowNet2 model.

    Args:
        flownetCSS (dict): The config of FlowNet2CSS estimator.
        flownetSD (dict): The config of FlowNet2SD estimator.
        flownet_fusion (dict): The config of fusion flownet.
        link_cfg (dict): The config dict of link used to connect the
            estimators. Defaults to dict(scale_factor=4, mode='nearest').
        flow_div (float): The divisor used to scale down ground truth.
            Defaults to 20.
        out_level (str): The level of output flow. Default to 'level2'.
        init_cfg (dict, list, optional): Config of dict weights initialization.
            Default: None.
    """

    def __init__(self,
                 flownetCSS: dict,
                 flownetSD: dict,
                 flownet_fusion: dict,
                 link_cfg: dict = dict(scale_factor=4, mode='nearest'),
                 flow_div: float = 20.,
                 out_level: str = 'level2',
                 init_cfg: Optional[Union[list, dict]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.flownetCSS = build_flow_estimator(flownetCSS)
        self.flownetSD = build_flow_estimator(flownetSD)
        self.flownet_fusion = build_flow_estimator(flownet_fusion)
        self.link = BasicLink(**link_cfg)
        self.flow_div = flow_div

        self.out_level = out_level

    def forward_train(
            self,
            imgs: torch.Tensor,
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward function for FlowNet2 when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """
        img_channels = imgs.shape[1] // 2
        img1 = imgs[:, :img_channels, ...]
        img2 = imgs[:, img_channels:, ...]

        flow_css = self.flownetCSS._forward(imgs)[self.out_level]
        flow_sd = self.flownetSD.decoder(
            self.flownetSD.encoder(imgs))[self.out_level]

        link_output_css = self.link(img1, img2, flow_css, self.flow_div)
        link_output_sd = self.link(img1, img2, flow_sd, self.flow_div)

        concat_feat = torch.cat(
            (img1, link_output_sd.scaled_flow, link_output_css.scaled_flow,
             link_output_sd.norm_scaled_flow, link_output_css.norm_scaled_flow,
             link_output_sd.brightness_err, link_output_css.brightness_err),
            dim=1)

        loss = self.flownet_fusion.forward_train(
            concat_feat, flow_gt=flow_gt, valid=valid)

        return loss

    def forward_test(
            self,
            imgs: torch.Tensor,
            img_metas: Optional[Sequence[dict]] = None) -> Sequence[ndarray]:
        """Forward function for FlowNet2 when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        img_channels = imgs.shape[1] // 2
        img1 = imgs[:, :img_channels, ...]
        img2 = imgs[:, img_channels:, ...]

        flow_css = self.flownetCSS._forward(imgs)[self.out_level]
        flow_sd = self.flownetSD.decoder(
            self.flownetSD.encoder(imgs))[self.out_level]

        link_output_css = self.link(img1, img2, flow_css, self.flow_div)
        link_output_sd = self.link(img1, img2, flow_sd, self.flow_div)

        concat_feat = torch.cat(
            (img1, link_output_sd.scaled_flow, link_output_css.scaled_flow,
             link_output_sd.norm_scaled_flow, link_output_css.norm_scaled_flow,
             link_output_sd.brightness_err, link_output_css.brightness_err),
            dim=1)

        return self.flownet_fusion.forward_test(
            concat_feat, img_metas=img_metas)
