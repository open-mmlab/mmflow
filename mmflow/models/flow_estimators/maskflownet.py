# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Tuple

import torch
from numpy import ndarray
from torch import Tensor

from mmflow.core.utils import SampleList, TensorDict
from mmflow.registry import MODELS
from ..builder import build_flow_estimator
from ..decoders.maskflownet_decoder import Upsample
from ..utils import Warp
from .pwcnet import PWCNet


def centralize(img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
    """Centralize input images.

    Args:
        img1 (Tensor): The first input image.
        img2 (Tensor): The second input image.

    Returns:
        Tuple[Tensor, Tensor]: The first centralized image and the second
            centralized image.
    """
    rgb_mean = torch.cat((img1, img2), 2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


@MODELS.register_module()
class MaskFlowNetS(PWCNet):
    """MaskFlowNetS model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_feat(self, imgs: Tensor) -> Sequence[TensorDict]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[TensorDict, TensorDict]: The feature pyramid of
                the first input image and the feature pyramid of secode input
                image.
        """
        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        img1, img2, _ = centralize(img1, img2)
        return self.encoder(img1), self.encoder(img2)


@MODELS.register_module()
class MaskFlowNet(MaskFlowNetS):
    """MaskFlowNet model."""

    def __init__(self,
                 *args,
                 maskflownetS: dict,
                 out_level: str = 'level2',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.maskflownetS = build_flow_estimator(maskflownetS)
        self.out_level = out_level
        self.flow_div = self.decoder.flow_div

    def extract_feat(self, imgs: Tensor) -> Sequence[TensorDict]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[TensorDict, TensorDict, TensorDict,
                TensorDict, TensorDict]: The feature pyramid of
                the first input image and the feature pyramid of secode input
                image in stage1 and stage2 of MaskFlownet, and estimated
                multi-level flow from the stage1.
        """
        in_channels = self.maskflownetS.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        img1, img2, _ = centralize(img1, img2)

        feat1, feat2 = self.maskflownetS.extract_feat(imgs)
        flows_stage1, mask_stage1 = self.maskflownetS.decoder(
            feat1, feat2, return_mask=True)

        img1 = torch.cat((img1, torch.zeros_like(mask_stage1)), dim=1)
        warped_img2 = Warp(align_corners=True)(
            img2, self.flow_div * Upsample(flows_stage1[self.out_level], 4))
        img2 = torch.cat((warped_img2, mask_stage1), dim=1)

        return feat1, feat2, self.encoder(img1), self.encoder(
            img2), flows_stage1

    def forward_train(self, imgs: Tensor,
                      batch_data_samples: SampleList) -> TensorDict:
        """Forward function for PWCNet when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            TensorDict: The losses of output.
        """
        feat1, feat2, feat3, feat4, flows_stage1 = self.extract_feat(imgs)
        return self.decoder.forward_train(
            feat1=feat1,
            feat2=feat2,
            feat3=feat3,
            feat4=feat4,
            batch_data_samples=batch_data_samples)

    def forward_test(
            self, imgs: Tensor,
            batch_data_samples: SampleList) -> Sequence[Dict[str, ndarray]]:
        """Forward function for PWCNet when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """

        H, W = imgs.shape[2:]
        feat1, feat2, feat3, feat4, flows_stage1 = self.extract_feat(imgs)
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return self.decoder.forward_test(feat1, feat2, feat3, feat4,
                                         flows_stage1, batch_img_metas)
