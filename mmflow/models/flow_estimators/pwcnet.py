# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

from numpy import ndarray
from torch import Tensor

from ..builder import FLOW_ESTIMATORS, build_decoder, build_encoder
from .base import FlowEstimator


@FLOW_ESTIMATORS.register_module()
class PWCNet(FlowEstimator):
    """PWC-Net model.

    Args:
        encoder (dict): The config of encoder.
        decoder (dict): The config of decoder.
        init_cfg (list, dict, optional): Config of dict weights initialization.
            Default: None.
    """

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 init_cfg: Optional[Union[dict, list]] = None,
                 **kwargs):

        super().__init__(init_cfg=init_cfg, **kwargs)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def extract_feat(self, imgs: Tensor) -> Dict[str, Tensor]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: The feature pyramid of
                the first input image and the feature pyramid of secode input
                image.
        """

        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        return self.encoder(img1), self.encoder(img2)

    def forward_train(
            self,
            imgs: Tensor,
            flow_gt: Tensor,
            valid: Optional[Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None) -> Dict[str, Tensor]:
        """Forward function for PWCNet when model training.

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
        feat1, feat2 = self.extract_feat(imgs)
        return self.decoder.forward_train(
            feat1=feat1, feat2=feat2, flow_gt=flow_gt, valid=valid)

    def forward_test(
            self,
            imgs: Tensor,
            img_metas: Optional[Sequence[dict]] = None) -> Sequence[ndarray]:
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
        feat1, feat2 = self.extract_feat(imgs)
        return self.decoder.forward_test(feat1, feat2, H, W, img_metas)
