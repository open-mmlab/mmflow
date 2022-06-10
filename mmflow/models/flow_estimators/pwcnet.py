# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmflow.core.utils import (OptMultiConfig, SampleList, TensorDict,
                               TensorList)
from mmflow.registry import MODELS
from ..builder import build_decoder, build_encoder
from .base import FlowEstimator


@MODELS.register_module()
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
                 init_cfg: OptMultiConfig = None,
                 **kwargs):

        super().__init__(init_cfg=init_cfg, **kwargs)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def extract_feat(self, imgs: Tensor) -> TensorDict:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: The feature pyramid of
                the first input image and the feature pyramid of second input
                image.
        """

        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        return self.encoder(img1), self.encoder(img2)

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
            Dict[str, Tensor]: The losses of output.
        """
        feat1, feat2 = self.extract_feat(imgs)
        return self.decoder.forward_train(
            feat1=feat1, feat2=feat2, batch_data_samples=batch_data_samples)

    def forward_test(self, imgs: Tensor,
                     batch_data_samples: SampleList) -> TensorList:
        """Forward function for PWCNet when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """

        feat1, feat2 = self.extract_feat(imgs)
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return self.decoder.forward_test(feat1, feat2, batch_img_metas)
