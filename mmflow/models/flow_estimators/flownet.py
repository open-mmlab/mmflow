# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Tuple, Union

from mmcv.utils import Config
from numpy import ndarray
from torch import Tensor

from mmflow.core.utils import SampleList, TensorDict
from mmflow.registry import MODELS
from ..builder import build_encoder
from .pwcnet import PWCNet


@MODELS.register_module()
class FlowNetS(PWCNet):
    """FlowNetS flow estimator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward_train(self, imgs: Tensor,
                      batch_data_samples: SampleList) -> TensorDict:
        """Forward function for FlowNetS when model training.

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

        feat = self.encoder(imgs)

        return self.decoder.forward_train(
            feat,
            batch_data_samples=batch_data_samples,
            return_multi_level_flow=self.freeze_net)

    def forward_test(
            self, imgs: Tensor,
            batch_data_samples: SampleList) -> Sequence[Dict[str, ndarray]]:
        """Forward function for FlowNetS when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        H, W = imgs.shape[2:]
        feat = self.encoder(imgs)
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return self.decoder.forward_test(
            feat,
            batch_img_metas=batch_img_metas,
            return_multi_level_flow=self.freeze_net)


@MODELS.register_module()
class FlowNetC(PWCNet):
    """FlowNetC flow estimator.

    Args:
        corr_level (str): The level to calculate the correlation.
        corr_encoder (Config): The config of correaltion encoder.
    """

    def __init__(self, corr_level: str, corr_encoder: Config, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.corr_level = corr_level
        self.corr_encoder = build_encoder(corr_encoder)

    def extract_feat(self, imgs: Tensor) -> Tuple[TensorDict, TensorDict]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[TensorDict, TensorDict]: The feature pyramid
                from the first image and the feature pyramid from feature
                correlation.
        """

        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        return feat1, self.corr_encoder(feat1[self.corr_level],
                                        feat2[self.corr_level])

    def forward_train(self, imgs: Tensor,
                      batch_data_samples: SampleList) -> TensorDict:
        """Forward function for FlowNetC when model training.

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

        feat1, corr_feat = self.extract_feat(imgs)

        return self.decoder.forward_train(
            feat1,
            corr_feat,
            batch_data_samples=batch_data_samples,
            return_multi_level_flow=self.freeze_net)

    def forward_test(
        self, imgs: Tensor, batch_data_samples: SampleList
    ) -> Union[TensorDict, Sequence[ndarray]]:
        """Forward function for FlowNetC when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """

        feat1, corr_feat = self.extract_feat(imgs)
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return self.decoder.forward_test(
            feat1,
            corr_feat,
            batch_img_metas=batch_img_metas,
            return_multi_level_flow=self.freeze_net)
