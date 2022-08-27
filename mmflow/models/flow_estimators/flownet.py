# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmengine.config import Config
from torch import Tensor

from mmflow.registry import MODELS
from mmflow.utils import OptSampleList, SampleList, TensorDict
from ..builder import build_encoder
from .pwcnet import PWCNet


@MODELS.register_module()
class FlowNetS(PWCNet):
    """FlowNetS flow estimator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def extract_feat(self, imgs: Tensor) -> TensorDict:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            TensorDict: The feature pyramid extracted from the concatenated
                input images.
        """
        return self.encoder(imgs)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Forward function for FlowNetS when model training.

        Args:
            inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is inputs[N, :3, H, W] and img2 is
                inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            data_samples (list[:obj:`FlowDataSample`]): Each item contains the
                meta information of each image and corresponding annotations.

        Returns:
            TensorDict: The losses of output.
        """

        return self.decoder.loss(self.extract_feat(inputs), data_samples)

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Forward function for FlowNetS when model testing.

        Args:
            inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is inputs[N, :3, H, W] and img2 is
                inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.

        Returns:
            Sequence[FlowDataSample]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """
        return self.decoder.predict(self.extract_feat(inputs), data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> TensorDict:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is batch_inputs[N, :3, H, W] and img2 is
                batch_inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.
        Returns:
            Dict[str, :obj:`FlowDataSample`]: The predicted optical flow
            from level6 to level2.
        """
        return self.decoder(self.extract_feat(inputs))


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
            Tuple[TensorDict, TensorDict]: The feature pyramid from the first
                image and the feature pyramid from feature correlation.
        """

        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        return feat1, self.corr_encoder(feat1[self.corr_level],
                                        feat2[self.corr_level])
