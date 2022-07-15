# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmcv.utils import Config
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
        return self.encoder(imgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Forward function for FlowNetS when model training.

        Args:
            batch_inputs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            TensorDict: The losses of output.
        """

        return self.decoder.loss(
            self.extract_feat(batch_inputs),
            batch_data_samples=batch_data_samples)

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Forward function for FlowNetS when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return self.decoder.predict(
            self.extract_feat(batch_inputs), batch_img_metas=batch_img_metas)

    def _forward(self,
                 batch_inputs: Tensor,
                 data_samples: OptSampleList = None) -> TensorDict:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is batch_inputs[N, :3, H, W] and img2 is
                batch_inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            batch_data_samples (list[:obj:`FlowDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_flow_fw``, ``gt_flow_bw``, ``gt_occ_fw`` and
                ``gt_occ_bw``. Default to None.
        Returns:
            Dict[str, :obj:`FlowDataSample`]: The predicted optical flow
            from level6 to level2.
        """
        return self.decoder(self.extract_feat(batch_inputs))


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
