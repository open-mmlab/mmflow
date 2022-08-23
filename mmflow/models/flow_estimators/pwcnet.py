# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from torch import Tensor

from mmflow.registry import MODELS
from mmflow.utils import OptMultiConfig, OptSampleList, SampleList, TensorDict
from ..builder import build_decoder, build_encoder
from .base_flow_estimator import FlowEstimator


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
                 **kwargs) -> None:

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

    def loss(self, inputs: Tensor,
             data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is inputs[N, :3, H, W] and img2 is
                inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            data_samples (list[:obj:`FlowDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_flow_fw``, ``gt_flow_bw``, ``gt_occ_fw`` and
                ``gt_occ_bw``.

        Returns:
            dict: A dictionary of loss components.
        """

        return self.decoder.loss(
            *self.extract_feat(inputs),
            data_samples=data_samples)

    def predict(self, inputs: Tensor,
                data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is inputs[N, :3, H, W] and img2 is
                inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            data_samples (list[:obj:`FlowDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_flow_fw``, ``gt_flow_bw``, ``gt_occ_fw`` and
                ``gt_occ_bw``.


        Returns:
            list[:obj:`FlowDataSample`]: Optical Flow results of the
            input images. Each FlowDataSample usually contain
            ``pred_flow_fw``.
        """

        img_metas = []
        for data_sample in data_samples:
            img_metas.append(data_sample.metainfo)
        return self.decoder.predict(*self.extract_feat(inputs),
                                    img_metas)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> TensorDict:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Input images of shape (N, 6, H, W).
                img1 is inputs[N, :3, H, W] and img2 is
                inputs[N, 3:, H, W]. These should usually be mean
                centered and std scaled.
            data_samples (list[:obj:`FlowDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_flow_fw``, ``gt_flow_bw``, ``gt_occ_fw`` and
                ``gt_occ_bw``. Default to None.
        Returns:
            Dict[str, :obj:`FlowDataSample`]: The predicted optical flow
            from level6 to level2.
        """
        return self.decoder(*self.extract_feat(inputs))
