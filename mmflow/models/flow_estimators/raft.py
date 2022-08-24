# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmflow.registry import MODELS
from mmflow.utils import OptSampleList, SampleList, TensorDict, TensorList
from ..builder import build_encoder
from .pwcnet import PWCNet


@MODELS.register_module()
class RAFT(PWCNet):
    """RAFT model.

    Args:
        num_levels (int): Number of levels in .
        radius (int): Number of radius in  .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        cxt_encoder (dict): Config dict for building context encoder.
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self,
                 num_levels: int,
                 radius: int,
                 cxt_channels: int,
                 h_channels: int,
                 cxt_encoder: dict,
                 freeze_bn: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.radius = radius
        self.context = build_encoder(cxt_encoder)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels

        assert self.num_levels == self.decoder.num_levels
        assert self.radius == self.decoder.radius
        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        """Set batch normalization layer evaluation mode."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feat(self,
                     imgs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]

        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        cxt_feat = self.context(img1)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return feat1, feat2, h_feat, cxt_feat

    def loss(
        self,
        inputs: Tensor,
        data_samples: SampleList,
        flow_init: Optional[Tensor] = None,
    ) -> TensorDict:
        """Forward function for RAFT when model training.

        Args:
            inputs (Tensor): The concatenated input images.
            data_samples (list[:obj:`FlowDataSample`]): Each item contains the
                meta information of each image and corresponding annotations.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(inputs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        return self.decoder.loss(
            feat1,
            feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            data_samples=data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None,
                 flow_init=None) -> TensorList:
        """_summary_

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
        Returns:
            TensorList: The list of tensor.
        """
        feat1, feat2, h_feat, cxt_feat = self.extract_feat(inputs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        return self.decoder(
            feat1, feat2, flow=flow_init, h_feat=h_feat, cxt_feat=cxt_feat)

    def predict(self,
                imgs: Tensor,
                data_samples: OptSampleList = None,
                flow_init: Optional[Tensor] = None) -> SampleList:
        """Forward function for RAFT when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.

        Returns:
            Sequence[FlowDataSample]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """
        train_iter = self.decoder.iters
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.decoder.iters = self.test_cfg.get('iters')

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        results = self.decoder.predict(
            feat1=feat1,
            feat2=feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            data_samples=data_samples)
        # recover iter in train
        self.decoder.iters = train_iter

        return results
