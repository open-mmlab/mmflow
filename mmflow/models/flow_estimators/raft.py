# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from numpy import ndarray

from ..builder import FLOW_ESTIMATORS, build_encoder
from .pwcnet import PWCNet


@FLOW_ESTIMATORS.register_module()
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
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feat(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward_train(
            self,
            imgs: torch.Tensor,
            flow_gt: torch.Tensor,
            valid: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward function for RAFT when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        return self.decoder.forward_train(
            feat1,
            feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            flow_gt=flow_gt,
            valid=valid)

    def forward_test(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None) -> Sequence[ndarray]:
        """Forward function for RAFT when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        train_iter = self.decoder.iters
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.decoder.iters = self.test_cfg.get('iters')

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        results = self.decoder.forward_test(
            feat1=feat1,
            feat2=feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            img_metas=img_metas)
        # recover iter in train
        self.decoder.iters = train_iter

        return results
