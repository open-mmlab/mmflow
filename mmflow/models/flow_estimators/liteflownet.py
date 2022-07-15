# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmflow.registry import MODELS
from mmflow.utils import TensorDict
from .pwcnet import PWCNet


@MODELS.register_module()
class LiteFlowNet(PWCNet):
    """LiteFlowNet model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_feat(
            self,
            imgs: Tensor) -> Tuple[Tensor, Tensor, TensorDict, TensorDict]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, TensorDict, TensorDict]: The
                first input image, the second input image, the feature pyramid
                of the first input image and the feature pyramid of second
                input image.
        """

        in_channels = self.encoder.in_channels

        # take from github.com:sniklaus/pytorch-liteflownet.git
        imgs_mean = [0.411618, 0.434631, 0.454253]

        for ich in range(in_channels):
            imgs[:, ich, :, :] = imgs[:, ich, :, :] - imgs_mean[ich]
            imgs[:, ich + in_channels, :, :] = (
                imgs[:, ich + in_channels, :, :] - imgs_mean[ich])

        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]

        return img1, img2, self.encoder(img1), self.encoder(img2)
