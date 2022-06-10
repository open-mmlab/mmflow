# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

from numpy import ndarray
from torch import Tensor

from mmflow.core.utils import SampleList, TensorDict
from mmflow.registry import MODELS
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
                of the first input image and the feature pyramid of secode
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

    def forward_train(self, imgs: Tensor,
                      batch_data_samples: SampleList) -> TensorDict:
        """Forward function for LiteFlowNet when model training.

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

        img1, img2, feat1, feat2 = self.extract_feat(imgs)

        return self.decoder.forward_train(
            img1, img2, feat1, feat2, batch_data_samples=batch_data_samples)

    def forward_test(self, imgs: Tensor,
                     batch_data_samples: SampleList) -> Sequence[ndarray]:
        """Forward function for LiteFlowNet when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """

        img1, img2, feat1, feat2 = self.extract_feat(imgs)
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return self.decoder.forward_test(img1, img2, feat1, feat2,
                                         batch_img_metas)
