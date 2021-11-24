# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from ..builder import FLOW_ESTIMATORS, build_decoder
from .base import FlowEstimator


@FLOW_ESTIMATORS.register_module()
class SpyNet(FlowEstimator):

    def __init__(self,
                 pyramid_levels,
                 decoder,
                 img_channels=3,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.pyramid_levels = pyramid_levels
        self.pyramid_levels.sort()
        self.img_channels = img_channels
        self.decoder = build_decoder(decoder)

    def downsample_images(self, imgs):
        imgs1 = dict()
        imgs2 = dict()
        img1 = imgs[:, :self.img_channels, ...]
        img2 = imgs[:, self.img_channels:, ...]

        imgs1[self.pyramid_levels[0]] = img1
        imgs2[self.pyramid_levels[0]] = img2

        for level in self.pyramid_levels[1:]:

            img1 = F.avg_pool2d(
                img1,
                kernel_size=2,
                stride=2,
            )
            img2 = F.avg_pool2d(
                img2,
                kernel_size=2,
                stride=2,
            )
            imgs1[level] = img1
            imgs2[level] = img2

        return imgs1, imgs2

    def forward_train(self, imgs, flow_gt, valid=None, img_meta=None):
        imgs1, imgs2 = self.downsample_images(imgs)

        return self.decoder.forward_train(
            imgs1=imgs1, imgs2=imgs2, flow_gt=flow_gt, valid=valid)

    def forward_test(self, imgs, img_metas=None):
        imgs1, imgs2 = self.downsample_images(imgs)

        return self.decoder.forward_test(
            imgs1=imgs1, imgs2=imgs2, img_metas=img_metas)
