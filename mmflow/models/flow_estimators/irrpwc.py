# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from mmflow.core.utils import SampleList, TensorDict, TensorList
from mmflow.registry import MODELS
from .pwcnet import PWCNet


@MODELS.register_module()
class IRRPWC(PWCNet):
    """IRR-PWC model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_feat(self, imgs: Tensor) -> Tuple[TensorDict, TensorDict]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: The feature pyramid of
                the first input image and the feature pyramid of seconds input
                image.
        """
        in_channels = self.encoder.in_channels
        img1 = imgs[:, :in_channels, ...]
        img2 = imgs[:, in_channels:, ...]
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        feat1['level0'] = img1
        feat2['level0'] = img2
        return feat1, feat2

    def forward_train(self, imgs: Tensor,
                      batch_data_samples: SampleList) -> TensorDict:
        """Forward function for IRR-PWC when model training.

        Args:
            imgs (Tensor): The concatenated input images.


        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        feat1, feat2 = self.extract_feat(imgs)
        return self.decoder.forward_train(
            feat1, feat2, batch_data_samples=batch_data_samples)

    def forward_test(self, imgs: Tensor,
                     batch_data_samples: SampleList) -> TensorList:
        """Forward function for IRR-PWC when model testing.

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

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the IRRPWC network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """

        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss_flow = log_vars['loss_flow'].detach()

        if log_vars.get('loss_occ') is not None:
            loss_occ = log_vars['loss_occ'].detach()

            loss_flow_weight = 1.
            loss_occ_weight = 1.

            if loss_flow.data > loss_occ.data and loss_occ.data != 0.:
                loss_flow_weight = 1.
                loss_occ_weight = loss_flow / loss_occ

            elif loss_flow.data < loss_occ.data and loss_flow.data != 0.:
                loss_flow_weight = loss_occ / loss_flow
                loss_occ_weight = 1.

            loss = log_vars['loss_flow'] * loss_flow_weight + log_vars[
                'loss_occ'] * loss_occ_weight
        else:
            loss = log_vars['loss_flow']

        log_vars['loss'] = loss

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
