# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Union

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule
from mmcv.utils import print_log
from torch import Tensor, device

from mmflow.core.utils import (OptConfigType, OptMultiConfig, SampleList,
                               stack_batch)
from mmflow.utils.logger import get_root_logger


class FlowEstimator(BaseModule, metaclass=ABCMeta):
    """Base class for flow estimator.

    Args:
    preprocess_cfg (dict or ConfigDict, optional):
            Model preprocessing config for processing the input data.
            it usually includes ``to_rgb``, ``mean`` and ``std`` for input
            images normization and ``sigma_range`` and ``clamp_range`` for
            adding Gaussian noise on images. Defaults to None.
        freeze_net (bool): Whether freeze the weights of model. If set True,
            the model will not update the weights.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Defaults to None.
    """

    def __init__(self,
                 preprocess_cfg: OptConfigType = None,
                 freeze_net: bool = False,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg)
        self.preprocess_cfg = preprocess_cfg
        self.status = dict(iter=0, max_iters=0)
        self.freeze_net = freeze_net
        # if set freeze_net True, the weights in this model
        # will be not updated and predict the flow maps.
        if self.freeze_net:
            logger = get_root_logger()
            print_log(
                f'Freeze the parameters in {self.__class__.__name__}',
                logger=logger)
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.preprocess_cfg is not None:
            assert isinstance(self.preprocess_cfg, dict)
            self.preprocess_cfg = copy.deepcopy(self.preprocess_cfg)

            self.to_rgb = preprocess_cfg.get('to_rgb', False)
            self.register_buffer(
                'pixel_mean',
                torch.tensor(preprocess_cfg['mean']).view(-1, 1, 1), False)
            self.register_buffer(
                'pixel_std',
                torch.tensor(preprocess_cfg['std']).view(-1, 1, 1), False)
            if preprocess_cfg.get('sigma_range', None) is not None:
                self.register_buffer(
                    'sigma_range', torch.tensor(preprocess_cfg['sigma_range']),
                    False)
                self.register_buffer(
                    'clamp_range', torch.tensor(preprocess_cfg['clamp_range']),
                    False)
        else:
            # Only used to provide device information
            warnings.warn('We treat `model.preprocess_cfg` is None.')
            self.register_buffer('pixel_mean', torch.tensor(1), False)

    @property
    def device(self) -> device:
        """Get the current device."""
        return self.pixel_mean.device

    @abstractmethod
    def forward_train(self, batch_inputs: Tensor,
                      batch_data_samples: SampleList, **kwargs) -> None:
        """Placeholder for forward function of flow estimator when training."""
        pass

    @abstractmethod
    def forward_test(self, batch_inputs: Tensor,
                     batch_data_samples: SampleList, **kwargs):
        """Placeholder for forward function of flow estimator when testing."""
        pass

    def forward(self,
                data: List[dict],
                return_loss=False,
                **kwargs) -> Union[dict, SampleList]:
        """The iteration step during training and testing. This method defines
        an iteration step during training and testing, except for the back
        propagation and optimizer updating during training, which are done in
        an optimizer hook.

        Args:
            data (list[dict]): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer`, dict, Optional): The
                optimizer of runner. This argument is unused and reserved.
                Default to None.
            test_mode (bool): Whether to return loss. In general,
                it will be set to ``False`` during training and ``True``
                during testing. Default to False.

        Returns:
            during training
                dict: It should contain at least 3 keys: ``loss``,
                ``log_vars``, ``num_samples``.
                    - ``loss`` is a tensor for back propagation, which can be a
                      weighted sum of multiple losses.
                    - ``log_vars`` contains all the variables to be sent to the
                        logger.
                    - ``num_samples`` indicates the batch size (when the model
                        is DDP, it means the batch size on each GPU), which is
                        used for averaging the logs.

            during testing
                list(obj:`FlowDataSample`): Detection results of the
                input images. Each DetDataSample usually contains
                ``pred_flow_fw`` or ``pred_flow_bw`` or
                ``pred_occ_fw`` or ``pred_occ_bw``.
        """
        batch_inputs, batch_data_samples = self.preprocess_data(data)

        if return_loss:
            losses = self.forward_train(batch_inputs, batch_data_samples,
                                        **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data))

            return outputs
        else:
            return self.forward_test(batch_inputs, batch_data_samples,
                                     **kwargs)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, test_mode=False)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data, test_mode=True)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

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

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def preprocess_data(self, data: List[dict]) -> tuple:
        """Process input data during training and simple testing phases.

        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple: It should contain 2 item.

                 - batch_inputs (Tensor): The batch input tensor.
                 - batch_data_samples (list[:obj:`FlowDataSample`]): The Data
                     Samples. It usually includes information such as
                     `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        """
        # img1s is list of tensor with shape 3,H,W
        img1s = [data_['inputs'][0, ...] for data_ in data]
        img2s = [data_['inputs'][1, ...] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        batch_data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        img1s = [_input.to(self.device) for _input in img1s]
        img2s = [_input.to(self.device) for _input in img2s]

        if self.preprocess_cfg is None:
            img1s, img2s = stack_batch(img1s, img2s)
            # concatenate image as channel dim
            return torch.cat((img1s, img2s), dim=1).float(), batch_data_samples

        # Normalize images
        if self.to_rgb:
            img1s = [img1[::-1, ...] for img1 in img1s]
            img2s = [img2[::-1, ...] for img2 in img2s]
        img1s = [(img1 - self.pixel_mean) / self.pixel_std for img1 in img1s]
        img2s = [(img2 - self.pixel_mean) / self.pixel_std for img2 in img2s]
        new_img1s = []
        new_img2s = []
        # Add Noise
        for img1, img2 in zip(img1s, img2s):
            if hasattr(self, 'sigma_range'):
                # create new sigma for each image pair
                sigma = torch.tensor(
                    random.uniform(*self.sigma_range), device=self.device)
                img1 = torch.clamp(
                    img1 + torch.randn_like(img1) * sigma,
                    min=self.clamp_range[0],
                    max=self.clamp_range[1])
                img2 = torch.clamp(
                    img2 + torch.randn_like(img2) * sigma,
                    min=self.clamp_range[0],
                    max=self.clamp_range[1])
            new_img1s.append(img1)
            new_img2s.append(img2)

        new_img1s = torch.stack(new_img1s, dim=0)
        new_img2s = torch.stack(new_img2s, dim=0)

        return torch.cat((new_img1s, new_img2s), dim=1), batch_data_samples
