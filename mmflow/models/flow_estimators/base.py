# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Optional, Union

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule
from mmcv.utils import print_log

from mmflow.utils.logger import get_root_logger


class FlowEstimator(BaseModule, metaclass=ABCMeta):
    """Base class for flow estimator.

    Args:
        freeze_net (bool): Whether freeze the weights of model. If set True,
            the model will not update the weights.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Defaults to None.
    """

    def __init__(self,
                 freeze_net: bool = False,
                 init_cfg: Optional[Union[list, dict]] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)

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

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Placeholder for forward function of flow estimator when training."""
        pass

    @abstractmethod
    def forward_test(self, *args, **kwargs):
        """Placeholder for forward function of flow estimator when testing."""
        pass

    def forward(self, *args, test_mode=True, **kwargs):
        if not test_mode:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

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
