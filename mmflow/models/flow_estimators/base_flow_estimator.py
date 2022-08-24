# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Union

from mmengine.logging import print_log
from mmengine.model import BaseModel
from torch import Tensor, device

from mmflow.utils import (OptConfigType, OptMultiConfig, OptSampleList,
                          SampleList, TensorDict)


class FlowEstimator(BaseModel, metaclass=ABCMeta):
    """Base class for flow estimator.

    Args:
        data_preprocessor (dict or ConfigDict, optional) :The pre-process
            config for processing the input data. it usually includes
            ``bgr_to_rgb``, ``rgb_to_bgr``, ``mean`` and ``std`` for input
            images normalization and ``sigma_range`` and ``clamp_range`` for
            adding Gaussian noise on images. Defaults to None.
        freeze_net (bool): Whether freeze the weights of model. If set True,
            the model will not update the weights.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 freeze_net: bool = False,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.status = dict(iter=0, max_iters=0)
        self.freeze_net = freeze_net
        # if set freeze_net True, the weights in this model
        # will be not updated and predict the flow maps.
        if self.freeze_net:
            print_log(
                f'Freeze the parameters in {self.__class__.__name__}',
                logger='current')
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def device(self) -> device:
        """Get the current device."""
        return self.pixel_mean.device

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> Union[dict, SampleList]:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`FlowDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, ...)
                in general.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`FlowDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs: Tensor,
             data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> TensorDict:
        """Network forward process.

        Usually includes encoder,and decoder forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def extract_feat(self, inputs: Tensor) -> TensorDict:
        """Extract features from images."""
        pass
