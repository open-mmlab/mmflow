# Copyright (c) OpenMMLab. All rights reserved.

from mmflow.registry import MODELS
from .raft import RAFT


@MODELS.register_module()
class Flow1D(RAFT):
    """Flow1D model.

    Args:
        radius (int): Number of radius in  .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        cxt_encoder (dict): Config dict for building context encoder.
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(num_levels=4, **kwargs)
