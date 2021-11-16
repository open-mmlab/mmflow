# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import Correlation
from mmcv.utils import Registry, build_from_cfg
from torch.nn import Module

OPERATORS = Registry('operators')

OPERATORS.register_module(module=Correlation)


def build_operators(cfg: dict) -> Module:
    """build opterator with config dict.

    Args:
        cfg (dict): The config dict of operator.

    Returns:
        Module: The built operator.
    """
    return build_from_cfg(cfg, OPERATORS)
