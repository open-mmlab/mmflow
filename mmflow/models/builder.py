# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch.nn as nn
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry, build_from_cfg
from torch.nn import Module

MODELS = Registry('models', parent=MMCV_MODELS)
ENCODERS = MODELS
DECODERS = MODELS
FLOW_ESTIMATORS = MODELS
LOSSES = MODELS
COMPONENTS = MODELS


def build(cfg: Union[Sequence[dict], dict],
          registry: Registry,
          default_args: Optional[dict] = None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_encoder(cfg: dict) -> Module:
    """Build encoder for flow estimator.

    Args:
        cfg (dict): Config for encoder.

    Returns:
        Module: Encoder module.
    """
    return build(cfg, ENCODERS)


def build_decoder(cfg: dict) -> Module:
    """Build decoder for flow estimator.

    Args:
        cfg (dict): Config for decoder.

    Returns:
        Module: Decoder module.
    """
    return build(cfg, DECODERS)


def build_components(cfg: dict) -> Module:
    """Build encoder for model component.

    Args:
        cfg (dict): Config for component of model.

    Returns:
        Module: Component of model.
    """
    return build(cfg, COMPONENTS)


def build_loss(cfg: dict) -> Module:
    """Build loss function.

    Args:
        cfg (dict): Config for loss function.

    Returns:
        Module: Loss function.
    """
    return build(cfg, LOSSES)


def build_flow_estimator(cfg: dict) -> Module:
    """Build flow estimator.

    Args:
        cfg (dict): Config for optical flow estimator.

    Returns:
        Module: Flow estimator.
    """
    return build(cfg, FLOW_ESTIMATORS)
