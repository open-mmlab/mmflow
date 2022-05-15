# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import Correlation
from torch.nn import Module

from mmflow.registry import MODELS

ENCODERS = MODELS
DECODERS = MODELS
FLOW_ESTIMATORS = MODELS
LOSSES = MODELS
COMPONENTS = MODELS
OPERATORS = MODELS

OPERATORS.register_module(module=Correlation)


def build_operators(cfg: dict) -> Module:
    """build opterator with config dict.

    Args:
        cfg (dict): The config dict of operator.

    Returns:
        Module: The built operator.
    """
    return OPERATORS.build(cfg)


def build_encoder(cfg: dict) -> Module:
    """Build encoder for flow estimator.

    Args:
        cfg (dict): Config for encoder.

    Returns:
        Module: Encoder module.
    """
    return ENCODERS.build(cfg)


def build_decoder(cfg: dict) -> Module:
    """Build decoder for flow estimator.

    Args:
        cfg (dict): Config for decoder.

    Returns:
        Module: Decoder module.
    """
    return DECODERS.build(cfg)


def build_components(cfg: dict) -> Module:
    """Build encoder for model component.

    Args:
        cfg (dict): Config for component of model.

    Returns:
        Module: Component of model.
    """
    return COMPONENTS.build(cfg)


def build_loss(cfg: dict) -> Module:
    """Build loss function.

    Args:
        cfg (dict): Config for loss function.

    Returns:
        Module: Loss function.
    """
    return LOSSES.build(cfg)


def build_flow_estimator(cfg: dict) -> Module:
    """Build flow estimator.

    Args:
        cfg (dict): Config for optical flow estimator.

    Returns:
        Module: Flow estimator.
    """
    return FLOW_ESTIMATORS.build(cfg)
