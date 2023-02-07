# Copyright (c) OpenMMLab. All rights reserved.
from .flow_inferencer import FlowInferencer
from .inference import inference_model, init_model

__all__ = ['init_model', 'inference_model', 'FlowInferencer']
