# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model
from .test import multi_gpu_test, single_gpu_test
from .train import init_random_seed, set_random_seed, train_model

__all__ = [
    'set_random_seed', 'train_model', 'init_model', 'inference_model',
    'multi_gpu_test', 'single_gpu_test', 'init_random_seed'
]
