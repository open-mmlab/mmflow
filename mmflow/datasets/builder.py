# Copyright (c) OpenMMLab. All rights reserved.
import platform
from typing import Optional

from mmengine.config import Config
from torch.utils.data import Dataset

from mmflow.registry import DATASETS

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(cfg: Config, default_args: Optional[dict] = None) -> Dataset:
    """Build Pytorch dataset.

    Args:
        cfg (mmengine.Config): Config dict of dataset. It should at
            least contain the key "type".
        default_args (dict, optional): Default initialization arguments.

    Returns:
        dataset: The built dataset based on the input config.
    """
    dataset = DATASETS.build(cfg, default_args=default_args)

    return dataset
