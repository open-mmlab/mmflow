# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmflow."""
from typing import Dict, Optional, Sequence, Union

import torch
from mmengine.config import ConfigDict

from mmflow.structures import FlowDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, Sequence[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

SampleList = Sequence[FlowDataSample]
OptSampleList = Optional[SampleList]

# Type hint of Tensor
TensorDict = Dict[str, torch.Tensor]
TensorList = Sequence[torch.Tensor]
