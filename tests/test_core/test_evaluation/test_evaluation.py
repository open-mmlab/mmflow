# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from mmflow.core.evaluation import online_evaluation


def _build_demo_model():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)
            self.status = dict()

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    return model


@patch('mmflow.core.eval_metrics', MagicMock)
def test_online_evaluation():
    # create dummy data
    dataloader = MagicMock()
    metric = 'EPE'
    model = _build_demo_model()

    # test mock result cannot compute EPE
    with pytest.raises(KeyError):
        online_evaluation(model, data_loader=dataloader, metric=metric)
