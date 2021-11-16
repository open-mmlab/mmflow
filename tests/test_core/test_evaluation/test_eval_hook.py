# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from mmcv.runner import build_runner
from torch.utils.data import DataLoader

from mmflow.core.evaluation import DistEvalHook, EvalHook


def _build_demo_runner_without_hook(runner_type='IterBasedRunner',
                                    max_iters=None,
                                    multi_optimizers=False):

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

    if multi_optimizers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.95),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.9),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_iters=max_iters))
    return runner


def _build_demo_runner(runner_type='IterBasedRunner',
                       max_iters=None,
                       multi_optimizers=False):

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_iters,
                                             multi_optimizers)

    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


@pytest.mark.parametrize('EvalHookParam', (EvalHook, DistEvalHook))
def test_evaluation_hook(EvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones(5, 2))

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 1. interval=1: perform evaluation after each iter.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=1)
    evalhook.evaluate = MagicMock()

    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after iter 1 & 2

    # 2. interval=2: perform evaluation after iter 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after iter 2
