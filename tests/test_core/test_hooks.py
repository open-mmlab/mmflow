# Copyright (c) OpenMMLab. All rights reserved.
import logging
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from mmcv.runner import IterTimerHook, PaviLoggerHook, build_runner
from torch.utils.data import DataLoader

from mmflow.core import MultiStageLrUpdaterHook


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimziers=False):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    if multi_optimziers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.)

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    return runner


def _build_demo_runner(runner_type='IterBasedRunner',
                       max_epochs=1,
                       max_iters=None,
                       multi_optimziers=False):

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_epochs,
                                             max_iters, multi_optimziers)

    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_multistagelr_updater_hook(multi_optimziers):
    # test lengths of  milestone_lrs, milestone_steps, steps, and gammas
    # must be equal.
    with pytest.raises(AssertionError):
        MultiStageLrUpdaterHook([0.1], [0, 6, 13], [[5], [10]], [0.2])

    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((30, 2)))

    runner = _build_demo_runner(
        multi_optimziers=multi_optimziers, max_iters=30, max_epochs=None)

    # add step LR scheduler
    hook = MultiStageLrUpdaterHook(
        by_epoch=False,
        milestone_lrs=[0.5, 0.3],
        milestone_iters=[6, 13],
        steps=[[8, 10], [15, 20, 25]],
        gammas=[0.2, 0.1])
    runner.register_hook(hook)
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 5),
            call(
                'train', {
                    'learning_rate/model1': 0.5,
                    'learning_rate/model2': 0.5,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 7),
            call(
                'train', {
                    'learning_rate/model1': 0.1,
                    'learning_rate/model2': 0.1,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 9),
            call(
                'train', {
                    'learning_rate/model1': 0.020000000000000004,
                    'learning_rate/model2': 0.020000000000000004,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 11),
            call(
                'train', {
                    'learning_rate/model1': 0.3,
                    'learning_rate/model2': 0.3,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 14),
            call(
                'train', {
                    'learning_rate/model1': 0.03,
                    'learning_rate/model2': 0.03,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 16),
            call(
                'train', {
                    'learning_rate/model1': 0.0030000000000000005,
                    'learning_rate/model2': 0.0030000000000000005,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 21),
            call(
                'train', {
                    'learning_rate/model1': 0.0003000000000000001,
                    'learning_rate/model2': 0.0003000000000000001,
                    'momentum/model1': 0.,
                    'momentum/model2': 0.,
                }, 26),
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.,
            }, 1),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.,
            }, 5),
            call('train', {
                'learning_rate': 0.5,
                'momentum': 0.,
            }, 7),
            call('train', {
                'learning_rate': 0.1,
                'momentum': 0.,
            }, 9),
            call('train', {
                'learning_rate': 0.020000000000000004,
                'momentum': 0.,
            }, 11),
            call('train', {
                'learning_rate': 0.3,
                'momentum': 0.,
            }, 14),
            call('train', {
                'learning_rate': 0.03,
                'momentum': 0.,
            }, 16),
            call('train', {
                'learning_rate': 0.0030000000000000005,
                'momentum': 0.,
            }, 21),
            call('train', {
                'learning_rate': 0.0003000000000000001,
                'momentum': 0.,
            }, 26),
        ]

    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)
