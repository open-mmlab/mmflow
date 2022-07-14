# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine.testing import assert_allclose

from mmflow.core.scheduler import (MultiStageLR, MultiStageMomentum,
                                   MultiStageParamScheduler)


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


def generate_targets(by_epoch=False):
    epoch_length = 11 if by_epoch else 1
    stage1_targets = [0.5] * 5 * epoch_length + [0.5**2] * 5 * epoch_length + [
        0.5**3
    ] * 5 * epoch_length
    stage2_targets = [0.3] * 5 * epoch_length + [
        0.3 * 0.1
    ] * 5 * epoch_length + [0.3 * 0.1 * 0.1] * 15 * epoch_length
    single_targets = stage1_targets + stage2_targets
    targets = [single_targets, single_targets]
    return targets


class TestMultiStageParamScheduler(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.5, momentum=0.01, weight_decay=5e-4)

    def _test_scheduler_value(self,
                              scheduler,
                              targets,
                              total_iters=40,
                              param_name='lr'):
        for iter in range(total_iters):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                # Test if the params updated by scheduler are same as targets
                assert_allclose(
                    target[iter],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, iter, target[iter],
                        param_group[param_name]),
                    atol=1e-8,
                    rtol=0)
            scheduler.step()

    def test_multistage_scheduler(self):
        # temp setting for MultiStage Parameter Scheduler
        gammas = [0.5, 0.1]
        milestone_params = [0.5, 0.3]
        milestone_iters = [0, 15]
        steps = [[5, 10], [20, 25]]

        # Test the lengths of milstone_params, milestone_iters,
        # gammas and steps are not equal
        with self.assertRaises(AssertionError):
            MultiStageParamScheduler(
                optimizer=self.optimizer,
                param_name='lr',
                milestone_params=milestone_params,
                milestone_iters=[0, 15, 20],
                gammas=gammas,
                steps=steps,
                by_epoch=False)

        # Test miltestone iter is larger than corresponding step
        with self.assertRaises(AssertionError):
            MultiStageParamScheduler(
                optimizer=self.optimizer,
                param_name='lr',
                milestone_params=milestone_params,
                milestone_iters=[0, 30],
                gammas=gammas,
                steps=steps,
                by_epoch=False)

        # Test MultiStageParamScheduler
        targets = generate_targets()
        total_iters = 40
        end_iters = 30
        scheduler = MultiStageParamScheduler(
            optimizer=self.optimizer,
            param_name='lr',
            milestone_params=milestone_params,
            milestone_iters=milestone_iters,
            gammas=gammas,
            steps=steps,
            by_epoch=False,
            end=end_iters)
        self._test_scheduler_value(scheduler, targets, total_iters)

    def test_multistage_scheduler_convert_iterbased(self):
        gammas = [0.5, 0.1]
        milestone_params = [0.5, 0.3]
        milestone_iters = [0, 15]
        steps = [[5, 10], [20, 25]]

        total_epochs = 40
        end_epochs = 30
        epoch_length = 11

        # Test by_epoch is wrongly set to False
        with self.assertRaises(AssertionError):
            MultiStageParamScheduler.build_iter_from_epoch(
                optimizer=self.optimizer,
                param_name='lr',
                milestone_params=milestone_params,
                milestone_iters=milestone_iters,
                gammas=gammas,
                steps=steps,
                by_epoch=False,
                end=end_epochs,
                epoch_length=epoch_length)

        # Test epoch_length is wrongly set to None
        with self.assertRaises(AssertionError):
            MultiStageParamScheduler.build_iter_from_epoch(
                optimizer=self.optimizer,
                param_name='lr',
                milestone_params=milestone_params,
                milestone_iters=milestone_iters,
                gammas=gammas,
                steps=steps,
                by_epoch=True,
                end=end_epochs)

        # Test the build_iter_from_epoch function of MultiStageParamScheduler
        targets = generate_targets(by_epoch=True)
        scheduler = MultiStageParamScheduler.build_iter_from_epoch(
            optimizer=self.optimizer,
            param_name='lr',
            milestone_params=milestone_params,
            milestone_iters=milestone_iters,
            gammas=gammas,
            steps=steps,
            by_epoch=True,
            end=end_epochs,
            epoch_length=epoch_length)
        self._test_scheduler_value(scheduler, targets,
                                   total_epochs * epoch_length)

    def test_multistage_lr(self):
        gammas = [0.5, 0.1]
        milestone_params = [0.5, 0.3]
        milestone_iters = [0, 15]
        steps = [[5, 10], [20, 25]]

        # Test MultiStageLR
        targets = generate_targets()
        total_iters = 40
        end_iters = 30
        scheduler = MultiStageLR(
            optimizer=self.optimizer,
            milestone_params=milestone_params,
            milestone_iters=milestone_iters,
            gammas=gammas,
            steps=steps,
            by_epoch=False,
            end=end_iters)
        self._test_scheduler_value(scheduler, targets, total_iters)

    def test_multistage_momentum(self):
        gammas = [0.5, 0.1]
        milestone_params = [0.5, 0.3]
        milestone_iters = [0, 15]
        steps = [[5, 10], [20, 25]]

        # Test MultiStageMomentum
        targets = generate_targets()
        total_iters = 40
        end_iters = 30
        scheduler = MultiStageMomentum(
            optimizer=self.optimizer,
            milestone_params=milestone_params,
            milestone_iters=milestone_iters,
            gammas=gammas,
            steps=steps,
            by_epoch=False,
            end=end_iters)
        self._test_scheduler_value(
            scheduler, targets, total_iters, param_name='momentum')
