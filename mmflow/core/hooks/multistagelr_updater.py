# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence

from mmcv.runner import HOOKS, IterBasedRunner, LrUpdaterHook


@HOOKS.register_module()
class MultiStageLrUpdaterHook(LrUpdaterHook):
    """Multi-Stage Learning Rate Hook.

    Args:
        milestone_lrs (Sequence[float]): The base LR for multi-stages.
        milestone_iters (Sequence[int]): The first iterations in different
            stages.
        steps (Sequence[Sequence[int]]): The steps to decay the LR in stages.
        gammas (Sequence[float]): The list of decay LR ratios.
        kwargs (any): The arguments of LrUpdaterHook.
    """

    def __init__(self, milestone_lrs: Sequence[float],
                 milestone_iters: Sequence[int],
                 steps: Sequence[Sequence[int]], gammas: Sequence[float],
                 **kwargs: Any) -> None:

        assert len(milestone_lrs) == len(milestone_iters) == len(steps) == len(
            gammas
        ), ('For MultiStageLr, lengths of milestones_lr and steps and gammas',
            f'must be equal, but got {len(milestone_lrs)}, ',
            f'{len(milestone_iters)}, {len(steps)}, and {len(gammas)}')
        for i in range(len(milestone_iters)):
            assert milestone_iters[i] < steps[i][0], (
                'miltestone step must be, '
                'less than step')

        self.milestone_lrs = milestone_lrs
        self.steps = steps
        self.gammas = gammas
        self.milestone_iters = milestone_iters

        super().__init__(**kwargs)

    def get_lr(self, runner: IterBasedRunner, base_lr: float) -> float:
        """Get current LR.

        Args:
            runner (IterBasedRunner): The runner to control the training
                workflow.
            base_lr (float): The base LR in training workflow.

        Returns:
            float: The current LR.
        """
        progress = runner.epoch if self.by_epoch else runner.iter

        if progress < self.milestone_iters[0]:
            return base_lr

        milestone = -1
        for i, milestone_iter in enumerate(self.milestone_iters[1:]):
            if progress < milestone_iter:
                milestone = i
                break

        exp = len(self.steps[milestone])
        for ii, s in enumerate(self.steps[milestone]):
            if progress < s:
                exp = ii
                break
        lr = self.milestone_lrs[milestone] * (self.gammas[milestone]**exp)

        return lr
