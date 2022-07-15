# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence

from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim.scheduler.momentum_scheduler import MomentumSchedulerMixin
from mmengine.optim.scheduler.param_scheduler import INF, _ParamScheduler
from torch.optim import Optimizer

from mmflow.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class MultiStageParamScheduler(_ParamScheduler):
    """Adjust the parameter value of each parameter group by multistage
    scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        milestone_params (Sequence[float]): The base params for multi-stages.
            For optical flow tasks, we usually set param_name to ``lr``,
            then the initial learning rate of every stage will be assigned
            by the element in milestone_params.
        milestone_iters (Sequence[int]): The first iterations in different
            stages.
        gammas (Sequence[float]): The list of decay param ratios.
        steps (Sequence[Sequence[int]]): The steps to decay the params
            in stages.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Optical flow works often use iter-based schedulers.
            Defaults to False.
        kwargs (any): The arguments of _ParamScheduler.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 milestone_params: Sequence[float],
                 milestone_iters: Sequence[int],
                 gammas: Sequence[float],
                 steps: Sequence[Sequence[int]],
                 by_epoch: bool = False,
                 **kwargs: Any):
        assert len(milestone_params) == len(milestone_iters) == len(
            steps
        ) == len(gammas), (
            'For MultiStageLr, lengths of milestones_lr and steps and gammas',
            f'must be equal, but got {len(milestone_params)}, ',
            f'{len(milestone_iters)}, {len(steps)}, and {len(gammas)}')

        for i in range(len(milestone_iters)):
            assert milestone_iters[i] < steps[i][0], (
                'miltestone iter must be less than corresponding step')

        self.milestone_params = milestone_params
        self.steps = steps
        self.gammas = gammas
        self.milestone_iters = milestone_iters

        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            by_epoch=by_epoch,
            **kwargs)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              milestone_iters,
                              steps,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        milestone_iters = [i * epoch_length for i in milestone_iters]
        steps = [[j * epoch_length for j in i] for i in steps]
        begin = begin * epoch_length
        if end != INF:
            end = end * epoch_length
        return cls(
            *args,
            milestone_iters=milestone_iters,
            steps=steps,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using non-chainable form of the scheduler."""
        if self.last_step < self.milestone_iters[0]:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        milestone = -1
        for i, milestone_iter in enumerate(self.milestone_iters[1:]):
            if self.last_step < milestone_iter:
                milestone = i
                break

        exp = len(self.steps[milestone])
        for ii, s in enumerate(self.steps[milestone]):
            if self.last_step < s:
                exp = ii
                break

        param = self.milestone_params[milestone] * (
            self.gammas[milestone]**exp)

        return [param for _ in self.optimizer.param_groups]


@PARAM_SCHEDULERS.register_module()
class MultiStageLR(LRSchedulerMixin, MultiStageParamScheduler):
    """Adjust the learning rate of each parameter group by multistage
    scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestone_params (Sequence[float]): The base lr for multi-stages.
            The initial learning rate of every stage will be assigned
            by the element in milestone_params.
        milestone_iters (Sequence[int]): The first iterations in different
            stages.
        gammas (Sequence[float]): The list of decay lr ratios.
        steps (Sequence[Sequence[int]]): The steps to decay the lrs in stages.
        by_epoch (bool): Whether the scheduled lrs are updated by
            epochs. Optical flow works often use iter-based schedulers.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class MultiStageMomentum(MomentumSchedulerMixin, MultiStageParamScheduler):
    """Adjust the momentum value of each parameter group by multistage
    scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestone_params (Sequence[float]): The base momentum for multi-stages.
            The initial momentum of every stage will be assigned
            by the element in milestone_params.
        milestone_iters (Sequence[int]): The first iterations in different
            stages.
        gammas (Sequence[float]): The list of decay momentum ratios.
        steps (Sequence[Sequence[int]]): The steps to decay the momentum
            in stages.
        by_epoch (bool): Whether the scheduled momentum are updated by
            epochs. Optical flow works often use iter-based schedulers.
            Defaults to False.
    """
