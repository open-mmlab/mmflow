# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Union

import mmcv
from mmcv.runner import Hook, IterBasedRunner, get_dist_info
from torch.utils.data import DataLoader

from .evaluation import (multi_gpu_online_evaluation,
                         single_gpu_online_evaluation)


class EvalHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        dataset_name (str, list, optional): The name of the dataset this
            evaluation hook will doing in.
        eval_kwargs (any): Evaluation arguments fed into the evaluate
            function of the dataset.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 interval: int = 1,
                 by_epoch: bool = False,
                 dataset_name: Optional[Union[str, Sequence[str]]] = None,
                 **eval_kwargs: Any) -> None:
        if not (isinstance(dataloader, DataLoader)
                or mmcv.is_list_of(dataloader, DataLoader)):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader if isinstance(
            dataloader,
            (tuple, list),
        ) else [dataloader]
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.dataset_name = dataset_name if isinstance(
            dataset_name, (tuple, list)) else [dataset_name]
        assert len(self.dataloader) == len(self.dataset_name)

    def after_train_iter(self, runner: IterBasedRunner) -> None:
        """After train iteration."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        runner.log_buffer.clear()
        self.evaluate(runner)

    def after_train_epoch(self, runner: IterBasedRunner) -> None:
        """After train epoch."""
        if not self.every_n_epochs(runner, self.interval):
            return

        self.evaluate(runner)

    def evaluate(self, runner: IterBasedRunner) -> None:
        """Evaluation function to call online evaluate function."""
        for i_dataset, i_dataloader in zip(self.dataset_name, self.dataloader):
            results_metrics = single_gpu_online_evaluation(
                runner.model, i_dataloader, **self.eval_kwargs)
            for name, val in results_metrics.items():
                if i_dataset is not None:
                    key = f'{name} in {i_dataset}'
                else:
                    key = name
                runner.log_buffer.output[key] = val
            runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        dataset_name (str, list, optional): The name of the dataset this
            evaluation hook will doing in.
        eval_kwargs (any): Evaluation arguments fed into the evaluate
            function of the dataset.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 interval: int = 1,
                 tmpdir: Optional[str] = None,
                 gpu_collect: bool = False,
                 by_epoch: bool = False,
                 dataset_name: Optional[Union[str, Sequence[str]]] = None,
                 **eval_kwargs: Any) -> None:
        if not (isinstance(dataloader, DataLoader)
                or mmcv.is_list_of(dataloader, DataLoader)):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')

        self.by_epoch = by_epoch
        self.dataloader = dataloader if isinstance(
            dataloader,
            (tuple, list),
        ) else [dataloader]
        self.interval = interval
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs
        self.dataset_name = dataset_name if isinstance(
            dataset_name, (tuple, list)) else [dataset_name]

        assert len(self.dataloader) == len(self.dataset_name)

    def evaluate(self, runner: IterBasedRunner):
        """Evaluation function to call online evaluate function."""
        for i_dataset, i_dataloader in zip(self.dataset_name, self.dataloader):
            results_metrics = multi_gpu_online_evaluation(
                runner.model, i_dataloader, **self.eval_kwargs)
            rank, _ = get_dist_info()
            if rank == 0:
                for name, val in results_metrics.items():
                    if i_dataset is not None:
                        key = f'{name} in {i_dataset}'
                    else:
                        key = name
                    runner.log_buffer.output[key] = val
                runner.log_buffer.ready = True
