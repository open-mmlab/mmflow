# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Sequence

import torch
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop

from mmflow.registry import LOOPS


@LOOPS.register_module()
class MultiValLoop(BaseLoop):
    """Loop for validation multi-datasets.

    Args:
        runner (Runner): A reference of runner.
        dataloader (list): A dataloader object or a dict to
            build a dataloader.
        evaluator (list, dict, Evaluator): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: List,
                 evaluator: List,
                 fp16: bool = False) -> None:
        self._runner = runner

        assert isinstance(dataloader, list)
        self.dataloaders = list()
        for loader in dataloader:
            if isinstance(loader, dict):
                diff_rank_seed = runner._randomness_cfg.get(
                    'diff_rank_seed', False)
                self.dataloaders.append(
                    runner.build_dataloader(
                        loader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed))
            else:
                self.dataloaders.append(loader)

        assert isinstance(evaluator, list)
        self.evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        assert len(self.evaluators) == len(self.dataloaders)
        self.fp16 = fp16

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        multi_metric = dict()
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            self.evaluator = evaluator
            self.dataloader = dataloader
            if hasattr(self.dataloader.dataset, 'metainfo'):
                self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
                self.runner.visualizer.dataset_meta = \
                    self.dataloader.dataset.metainfo
            else:
                warnings.warn(
                    f'Dataset {self.dataloader.dataset.__class__.__name__} '
                    'has no metainfo. ``dataset_meta`` in evaluator, metric'
                    ' and visualizer will be None.')
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
                # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `val_evaluator`')
            else:
                multi_metric.update(metrics)
        self.runner.call_hook('after_val_epoch', metrics=multi_metric)
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx: int, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # data_samples should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            data_samples = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples, data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=data_samples)


@LOOPS.register_module()
class MultiTestLoop(BaseLoop):
    """Loop for validation multi-datasets.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: List,
                 evaluator: List,
                 fp16: bool = False) -> None:
        self._runner = runner
        assert isinstance(dataloader, list)
        self.dataloaders = list()
        for loader in dataloader:
            if isinstance(loader, dict):
                diff_rank_seed = runner._randomness_cfg.get(
                    'diff_rank_seed', False)
                self.dataloaders.append(
                    runner.build_dataloader(
                        loader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed))
            else:
                self.dataloaders.append(loader)

        assert isinstance(evaluator, list)
        self.evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        assert len(self.evaluators) == len(self.dataloaders)
        self.fp16 = fp16

    def run(self):
        """Launch test."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        multi_metric = dict()
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            self.evaluator = evaluator
            self.dataloader = dataloader
            if hasattr(self.dataloader.dataset, 'metainfo'):
                self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
                self.runner.visualizer.dataset_meta = \
                    self.dataloader.dataset.metainfo
            else:
                warnings.warn(
                    f'Dataset {self.dataloader.dataset.__class__.__name__} '
                    'has no metainfo. ``dataset_meta`` in evaluator, metric'
                    ' and visualizer will be None.')
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
                # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `test_evaluator`')
            else:

                multi_metric.update(metrics)
        self.runner.call_hook('after_test_epoch', metrics=multi_metric)
        self.runner.call_hook('after_test')

    @torch.no_grad()
    def run_iter(self, idx: int, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            data_samples = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_samples, data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=data_samples)
