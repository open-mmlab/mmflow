# Copyright (c) OpenMMLab. All rights reserved.
import platform
import random
from functools import partial
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader, Dataset

from .samplers import DistributedSampler, MixedBatchDistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg: Union[mmcv.Config, Sequence[mmcv.Config]],
                  default_args: Optional[dict] = None) -> Dataset:
    """Build Pytorch dataset.

    Args:
        cfg (mmcv.Config): Config dict of dataset or list of config dict.
            It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.

    .. note::
        If the input config is a list, this function will concatenate them
        automatically.

    Returns:
        dataset: The built dataset based on the input config.
    """
    from .dataset_wrappers import ConcatDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     sample_ratio: Optional[Sequence] = None,
                     num_gpus: int = 1,
                     dist: bool = True,
                     shuffle: bool = True,
                     seed: Optional[int] = None,
                     persistent_workers: bool = False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        sample_ratio (list, optional): The ratio for samples in mixed branch,
            sum of sample_ratio must be equal to 1. and the length must be
            equal to the length of datasets, e.g branch=8,
            sample_ratio=(0.5,0.25,0.25) means in one branch 4 samples from
            dataset1, 2 samples from dataset2 and 2 samples from dataset3.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, optional): the seed for generating random numbers for data
            workers. Default to None.
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu

        if sample_ratio is None:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
            shuffle = False
        else:
            from .dataset_wrappers import ConcatDataset
            sampler = MixedBatchDistributedSampler(
                datasets=dataset,
                sample_ratio=sample_ratio,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed)
            shuffle = False
            dataset = ConcatDataset(dataset)
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if torch.__version__ >= '1.7.0':
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            shuffle=shuffle,
            worker_init_fn=init_fn,
            persistent_workers=persistent_workers,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            shuffle=shuffle,
            worker_init_fn=init_fn,
            **kwargs)
    return data_loader


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """Worker initialization function.

    Args:
        worker_id (int): the worker id for each worker subprocess.
        num_workers (int): how many subprocesses to use for data loading.
        rank (int): the rank of current process group.
        seed (int): the seed for generating random numbers for data workers.
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
