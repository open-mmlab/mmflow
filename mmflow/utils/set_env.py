# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os
import platform
import warnings

import cv2
import torch.multiprocessing as mp
from mmengine import DefaultScope

from ..utils import get_root_logger


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    logger = get_root_logger()

    # set multi-process start method
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', None)
        current_method = mp.get_start_method(allow_none=False)
        if mp_start_method in ('fork', 'spawn', 'forkserver'):
            logger.info(
                f'Multi-processing start method is `{mp_start_method}`')
            mp.set_start_method(mp_start_method, force=True)
        else:
            logger.info(f'Multi-processing start method is `{current_method}`')

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', None)
    if isinstance(opencv_num_threads, int):
        logger.info(f'OpenCV num_threads is `{opencv_num_threads}`')
        cv2.setNumThreads(opencv_num_threads)
    else:
        logger.info(f'OpenCV num_threads is `{cv2.getNumThreads()}')

    if cfg.data.train_dataloader.workers_per_gpu > 1:
        # setup OMP threads
        # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
        omp_num_threads = cfg.get('omp_num_threads', None)
        if 'OMP_NUM_THREADS' not in os.environ:
            if isinstance(omp_num_threads, int):
                logger.info(f'OMP num threads is {omp_num_threads}')
                os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
        else:
            logger.info(f'OMP num threads is {os.environ["OMP_NUM_THREADS"] }')

        # setup MKL threads
        if 'MKL_NUM_THREADS' not in os.environ:
            mkl_num_threads = cfg.get('mkl_num_threads', None)
            if isinstance(mkl_num_threads, int):
                logger.info(f'MKL num threads is {mkl_num_threads}')
                os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)
        else:
            logger.info(f'MKL num threads is {os.environ["MKL_NUM_THREADS"]}')


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmflow into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmflow default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmflow`, and all registries will build modules from mmflow's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmflow.core  # noqa: F401,F403
    import mmflow.datasets  # noqa: F401,F403
    import mmflow.models  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmflow')
        if never_created:
            DefaultScope.get_instance('mmflow', scope_name='mmflow')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmflow':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmflow", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmflow". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmflow-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmflow')
