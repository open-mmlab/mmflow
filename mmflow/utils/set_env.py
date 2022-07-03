# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform

import cv2
import torch.multiprocessing as mp

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

    if cfg.data.get('train_dataloader') is not None:
        workers_per_gpu = cfg.data.train_dataloader.get('workers_per_gpu', 1)
    elif cfg.data.get('test_dataloader') is not None:
        workers_per_gpu = cfg.data.test_dataloader.get('workers_per_gpu', 1)
    else:
        workers_per_gpu = 0
    if workers_per_gpu > 1:
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
