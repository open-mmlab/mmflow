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
        current_method = mp.get_start_method(allow_none=True)
        if type(mp_start_method) is str:
            logger.info(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`.')
            mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', None)
    if isinstance(opencv_num_threads, int):
        logger.info(f'OpenCV num_threads is {opencv_num_threads}')
        cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        omp_num_threads = cfg.get('omp_num_threads', None)
        if isinstance(omp_num_threads, int):
            logger.info(f'OMP num threads is {omp_num_threads}')
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'mkl_num_threads' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = cfg.get('mkl_num_threads', None)
        if isinstance(mkl_num_threads, int):
            logger.info(f'MKL num threads is {mkl_num_threads}')
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)
