# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
from os.path import dirname, exists, isdir, join, relpath

import numpy as np
from mmengine import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from torch import nn

from mmflow.models import build_flow_estimator
from mmflow.utils import register_all_modules

register_all_modules()


def _get_config_directory():
    """Find the predefined segmentor config directory."""
    try:
        # Assume we are running in the source mmflow repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmflow
        repo_dpath = dirname(dirname(mmflow.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_config_build_flow_estimator():
    """Test that all segmentation models defined in the configs can be
    initialized."""
    config_dpath = _get_config_directory()
    print(f'Found config_dpath = {config_dpath!r}')

    config_fpaths = []
    # one config each sub folder
    for sub_folder in os.listdir(config_dpath):
        if isdir(sub_folder):
            config_fpaths.append(
                list(glob.glob(join(config_dpath, sub_folder, '*.py')))[0])
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]
    print(f'Using {len(config_names)} config files')
    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)
        config_mod.model
        print(f'Building segmentor, config_fpath = {config_fpath!r}')
        print(f'building {config_fname}')
        flow_estimator = build_flow_estimator(config_mod.model)
        assert flow_estimator is not None


def test_config_data_pipeline():
    """Test whether the data pipeline is valid and can process corner cases.

    CommandLine:
        xdoctest -m tests/test_config.py test_config_build_data_pipeline
    """

    register_all_modules()
    config_dpath = _get_config_directory()
    print(f'Found config_dpath = {config_dpath!r}')

    import glob
    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print(f'Using {len(config_names)} config files')

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        print(f'Building data pipeline, config_fpath = {config_fpath!r}')
        config_mod = Config.fromfile(config_fpath)
        if hasattr(config_mod, 'train_dataloader'):
            config_mod.train_dataloader.dataset.lazy_init = True
            Runner.build_dataloader(config_mod.train_dataloader)
        if hasattr(config_mod, 'test_dataloader'):
            config_mod.test_dataloader.dataset.lazy_init = True
            Runner.build_dataloader(config_mod.test_dataloader)
