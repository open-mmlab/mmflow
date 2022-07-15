# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

import mmcv
import pytest
from mmengine import DefaultScope

import mmflow
from mmflow.utils import register_all_modules


def test_collect_env():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip('skipping tests that require PyTorch')

    from mmflow.utils import collect_env
    env_info = collect_env()
    expected_keys = [
        'sys.platform', 'Python', 'CUDA available', 'PyTorch',
        'PyTorch compiling details', 'OpenCV', 'MMCV', 'MMCV Compiler',
        'MMCV CUDA Compiler', 'MMFlow', 'GCC'
    ]
    for key in expected_keys:
        assert key in env_info

    if env_info['CUDA available']:
        for key in ['CUDA_HOME', 'NVCC']:
            assert key in env_info

    assert env_info['sys.platform'] == sys.platform
    assert env_info['Python'] == sys.version.replace('\n', '')
    assert env_info['MMCV'] == mmcv.__version__
    assert mmflow.__version__ in env_info['MMFlow']


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmflow.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmflow.datasets', None)
        sys.modules.pop('mmflow.datasets.flyingchairs', None)
        DATASETS._module_dict.pop('FlyingChairs', None)
        self.assertFalse('FlyingChairs' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('FlyingChairs' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmflow.datasets')
        sys.modules.pop('mmflow.datasets.flyingchairs')
        DATASETS._module_dict.pop('FlyingChairs', None)
        self.assertFalse('FlyingChairs' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('FlyingChairs' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmflow')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmflow"'):
            register_all_modules(init_default_scope=True)
