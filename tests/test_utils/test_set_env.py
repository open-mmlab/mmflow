# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmengine.registry import init_default_scope


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
