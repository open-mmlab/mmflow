# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


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
    import mmflow.datasets  # noqa: F401,F403
    import mmflow.engine  # noqa: F401,F403
    import mmflow.evaluation  # noqa: F401,F403
    import mmflow.models  # noqa: F401,F403
    import mmflow.structures  # noqa: F401,F403
    import mmflow.visualization  # noqa: F401,F403

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
