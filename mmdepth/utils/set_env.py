# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmdepth into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmdepth default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmdepth`, and all registries will build modules from mmdepth's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmdepth.datasets  # noqa: F401,F403
    import mmdepth.engine  # noqa: F401,F403
    import mmdepth.evaluation  # noqa: F401,F403
    import mmdepth.models  # noqa: F401,F403
    import mmdepth.structures  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmdepth')
        if never_created:
            DefaultScope.get_instance('mmdepth', scope_name='mmdepth')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmdepth':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmdepth", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmdepth". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmdepth-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmdepth')
