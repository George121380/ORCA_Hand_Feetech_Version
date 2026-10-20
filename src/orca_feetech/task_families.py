"""Explicit job/recovery dispatch; the legacy default remains reproducible."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskFamily:
    cli_script: str
    config_class: type
    load_config: object


def task_family(name):
    if name=='legacy-v7':
        from .manipulation import TaskConfig
        from .checkpoint import load_config
        return TaskFamily('manipulation.py',TaskConfig,load_config)
    if name=='v8':
        from .manipulation_v8 import TaskConfigV8,load_v8_config
        return TaskFamily('manipulation_v8.py',TaskConfigV8,load_v8_config)
    if name=='pen-grasp-v1':
        from .pen_grasp import GraspConfig,load_grasp_config
        return TaskFamily('pen_grasp.py',GraspConfig,load_grasp_config)
    if name=='pen-rotate-v1':
        from .pen_rotate import RotationConfig,load_rotation_config
        return TaskFamily('pen_rotate.py',RotationConfig,load_rotation_config)
    raise ValueError('Unknown explicit task family')
