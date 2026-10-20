"""Explicit corrected-wrist/braking task, separate from historical v6/v7."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .checkpoint import _load_config
from .manipulation import ManipulationEnv,TaskConfig,build_model,FRAME_SIZE
from .model import ModelSpec
from .reference_control import BrakingActionCodec,CONTROLLER_ID
from .wrist_diagnostic import correct_wrist_reference


TASK_ID_V8='orca-v1-printable-yaw-v8'
RESET_HAND_SPEED=.1
RESET_WRIST_TOLERANCE=.01


def task_contract():
    return dict(task_id=TASK_ID_V8,controller=CONTROLLER_ID,
        wrist_equality='q - model.qpos0[wrist] = neutral - model.qpos0[wrist]',
        wrist_target_rad=float(ModelSpec.load().neutral[-1]),
        invalid_observation='Request zero velocity with bounded deceleration; position can continue moving',
        observation='74 x 4; controller velocity is actual position-command difference / 0.05s',
        reset=dict(object_rest='v6 physical settling; no object velocity clearing',
                   maximum_actual_hand_speed_rad_s=RESET_HAND_SPEED,
                   maximum_actual_wrist_neutral_error_rad=RESET_WRIST_TOLERANCE),
        reward_and_rotation_acceptance='Unchanged v7 full-SO3, displacement, contact and speed gates')


@dataclass
class TaskConfigV8(TaskConfig):
    def fingerprint(self,*,task_id=TASK_ID_V8):
        if task_id!=TASK_ID_V8:
            raise ValueError('v8 config cannot masquerade as a legacy task')
        payload=dict(base=super().fingerprint(task_id=task_id),contract=task_contract())
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def build_v8_model(config):
    model=build_model(config,task_id=TASK_ID_V8)
    correct_wrist_reference(model,ModelSpec.load())
    return model


class ManipulationV8Env(ManipulationEnv):
    model_builder=staticmethod(build_v8_model)
    action_codec_class=BrakingActionCodec

    def __init__(self,config=None,**kwargs):
        config=TaskConfigV8() if config is None else config
        if not isinstance(config,TaskConfigV8):
            raise TypeError('v8 requires its explicit TaskConfigV8 contract')
        super().__init__(config,**kwargs)
        self.hand_dofs=self.model.jnt_dofadr[self.model.actuator_trnid[self.actuators,0]]

    def _reset_ready(self):
        return bool(np.max(np.abs(self.data.qvel[self.hand_dofs]))<RESET_HAND_SPEED
            and abs(self.data.qpos[self.qadr[-1]]-self.hand_spec.neutral[-1])<RESET_WRIST_TOLERANCE)


def load_v8_config(checkpoint):
    config,meta=_load_config(checkpoint,TASK_ID_V8,FRAME_SIZE,TaskConfigV8)
    if meta.get('task_contract')!=task_contract():
        raise ValueError('v8 wrist/controller/reset contract mismatch')
    return config,meta
