"""Goal-conditioned yaw after an immutable, physically executed learned grasp.

This is one small reorientation, not a revolution or continuous pen spinning.
The carrier keeps the release-origin audits and its physical/controller history.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from .checkpoint import _load_config
from .manipulation import TaskConfig, FRAME_SIZE, rotation_error, yaw_rotation
from .manipulation_v8 import TaskConfigV8
from .pen_grasp import PenGraspEnv, load_grasp_config, grasp_contract


TASK_ID_ROTATE='orca-v1-pen-rotate-v1'
PREFIX_SHA256='9ff037afb2ae122b50ec7128fe792314216d60c6536d1c8eba94e09a6928c374'
PREFIX_FINGERPRINT='6cfaf0cb817ba86afaa63dea484de1f050b6da2737cd961fba4ff9b1ead33f2d'
PHYSICAL_KEYS=('object','mass_kg','friction','max_velocity','max_acceleration',
               'delay_max_steps','command_bias_deg','encoder_bias_deg','lower','upper')


def rotation_contract():
    return dict(task_id=TASK_ID_ROTATE,grasp_contract=grasp_contract(),
        prefix_sha256=PREFIX_SHA256,prefix_fingerprint=PREFIX_FINGERPRINT,prefix_policy_steps=200704,
        reset='One physical empty reach, manual release, then140 deterministic frozen grasp actions; failed prefixes count, no retry or state selection',
        handoff='Keep qpos/qvel/time/controller/queue/camera/RNG and preceding three observation frames; never clear velocity or reset release-origin audits',
        goal='One world-Z +/-angle target from the last measured prefix-end rotation; direction sampled only after the physical prefix',
        actor='Same296 deployable observations and16 bounded braking velocity actions; no force or truth velocity inputs',
        horizon='Full continuation horizon, at least7s after the7s learned prefix; no early success',
        success='Final continuous1s: fullSO3<8deg, hand contact, loaded thumb/middle opposition, v<.015m/s, w<.3rad/s; final1s spans<2deg and2mm; all original2ms audits retained',
        reward=dict(error_progress=8.,goal_rest_pair=.25,pair=.03,error_per_radian=-.02,
                    action_squared=-.002,action_difference_squared=-.002,displacement=-.3,
                    failure=-20.,terminal_success=20.,goal_score_width_rad=.14),
        credit='One reorientation subgoal; completed revolutions always0; not continuous rotation or hardware acceptance')


@dataclass
class RotationConfig(TaskConfigV8):
    object: str='pen'
    angle: float=15.
    horizon: int=140
    randomization: float=.3

    def __post_init__(self):
        super().__post_init__()
        if self.object!='pen' or self.turns!=1 or self.horizon<140 or self.angle<15:
            raise ValueError('Rotation v1 requires pen, one subgoal, angle>=15 and horizon>=140')

    def fingerprint(self,*,task_id=TASK_ID_ROTATE):
        if task_id!=TASK_ID_ROTATE:
            raise ValueError('Rotation config requires its explicit task ID')
        payload=dict(base=TaskConfig.fingerprint(self,task_id=task_id),contract=rotation_contract())
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def validate_grasp_prefix(checkpoint):
    checkpoint=Path(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=PREFIX_SHA256:
        raise ValueError('Rotation prefix must be the immutable accepted94/100 grasp policy')
    config,meta=load_grasp_config(checkpoint)
    if (config.fingerprint()!=PREFIX_FINGERPRINT or not meta.get('trained')
            or meta.get('actual_steps')!=200704):
        raise ValueError('Frozen grasp prefix metadata mismatch')
    return config,meta


def final_rotation_success(window,info,reason,*,at_horizon):
    """Require retained success at the declared horizon, not a transient hit."""
    if not at_horizon or reason or len(window)!=20 or not all(row[2] for row in window):
        return False,None,None
    position,rotation,_=window[-1]
    angular=max(float(np.rad2deg(rotation_error(r,rotation))) for _,r,_ in window)
    distance=max(float(np.linalg.norm(p-position)) for p,_,_ in window)
    return bool(not info['dropped'] and angular<2 and distance<.002),angular,distance


class PenRotateEnv(gym.Env):
    metadata=PenGraspEnv.metadata

    def __init__(self,config=None,*,negative_goal_probability=.5,minimum_goal_angle=None,
                 prefix_checkpoint=None):
        from stable_baselines3 import PPO
        import torch
        self.config=RotationConfig() if config is None else config
        if not isinstance(self.config,RotationConfig):
            raise TypeError('Rotation task requires RotationConfig')
        if minimum_goal_angle is not None or not 0<=negative_goal_probability<=1:
            raise ValueError('Rotation v1 uses full-angle goals only')
        checkpoint=prefix_checkpoint or os.environ.get('ORCA_GRASP_CHECKPOINT')
        if checkpoint is None:
            raise ValueError('Set ORCA_GRASP_CHECKPOINT to frozen accepted policy.zip with config.json')
        prefix_config,_=validate_grasp_prefix(checkpoint)
        if any(asdict(prefix_config)[k]!=asdict(self.config)[k] for k in PHYSICAL_KEYS):
            raise ValueError('Rotation changes the frozen prefix physical/action/observation envelope')
        torch.set_num_threads(1)
        self.prefix=PPO.load(checkpoint,device='cpu')
        self.carrier=PenGraspEnv(replace(prefix_config,randomization=self.config.randomization))
        self.action_space=self.carrier.action_space
        self.observation_space=self.carrier.observation_space
        self.negative_goal_probability=negative_goal_probability

    @property
    def goal(self):
        return self.carrier.goal

    def pose(self):
        return self.carrier.pose()

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        options={} if options is None else dict(options)
        if set(options)-{'placement_index','direction'}:
            raise ValueError('Rotation reset accepts placement_index and evaluation direction only')
        forced_direction=options.pop('direction',None)
        if forced_direction is not None and forced_direction not in (-1,1):
            raise ValueError('Direction must be -1 or1')
        obs,info=self.carrier.reset(seed=seed,options=options)
        for _ in range(140):
            action=self.prefix.predict(obs,deterministic=True)[0]
            obs,_,done,truncated,info=self.carrier.step(action)
            if done or truncated:
                break
        self.prefix_success=bool(info['is_success'])
        self.prefix_steps=self.carrier.steps
        self.prefix_reason=info['termination_reason']
        self.steps=0
        self.completed=0
        self.termination_reason=None if self.prefix_success else 'grasp_prefix_failed'
        self.previous_action=self.carrier.previous_action.copy()
        self.window=deque(maxlen=20)
        self.first_goal_tick=None
        # Goal selection cannot change the preceding physical preparation distribution.
        self.direction=(int(forced_direction) if forced_direction is not None else
            (-1 if self.carrier.np_random.random()<self.negative_goal_probability else 1))
        self.goal_capture_rotation=self.carrier.measured_rot.copy()
        self.carrier.goal=yaw_rotation(np.deg2rad(self.direction*self.config.angle))@self.goal_capture_rotation
        self.carrier.goal_angle=np.deg2rad(self.config.angle)
        self.start_position,self.start_rotation=self.pose()
        self.last_error=rotation_error(self.start_rotation,self.goal)
        obs=self.carrier._observation()  # Retain preceding three real history frames.
        return obs,dict(self.info(),reset_attempts=1,settle_seconds=0.)

    def info(self):
        original=self.carrier.info()
        failure=self.carrier._constraint_reason() or ('dropped' if original['dropped'] else
            'vision_timeout' if original['vision_age']>.25 else None)
        # Truth is only used for scoring; actor observations remain unchanged.
        gate=bool(not failure and original['angle_error_deg']<8 and self.carrier.contact()
            and original['loaded_opposition'] and original['resting'])
        result=dict(original,source_grasp_completed=original['is_success'],
            source_grasp_stable=original['stable'],is_success=bool(self.completed),stable=gate,
            completed_holds=0,completed_turns=0,completed_subgoals=int(self.completed),
            prefix_grasp_success=self.prefix_success,prefix_steps=self.prefix_steps,
            prefix_end_reason=self.prefix_reason,rotation_steps=self.steps,
            prefix_seconds=self.prefix_steps*.05,preparation_seconds=6.5+self.prefix_steps*.05,
            goal_angle_deg=self.config.angle,direction=self.direction,
            first_goal_tick=self.first_goal_tick,termination_reason=self.termination_reason,
            rotation_from_handoff_deg=float(np.rad2deg(rotation_error(self.pose()[1],self.start_rotation))))
        return result

    def step(self,action):
        if not self.prefix_success:
            return self.carrier._observation(),-20.,True,False,self.info()
        if self.termination_reason is not None:
            raise RuntimeError('Reset after the rotation episode terminates')
        self.carrier._advance(action)
        self.steps+=1
        info=self.info()
        reason=self.carrier._constraint_reason() or ('dropped' if info['dropped'] else
            'vision_timeout' if info['vision_age']>.25 else None)
        position,rotation=self.pose()
        self.window.append((position.copy(),rotation.copy(),info['stable']))
        if info['stable'] and self.first_goal_tick is None:
            self.first_goal_tick=self.steps  # Single tick only; never success credit.
        success,span,distance=final_rotation_success(self.window,info,reason,
                                                   at_horizon=self.steps>=self.config.horizon)
        error=rotation_error(rotation,self.goal)
        rest=float(np.exp(-min(100.,(info['linear_speed_mps']/.015)**2+(info['angular_speed_rps']/.3)**2)))
        goal_score=float(np.exp(-min(100.,(error/.14)**2)))
        reward=(8*(self.last_error-error)+.25*goal_score*rest*info['pair_score']+.03*info['pair_score']
            -.02*error-.002*np.mean(np.square(action))
            -.002*np.mean(np.square(np.asarray(action)-self.previous_action))
            -.3*info['displacement_m']-20*bool(reason)+20*success)
        self.last_error=error
        self.previous_action=np.asarray(action).copy()
        self.completed=int(success)
        self.termination_reason=reason or ('success' if success else 'horizon' if self.steps>=self.config.horizon else None)
        info=self.info()
        info.update(final_1s_orientation_span_deg=span,final_1s_position_span_m=distance)
        return self.carrier._observation(),float(reward),bool(reason or self.steps>=self.config.horizon),False,info

    def render(self):
        return self.carrier.render()

    def close(self):
        self.carrier.close()


def load_rotation_config(checkpoint):
    config,meta=_load_config(checkpoint,TASK_ID_ROTATE,FRAME_SIZE,RotationConfig)
    if meta.get('task_contract')!=rotation_contract():
        raise ValueError('Rotation prefix/goal/reward/retention contract mismatch')
    return config,meta
