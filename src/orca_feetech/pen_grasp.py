"""Learn a loaded grasp after manual placement; this task does not rotate a pen."""
from collections import deque
from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json

import gymnasium as gym
import mujoco
import numpy as np

from .checkpoint import _load_config
from .contact_support import hand_object_contacts
from .manipulation import TaskConfig,ObservationCodec,DT,FRAME_SIZE,build_model,rotation_error,yaw_rotation
from .manipulation_v8 import ManipulationV8Env,TaskConfigV8,task_contract as v8_contract
from .model import ModelSpec
from .objects import geometry_signature
from .reference_control import BrakingActionCodec
from .wrist_diagnostic import correct_wrist_reference


TASK_ID_GRASP='orca-v1-pen-grasp-v1'
ASSET_SHA256='d9f85d805d08690774fad3b61df929f9ae6b519e35eff60d38e365ccfe3157ee'


def grasp_asset():
    raw=files('orca_feetech').joinpath('data/pen_grasp_v1.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ASSET_SHA256:
        raise ValueError('Frozen grasp preparation manifest changed')
    asset=json.loads(raw)
    geometry_sha=hashlib.sha256(json.dumps(geometry_signature('pen'),sort_keys=True).encode()).hexdigest()
    if (asset['model_sha256']!=ModelSpec.load().source_sha256
            or asset['geometry_sha256']!=geometry_sha):
        raise ValueError('Grasp preparation model/geometry mismatch')
    return asset


def grasp_contract():
    controls={k:v for k,v in v8_contract().items() if k not in ('task_id','reset','reward_and_rotation_acceptance')}
    return dict(task_id=TASK_ID_GRASP,base_control_and_physics=controls,asset_sha256=ASSET_SHA256,
        objective='7s retention, then a continuous final0.5s stationary loaded thumb/middle pinch',
        orientation='Free settling; initial rotation is an observation reference, not an orientation target',
        reset='6.5s physical empty reach each reset, then one independent manual release; no retry or successful-state selection',
        release_jitter=dict(position_uniform_half_width_m=.00025,yaw_uniform_half_width_deg=.5,
                            base_placements='All seven fixed placement buckets'),
        failures=dict(penetration_m=.002,maximum_displacement_m=.04,maximum_normal_weight_ratio=10.,
                      minimum_height_m=.10,maximum_vision_age_s=.25),
        final_half_second=dict(linear_speed_m_s=.015,angular_speed_rad_s=.3,
            orientation_span_deg=2.,position_span_m=.002,minimum_each_normal_weight_ratio=.05,
            maximum_opposed_normal_dot=-.5),
        reward=dict(pair_score=.05,pair_times_rest_score=.2,action_squared=.002,
                    action_difference_squared=.002,displacement=.3,failure=-20.,final_success=10.),
        geometry_audit='Every2ms integration step and final forward',
        formal_gate='100 episodes, success>=80%, drops<=10%; separate from rotation/sim2real acceptance')


@dataclass
class GraspConfig(TaskConfigV8):
    object: str='pen'
    angle: float=10.
    horizon: int=140
    randomization: float=0.

    def __post_init__(self):
        super().__post_init__()
        if self.object!='pen' or self.angle!=10 or self.turns!=1 or self.horizon!=140:
            raise ValueError('Grasp v1 fixes pen, placeholder angle10, turns1 and horizon140; no rotation curriculum')

    def fingerprint(self,*,task_id=TASK_ID_GRASP):
        if task_id!=TASK_ID_GRASP:
            raise ValueError('Grasp config cannot masquerade as a rotation task')
        grasp_asset()
        payload=dict(base=TaskConfig.fingerprint(self,task_id=task_id),contract=grasp_contract())
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def build_grasp_model(config):
    model=build_model(config,task_id=TASK_ID_GRASP)
    correct_wrist_reference(model,ModelSpec.load())
    return model


def pinch_load(env):
    contacts,excluded=hand_object_contacts(env.model,env.data,env.geoms)
    forces={}
    normals={}
    for c in contacts:
        normal=c['sign']*c['frame'][0]
        force=max(0.,float(c['solver_force_world_n']@normal))
        forces[c['body']]=forces.get(c['body'],0.)+force
        if force>1e-6:
            normals.setdefault(c['body'],[]).append(normal)
    thumb,middle='right_thumb_dp','right_middle_ip'
    dots=[float(a@b) for a in normals.get(thumb,[]) for b in normals.get(middle,[])]
    weight=float(env.model.body_mass[env.body]*np.linalg.norm(env.model.opt.gravity))
    loaded=bool(min(forces.get(thumb,0.),forces.get(middle,0.))>=.05*weight and dots and min(dots)<=-.5)
    score=min(1.,min(forces.get(thumb,0.),forces.get(middle,0.))/(.05*weight))
    score*=float(np.clip(-min(dots,default=0.)/.5,0,1))
    return dict(normal_force_by_body_n=forces,minimum_thumb_target_normal_dot=min(dots,default=None),
                summed_normal_force_n=sum(forces.values()),object_weight_n=weight,
                loaded_opposition=loaded,pair_score=score,excluded_force_support=bool(excluded))


class PenGraspEnv(ManipulationV8Env):
    model_builder=staticmethod(build_grasp_model)

    def __init__(self,config=None,**kwargs):
        config=GraspConfig() if config is None else config
        if not isinstance(config,GraspConfig):
            raise TypeError('Grasp task requires GraspConfig')
        if kwargs.get('minimum_goal_angle') is not None or kwargs.get('negative_goal_probability',.5)!=.5:
            raise ValueError('Grasp task has no goal-angle or directional mixture')
        super().__init__(config,**kwargs)
        self.asset=grasp_asset()
        self.target=np.r_[self.asset['target_qpos_rad'],self.hand_spec.neutral[-1]]
        if np.any(self.target<self.lower) or np.any(self.target>self.upper):
            raise ValueError('Pregrasp target outside measured ROM')
        self.geom_body_names=[self.model.body(i).name or '' for i in self.model.geom_bodyid]

    def _geometry(self):
        self_depth=object_depth=0.
        excluded=False
        for c in self.data.contact:
            if c.dist>=0:
                continue
            names=[self.geom_body_names[c.geom1],self.geom_body_names[c.geom2]]
            if c.geom1 in self.geoms or c.geom2 in self.geoms:
                other=names[1] if c.geom1 in self.geoms else names[0]
                object_depth=max(object_depth,-float(c.dist))
                excluded|=not other.startswith('right_') or other=='right_tower'
            elif all(n.startswith('right_') for n in names):
                self_depth=max(self_depth,-float(c.dist))
        return self_depth,object_depth,bool(excluded)

    def _place_object(self,position,rotation):
        hand_q=self.data.qpos[self.qadr].copy()
        hand_v=self.data.qvel[self.hand_dofs].copy()
        quaternion=np.empty(4)
        mujoco.mju_mat2Quat(quaternion,np.asarray(rotation).ravel())
        self.data.qpos[self.oq:self.oq+7]=np.r_[position,quaternion]
        self.data.qvel[self.ov:self.ov+6]=0.  # Human-supported object released once at episode start.
        self.data.qacc_warmstart[:]=0.  # New contacts: numerical warmstart only.
        mujoco.mj_forward(self.model,self.data)
        if (not np.array_equal(hand_q,self.data.qpos[self.qadr]) or
                not np.array_equal(hand_v,self.data.qvel[self.hand_dofs])):
            raise RuntimeError('Manual release changed the physically reached hand')

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        options=options or {}
        if set(options)-{'placement_index'}:
            raise ValueError('Grasp reset only accepts a placement index')
        index=options.get('placement_index')
        if index is None:
            index=int(self.np_random.integers(7))
        if type(index) is not int or not 0<=index<7:
            raise ValueError('Placement index must be0..6')
        self.placement_index=index
        self._randomize()
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.qadr]=self.hand_spec.neutral
        self.data.ctrl[self.actuators]=self.hand_spec.neutral
        self.data.qpos[self.oq:self.oq+7]=[.6,0,.8,1,0,0,0]
        mujoco.mj_forward(self.model,self.data)
        self.codec=BrakingActionCodec(self.config,self.hand_spec.neutral)
        self.queue=deque([self.codec.q.copy() for _ in range(self.delay+1)],maxlen=self.delay+1)
        reach_self=0.
        for tick in range(130):
            phase=float(np.clip(((tick+1)*DT-.5)/5.,0,1))
            blend=phase**3*(10+phase*(-15+6*phase))
            desired=self.hand_spec.neutral[:16]+blend*(self.target[:16]-self.hand_spec.neutral[:16])
            command=self.codec.step(np.clip((desired-self.codec.q[:16])/(self.config.max_velocity*DT),-1,1))
            self.queue.append(command)
            self.data.ctrl[self.actuators]=np.clip(self.queue[0]+self.bias,self.lower,self.upper)
            for _ in range(25):
                mujoco.mj_step(self.model,self.data)
                reach_self=max(reach_self,self._geometry()[0])
        mujoco.mj_forward(self.model,self.data)
        reach_self=max(reach_self,self._geometry()[0])
        dofs=self.data.qvel[self.hand_dofs]
        reach_error=float(np.sqrt(np.mean((self.data.qpos[self.qadr[:16]]-self.target[:16])**2)))
        reached=bool(reach_self<=.002 and reach_error<.03 and np.max(np.abs(dofs))<.1
                     and abs(self.data.qpos[self.qadr[-1]]-self.hand_spec.neutral[-1])<.01)
        self.reset_failure=None if reached else 'pregrasp_reach_failed'
        self.termination_reason=None
        position=(np.array(self.asset['object_position_m'])+self.asset['placement_offsets_m'][index]
                  +self.np_random.uniform(-.00025,.00025,3))
        rotation=yaw_rotation(np.deg2rad(self.np_random.uniform(-.5,.5)))@self.asset['object_rotation']
        if reached:
            self._place_object(position,rotation)
        self.reached_hand_qpos=self.data.qpos[self.qadr].copy()
        self.reached_hand_qvel=self.data.qvel[self.hand_dofs].copy()
        self.origin,self.goal=self.pose()
        self.goal_angle=0.
        self.increment=np.eye(3)
        self.direction=1
        self.age=0.
        self.valid=True
        self.steps=self.stable=self.completed=0
        self.previous_action=np.zeros(16)
        self.measured_pos,self.measured_rot=self.origin.copy(),self.goal.copy()
        self.observer=ObservationCodec(self.config)
        self.last_window=deque(maxlen=10)
        self.peak_self=self.peak_object=self.peak_normal=self.peak_displacement=0.
        self.excluded=False
        self.data.time=0.
        self._audit()
        if self._constraint_reason():
            self.reset_failure=self.reset_failure or 'release_initial_geometry_or_state_rejected'
        self.reset_diagnostics=dict(reached=reached,target_error_rms_rad=reach_error,
                                    peak_empty_reach_self_penetration_m=reach_self)
        return self._observation(),dict(self.info(),reset_attempts=1,settle_seconds=0.)

    def _audit(self):
        a,b,bad=self._geometry()
        self.peak_self=max(self.peak_self,a)
        self.peak_object=max(self.peak_object,b)
        self.excluded|=bad
        load=pinch_load(self)
        self.peak_normal=max(self.peak_normal,load['summed_normal_force_n'])
        self.peak_displacement=max(self.peak_displacement,float(np.linalg.norm(self.pose()[0]-self.origin)))

    def _constraint_reason(self):
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(w.number for w in self.data.warning)):
            return 'numerical_failure'
        if max(self.peak_self,self.peak_object)>.002:
            return 'penetration_limit'
        if self.excluded:
            return 'excluded_support'
        weight=float(self.model.body_mass[self.body]*np.linalg.norm(self.model.opt.gravity))
        if self.peak_normal>10*weight:
            return 'normal_force_limit'
        if self.peak_displacement>=.04:
            return 'displacement_limit'
        return None

    def _physics_step(self):
        for _ in range(25):
            mujoco.mj_step(self.model,self.data)
            self._audit()
        mujoco.mj_forward(self.model,self.data)
        self._audit()

    def info(self):
        info=super().info()
        load=pinch_load(self)
        resting=bool(info['linear_speed_mps']<.015 and info['angular_speed_rps']<.3)
        stable=bool(resting and load['loaded_opposition'] and self.contact())
        return dict(info,**load,resting=resting,stable=stable,is_success=bool(self.completed),
            placement_index=self.placement_index,reset_failure=self.reset_failure,
            completed_holds=int(self.completed),
            preparation_seconds=6.5,manual_release_initialization=True,
            peak_self_penetration_m=self.peak_self,peak_object_penetration_m=self.peak_object,
            peak_normal_force_n=self.peak_normal,maximum_displacement_m=self.peak_displacement,
            termination_reason=getattr(self,'termination_reason',None))

    def step(self,action):
        if self.reset_failure:
            self.termination_reason=self.reset_failure
            return self._observation(),-20.,True,False,self.info()
        info=self._advance(action)
        position,rotation=self.pose()
        self.last_window.append((position,rotation,info['stable']))
        reason=self._constraint_reason()
        reason=reason or ('dropped' if info['dropped'] else 'vision_timeout' if self.age>.25 else None)
        angular_span=position_span=None
        if len(self.last_window)==10:
            angular_span=max(float(np.rad2deg(rotation_error(r,rotation))) for _,r,_ in self.last_window)
            position_span=max(float(np.linalg.norm(p-position)) for p,_,_ in self.last_window)
        success=bool(self.steps==140 and not reason and len(self.last_window)==10
            and all(s for _,_,s in self.last_window) and angular_span<2 and position_span<.002)
        rest_score=float(np.exp(-min(100.,(info['linear_speed_mps']/.015)**2+(info['angular_speed_rps']/.3)**2)))
        reward=(.05*info['pair_score']+.2*info['pair_score']*rest_score
            -.002*np.mean(np.square(action))-.002*np.mean(np.square(np.asarray(action)-self.previous_action))
            -.3*info['displacement_m']-20*bool(reason)+10*success)
        self.previous_action=np.asarray(action).copy()
        self.completed=int(success)
        self.termination_reason=reason or ('success' if success else 'horizon' if self.steps>=140 else None)
        info=self.info()
        info.update(final_500ms_orientation_span_deg=angular_span,final_500ms_position_span_m=position_span)
        return self._observation(),float(reward),bool(reason or self.steps>=140),False,info


def load_grasp_config(checkpoint):
    config,meta=_load_config(checkpoint,TASK_ID_GRASP,FRAME_SIZE,GraspConfig)
    if meta.get('task_contract')!=grasp_contract():
        raise ValueError('Grasp objective/reset/controller contract mismatch')
    return config,meta
