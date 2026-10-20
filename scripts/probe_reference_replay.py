"""Bounded open-loop replay using the existing shared ActionCodec.

Synthetic initialization only once per run. Physics is free thereafter. No
training, ordinary reset, hardware, reference-state injection during playback
or policy acceptance. Schedule completion is not object tracking success.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from orca_feetech.manipulation import ActionCodec, DT, ManipulationEnv, TaskConfig, rotation_error
import probe_reference_hold as hold_helpers
from probe_reference_hold import geometry_hash, snapshot, validate_reference


class Trajectory:
    def __init__(self,reference,mode):
        if mode not in ('baseline','refined'):
            raise ValueError('Use baseline or refined keyframes')
        rows=reference['trials']
        self.times=np.array([r['time_s'] for r in rows],float)
        if (len(rows)<2 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times)<=0)):
            raise ValueError('Expected at least two strictly ordered keyframes')
        self.source_start=float(self.times[0])
        self.times-=self.source_start
        if not .05<=self.times[-1]<=7.5:
            raise ValueError('Bounded probe requires a 0.05..7.5 second source span')
        self.duration=float(self.times[-1])
        self.joints=np.array([r['qpos'] if mode=='refined' else r['baseline']['qpos'] for r in rows])
        self.positions=np.array([r['object_position_m'] for r in rows])
        quaternions=np.array([r['object_quaternion_wxyz'] for r in rows])
        self.rotations=Slerp(self.times,Rotation.from_quat(quaternions[:,[1,2,3,0]]))

    def sample(self,time):
        time=float(np.clip(time,0,self.duration))
        q=np.array([np.interp(time,self.times,self.joints[:,j]) for j in range(16)])
        position=np.array([np.interp(time,self.times,self.positions[:,j]) for j in range(3)])
        return q,position,self.rotations(time).as_matrix()


def command_kinematics(initial,samples,config):
    commands=np.array([initial,*[s['command'] for s in samples]])[:,:16]
    if len(commands)<2:
        return dict(peak_velocity_rad_s=0.,peak_acceleration_rad_s2=0.,acceleration_exceeded_samples=0)
    velocity=np.diff(commands,axis=0)/DT
    acceleration=np.diff(np.vstack([np.zeros(16),velocity]),axis=0)/DT
    return dict(peak_velocity_rad_s=float(np.max(np.abs(velocity))),
                peak_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
                acceleration_exceeded_samples=int(np.sum(np.abs(acceleration)>config.max_acceleration+1e-8)))


def replay(env,trajectory,slowdown):
    if not 1<=slowdown<=4:
        raise ValueError('Use slowdown between 1 and 4')
    ticks=int(np.ceil(trajectory.duration*slowdown/DT))
    if ticks>600:
        raise ValueError('Replay exceeds 600 control ticks')
    q,position,rotation=trajectory.sample(0)
    command=env.hand_spec.neutral.copy()
    command[:16]=q
    mujoco.mj_resetData(env.model,env.data)
    env.data.qpos[env.qadr]=command
    quaternion=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
    env.data.qpos[env.oq:env.oq+7]=np.r_[position,quaternion]
    env.data.ctrl[env.actuators]=command
    mujoco.mj_forward(env.model,env.data)
    origin,initial_rotation=env.pose()
    codec=ActionCodec(env.config,command)
    initial_command=command.copy()
    initial=snapshot(env,origin,initial_rotation,command)
    samples=[]
    termination='schedule_complete'
    numerical_failure=None
    peak_velocity=peak_acceleration=0.
    for tick in range(ticks):
        source_time=min((tick+1)*DT/slowdown,trajectory.duration)
        target,reference_position,reference_rotation=trajectory.sample(source_time)
        # Position error becomes a bounded velocity request. The shared codec
        # limits its internal velocity/acceleration state and fixes the wrist.
        # Effective position-command acceleration is audited separately: the
        # existing codec's hard ROM clipping may introduce faster changes.
        action=np.clip((target-codec.q[:16])/(env.config.max_velocity*DT),-1,1)
        previous_velocity=codec.velocity.copy()
        command=codec.step(action)
        peak_velocity=max(peak_velocity,float(np.max(np.abs(codec.velocity))))
        peak_acceleration=max(peak_acceleration,float(np.max(np.abs(codec.velocity-previous_velocity)))/DT)
        env.data.ctrl[env.actuators]=command
        mujoco.mj_step(env.model,env.data,nstep=25)
        mujoco.mj_forward(env.model,env.data)
        if (not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all()
                or abs(env.data.time-(tick+1)*DT)>1e-8):
            numerical_failure='Non-finite state or solver time reset'
            termination='numerical_failure'
            break
        sample=snapshot(env,origin,initial_rotation,command)
        actual_position,actual_rotation=env.pose()
        sample.update(source_time_s=float(source_time+trajectory.source_start),
            reference_position_m=reference_position.tolist(),
            reference_position_error_m=float(np.linalg.norm(actual_position-reference_position)),
            reference_shaft_error_deg=float(np.rad2deg(np.arccos(np.clip(
                actual_rotation[:,1]@reference_rotation[:,1],-1,1)))),
            reference_rotation_error_deg=float(np.rad2deg(rotation_error(actual_rotation,reference_rotation))),
            command=command.tolist(),controller_velocity_rad_s=codec.velocity.tolist(),
            joint_reference_error_rms_rad=float(np.sqrt(np.mean((env.data.qpos[env.qadr[:16]]-target)**2))))
        samples.append(sample)
        if sample['object_position_m'][2]<.10 or sample['displacement_m']>.12:
            termination='dropped_or_left_initial_workspace'
            break
        if sample['tower_contact']:
            termination='tower_contact'
            break
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
    if warnings:
        numerical_failure=numerical_failure or 'MuJoCo solver warning'
        termination='numerical_failure'
    return dict(slowdown=slowdown,initial=initial,samples=samples,
        simulated_s=float(env.data.time),planned_s=ticks*DT,maximum_control_ticks=600,
        control_ticks=len(samples),termination=termination,warnings=warnings,numerical_failure=numerical_failure,
        command_schedule_completed=bool(termination=='schedule_complete' and len(samples)==ticks),
        acceptance_met=False,peak_controller_velocity_rad_s=peak_velocity,
        peak_controller_acceleration_rad_s2=peak_acceleration,
        effective_command_kinematics=command_kinematics(initial_command,samples,env.config),
        mean_reference_shaft_error_deg=float(np.mean([x['reference_shaft_error_deg'] for x in samples])) if samples else None,
        maximum_reference_shaft_error_deg=max((x['reference_shaft_error_deg'] for x in samples),default=None),
        mean_reference_position_error_m=float(np.mean([x['reference_position_error_m'] for x in samples])) if samples else None,
        mean_joint_reference_error_rms_rad=float(np.mean([x['joint_reference_error_rms_rad'] for x in samples])) if samples else None,
        end=samples[-1] if samples else initial)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--slowdown',type=float,nargs='+',default=[1.,2.,4.])
    args=p.parse_args()
    if (not 1<=len(args.slowdown)<=3 or len(set(args.slowdown))!=len(args.slowdown)
            or any(not 1<=s<=4 for s in args.slowdown)):
        p.error('Use at most three unique slowdown factors in 1..4')
    raw=args.reference.read_bytes()
    reference=json.loads(raw)
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        validate_reference(reference,env)
        trajectories={mode:Trajectory(reference,mode) for mode in ('baseline','refined')}
        args.output.mkdir(parents=True,exist_ok=False)
        report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',
            purpose='Sparse reference replay in nominal physics through the existing shared ActionCodec; no policy acceptance',
            is_physics_rollout=True,acceptance_met=False,randomization_applied=False,
            maximum_replays=6,maximum_control_ticks_per_replay=600,
            reference_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=reference['source_sha256'],
            source_url=reference.get('source_url'),reference_attribution=reference.get('reference_attribution'),
            model_sha256=env.hand_spec.source_sha256,geometry_sha256=geometry_hash(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256(Path(hold_helpers.__file__).read_bytes()).hexdigest(),
            runtime=dict(system=platform.system(),python=platform.python_version(),
                         mujoco=mujoco.__version__,numpy=np.__version__),
            conditions=dict(source_keyframes=len(reference['trials']),source_duration_s=trajectories['refined'].duration,
                joint_interpolation='Piecewise linear position; bounded velocity requests through ActionCodec',
                object_reference_interpolation='Linear position and shortest-arc quaternion SLERP; diagnostic only',
                max_controller_velocity_rad_s=env.config.max_velocity,
                max_controller_acceleration_rad_s2=env.config.max_acceleration,
                initial_velocities='Zero only at synthetic initialization',
                state_injection_during_playback=False,control_dt_s=DT,physics_dt_s=float(env.model.opt.timestep)),
            limitations=['Sparse keyframes are not a validated continuous retargeted motion.',
                'Open-loop playback has no object feedback, residual RL, teacher, randomization or vision.',
                'Synthetic initial placement is not reachable approach from ordinary reset.',
                'Hard ROM clipping can violate effective position-command acceleration limits despite bounded internal state.',
                'This controller is one bounded diagnostic; failure does not establish task impossibility.',
                'Completing commands or accumulating angular travel does not establish continuous rotation success.'],trials=[])
        def save():
            temp=args.output/'probe.tmp'
            temp.write_text(json.dumps(report,indent=2)+'\n')
            temp.replace(args.output/'probe.json')
        save()
        try:
            for slowdown in args.slowdown:
                for mode,trajectory in trajectories.items():
                    report['trials'].append(dict(mode=mode,**replay(env,trajectory,slowdown)))
                    save()
        except BaseException as error:
            report.update(status='failed',error=repr(error))
            save()
            raise
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps([{k:r[k] for k in ('mode','slowdown','termination','simulated_s',
            'mean_reference_shaft_error_deg','mean_reference_position_error_m',
            'peak_controller_velocity_rad_s','peak_controller_acceleration_rad_s2',
            'effective_command_kinematics')} for r in report['trials']],indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
