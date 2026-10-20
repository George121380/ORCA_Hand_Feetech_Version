"""Bounded nominal-physics hold probe for offline retargeted keyframes.

Each synthetic initial state is inserted once at rest. Only actuator targets
are held afterwards; object/joint states and velocities are never rewritten.
This is not a normal task reset, reference replay, RL or policy acceptance.
Moving reference phases need not be statically stable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.manipulation import (
    ManipulationEnv, RESET_ANGULAR_SPEED, RESET_LINEAR_SPEED, TaskConfig, rotation_error,
)
from orca_feetech.model import JOINT_NAMES
from orca_feetech.objects import geometry_signature


def geometry_hash():
    return hashlib.sha256(json.dumps(geometry_signature('pen'),sort_keys=True).encode()).hexdigest()


def validate_reference(reference,env):
    if (reference.get('status')!='complete' or reference.get('is_physics_rollout') is not False
            or reference.get('model_sha256')!=env.hand_spec.source_sha256
            or reference.get('geometry_sha256')!=geometry_hash()
            or reference.get('joint_names')!=list(JOINT_NAMES)):
        raise ValueError('Reference/model/geometry contract mismatch')
    rows=reference.get('trials',[])
    if not 1<=len(rows)<=64:
        raise ValueError('Expected 1..64 bounded keyframes')
    for row in rows:
        pose=np.r_[row['object_position_m'],row['object_quaternion_wxyz']]
        if pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-5:
            raise ValueError('Invalid object pose')
        for q in (row['qpos'],row['baseline']['qpos']):
            q=np.asarray(q,float)
            if (q.shape!=(16,) or not np.isfinite(q).all()
                    or np.any(q<env.lower[:16]) or np.any(q>env.upper[:16])):
                raise ValueError('Reference joint state outside model ROM')


def snapshot(env,origin,rotation,command):
    pos,rot=env.pose()
    velocity=env.data.qvel[env.ov:env.ov+6]
    penetration=0.
    normal_force=0.
    tower=False
    for i,c in enumerate(env.data.contact):
        if not any(g in (c.geom1,c.geom2) for g in env.geoms) or c.dist>0:
            continue
        other=c.geom2 if c.geom1 in env.geoms else c.geom1
        body=env.model.body(env.model.geom_bodyid[other]).name or ''
        penetration=max(penetration,float(-c.dist))
        tower |= body=='right_tower'
        if body.startswith('right_') and body!='right_tower':
            force=np.zeros(6)
            mujoco.mj_contactForce(env.model,env.data,i,force)
            normal_force+=max(0.,float(force[0]))
    linear=float(np.linalg.norm(velocity[:3]))
    angular=float(np.linalg.norm(velocity[3:]))
    displacement=float(np.linalg.norm(pos-origin))
    shaft_change=float(np.rad2deg(np.arccos(np.clip(rot[:,1]@rotation[:,1],-1,1))))
    retained=bool(env.contact() and normal_force>1e-6 and not tower
                  and pos[2]>.10 and displacement<.04)
    return dict(time_s=float(env.data.time),object_position_m=pos.tolist(),
                displacement_m=displacement,rotation_change_deg=float(np.rad2deg(rotation_error(rot,rotation))),
                shaft_change_deg=shaft_change,
                linear_speed_mps=linear,angular_speed_rps=angular,hand_contact=bool(env.contact()),
                hand_normal_force_sum_n=normal_force,tower_contact=bool(tower),
                deepest_penetration_m=penetration,retained=retained,
                resting=bool(retained and linear<RESET_LINEAR_SPEED and angular<RESET_ANGULAR_SPEED),
                command_error_rms_rad=float(np.sqrt(np.mean((env.data.qpos[env.qadr[:16]]-command[:16])**2))))


def hold(env,q,pose,seconds,*,target=None):
    if not .5<=seconds<=2 or abs(seconds/.05-round(seconds/.05))>1e-8:
        raise ValueError('Use 0.5..2 seconds in 0.05 second intervals')
    actual=env.hand_spec.neutral.copy()
    actual[:16]=q
    command=actual.copy()
    if target is not None:
        target=np.asarray(target,float)
        if (target.shape!=(16,) or not np.isfinite(target).all()
                or np.any(target<env.lower[:16]) or np.any(target>env.upper[:16])):
            raise ValueError('Target must contain 16 finite joint positions within ROM')
        command[:16]=target
    mujoco.mj_resetData(env.model,env.data)
    env.data.qpos[env.qadr]=actual
    env.data.qpos[env.oq:env.oq+7]=pose
    env.data.ctrl[env.actuators]=command
    mujoco.mj_forward(env.model,env.data)
    origin,rotation=env.pose()
    initial=snapshot(env,origin,rotation,command)
    samples=[]
    numerical_failure=None
    for tick in range(round(seconds/.05)):
        mujoco.mj_step(env.model,env.data,nstep=25)
        mujoco.mj_forward(env.model,env.data)
        if (not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all()
                or abs(env.data.time-(tick+1)*.05)>1e-8):
            numerical_failure='Non-finite state or solver time reset'
            break
        samples.append(snapshot(env,origin,rotation,command))
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
    if warnings:
        numerical_failure=numerical_failure or 'MuJoCo solver warning'
    end=samples[-1] if samples else initial
    return dict(initial=initial,end=end,simulated_s=float(env.data.time),samples=samples,
                warnings=warnings,numerical_failure=numerical_failure,
                retained_at_end=bool(not numerical_failure and end['retained']),
                resting_last_200ms=bool(not numerical_failure and len(samples)>=4
                                       and all(x['resting'] for x in samples[-4:])),
                shaft_preserved_last_200ms=bool(not numerical_failure and len(samples)>=4
                    and all(x['resting'] and x['shaft_change_deg']<8 for x in samples[-4:])),
                full_pose_preserved_last_200ms=bool(not numerical_failure and len(samples)>=4
                    and all(x['resting'] and x['rotation_change_deg']<8 for x in samples[-4:])),
                peak_displacement_m=max(x['displacement_m'] for x in [initial,*samples]),
                peak_linear_speed_mps=max(x['linear_speed_mps'] for x in [initial,*samples]),
                peak_angular_speed_rps=max(x['angular_speed_rps'] for x in [initial,*samples]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=1.)
    args=p.parse_args()
    if not .5<=args.seconds<=2 or abs(args.seconds/.05-round(args.seconds/.05))>1e-8:
        p.error('Use 0.5..2 seconds in 0.05 second intervals')
    raw=args.reference.read_bytes()
    reference=json.loads(raw)
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        validate_reference(reference,env)
        args.output.mkdir(parents=True,exist_ok=False)
        report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',
            purpose='Synthetic-at-rest, nominal-physics stationary hold diagnostic; not trajectory or policy acceptance',
            is_physics_rollout=True,acceptance_met=False,randomization_applied=False,
            seconds_per_hold=args.seconds,maximum_holds=128,
            reference_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=reference['source_sha256'],
            source_url=reference.get('source_url'),reference_attribution=reference.get('reference_attribution'),
            model_sha256=env.hand_spec.source_sha256,geometry_sha256=geometry_hash(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime=dict(system=platform.system(),python=platform.python_version(),
                         mujoco=mujoco.__version__,numpy=np.__version__),
            conditions=dict(initial_velocities='Zero only at synthetic initialization',
                            control='Constant nominal position-servo targets; fixed wrist equality',
                            diagnostic_preservation_angle_deg=8.,
                            physics_timestep_s=float(env.model.opt.timestep),sample_interval_s=.05),
            limitations=['No approach from a canonical reset, trajectory replay, retiming, delay or randomization.',
                         'Static hold failure does not prove a moving reference phase is infeasible.',
                         'Summed normal contact forces do not establish force closure.',
                         'A stationary hold is not rotation success or deployment acceptance.'],trials=[])
        def save():
            temporary=args.output/'probe.tmp'
            temporary.write_text(json.dumps(report,indent=2)+'\n')
            temporary.replace(args.output/'probe.json')
        save()
        try:
            for row in reference['trials']:
                pose=np.r_[row['object_position_m'],row['object_quaternion_wxyz']]
                report['trials'].append(dict(frame=row['frame'],time_s=row['time_s'],
                    baseline=hold(env,row['baseline']['qpos'],pose,args.seconds),
                    refined=hold(env,row['qpos'],pose,args.seconds)))
                save()
        except BaseException as error:
            report.update(status='failed',error=repr(error))
            save()
            raise
        report['summary']={}
        for mode in ('baseline','refined'):
            runs=[x[mode] for x in report['trials']]
            report['summary'][mode]=dict(holds=len(runs),
                retained_at_end=sum(x['retained_at_end'] for x in runs),
                resting_last_200ms=sum(x['resting_last_200ms'] for x in runs),
                shaft_preserved_last_200ms=sum(x['shaft_preserved_last_200ms'] for x in runs),
                full_pose_preserved_last_200ms=sum(x['full_pose_preserved_last_200ms'] for x in runs),
                numerical_failures=sum(x['numerical_failure'] is not None for x in runs),
                mean_end_displacement_m=float(np.mean([x['end']['displacement_m'] for x in runs])),
                mean_end_rotation_change_deg=float(np.mean([x['end']['rotation_change_deg'] for x in runs])),
                mean_end_shaft_change_deg=float(np.mean([x['end']['shaft_change_deg'] for x in runs])),
                mean_end_command_error_rms_rad=float(np.mean([x['end']['command_error_rms_rad'] for x in runs])),
                maximum_linear_speed_mps=max(x['peak_linear_speed_mps'] for x in runs),
                maximum_angular_speed_rps=max(x['peak_angular_speed_rps'] for x in runs))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report['summary'],indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
