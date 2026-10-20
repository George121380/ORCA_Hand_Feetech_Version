"""Bounded IK-targeted physical pinch approaches; not a training reset asset.

IK uses a separate kinematic model. Every actual approach starts from an
ordinary physical settle and runs free object dynamics without state injection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.contact_support import hand_object_contacts
from orca_feetech.manipulation import DT, ManipulationEnv, TaskConfig, rotation_error
from orca_feetech.model import ModelSpec
from orca_feetech.objects import specification
from orca_feetech.reference_control import BrakingActionCodec, CONTROLLER_ID
import probe_physical_trajectory as physical
import probe_reference_hold as hold_probe
import probe_reference_retarget as retarget
from probe_reference_replay import command_kinematics


SEEDS=(1980000,1980001,1980002,1980003)
PAIR_CONFIGS=(('index',1,-.040),('middle',2,-.015))
CLEARANCES=(.002,-.0005)
APPROACH_SECONDS=5.
HOLD_SECONDS=2.


def pinch_targets(tips,position,rotation,finger_index,station,clearance):
    targets=np.asarray(tips,float).copy()
    if targets.shape!=(5,3) or finger_index not in (1,2) or clearance not in CLEARANCES:
        raise ValueError('Expected five tips, index/middle pair and a fixed clearance')
    design=specification('pen')['pen_design']
    if abs(station)>design['shaft_end_m']-.005:
        raise ValueError('Pinch station must lie on the printable cylindrical shaft')
    axis=rotation[:,1]
    reference=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0.,1,0])
    radial=reference-axis*(reference@axis)
    radial/=np.linalg.norm(radial)
    center=position+station*axis
    radius=design['shaft_diameter_m']/2+clearance
    targets[0]=center-radius*radial
    targets[finger_index]=center+radius*radial
    return targets


def pinch_load(env,finger):
    contacts,_=hand_object_contacts(env.model,env.data,env.geoms)
    normal_forces={}
    normals={}
    for c in contacts:
        normal=c['sign']*c['frame'][0]
        force=max(0.,float(c['solver_force_world_n']@normal))
        normal_forces[c['body']]=normal_forces.get(c['body'],0.)+force
        if force>1e-6:
            normals.setdefault(c['body'],[]).append(normal)
    thumb='right_thumb_dp'
    other=f'right_{finger}_ip'
    dots=[float(a@b) for a in normals.get(thumb,[]) for b in normals.get(other,[])]
    weight=float(env.model.body_mass[env.body]*np.linalg.norm(env.model.opt.gravity))
    opposed=bool(normal_forces.get(thumb,0.)>=.05*weight and normal_forces.get(other,0.)>=.05*weight
                 and dots and min(dots)<=-.5)
    return dict(normal_force_by_body_n=normal_forces,
        summed_normal_force_n=sum(normal_forces.values()),minimum_thumb_target_normal_dot=min(dots,default=None),
        loaded_opposition=opposed,object_weight_n=weight)


def approach(env,state,initial_command,target,finger):
    target=np.asarray(target,float)
    if (target.shape!=(16,) or not np.isfinite(target).all()
            or np.any(target<env.lower[:16]) or np.any(target>env.upper[:16])):
        raise ValueError('Expected a finite target inside the original 16-joint ROM')
    if finger not in ('index','middle'):
        raise ValueError('Expected index or middle pinch')
    mujoco.mj_setState(env.model,env.data,state,physical.STATE_SPEC)
    for warning in env.data.warning:
        warning.number=0
        warning.lastinfo=0
    mujoco.mj_forward(env.model,env.data)
    origin,rotation=env.pose()
    initial=hold_probe.snapshot(env,origin,rotation,initial_command)
    initial.update(pinch_load(env,finger))
    codec=BrakingActionCodec(env.config,initial_command)
    self_peak,object_peak,excluded=physical.geometry(env)
    samples=[]
    numerical_failure=None
    start_time=float(env.data.time)
    ticks=round((APPROACH_SECONDS+HOLD_SECONDS)/DT)
    for tick in range(ticks):
        phase=min((tick+1)*DT/APPROACH_SECONDS,1.)
        blend=phase**3*(10+phase*(-15+6*phase))
        desired=initial_command[:16]+blend*(target-initial_command[:16])
        command=codec.step(np.clip((desired-codec.q[:16])/(env.config.max_velocity*DT),-1,1))
        env.data.ctrl[env.actuators]=command
        for _ in range(25):
            mujoco.mj_step(env.model,env.data)
            a,b,bad=physical.geometry(env)
            self_peak=max(self_peak,a)
            object_peak=max(object_peak,b)
            excluded |= bad
        mujoco.mj_forward(env.model,env.data)
        a,b,bad=physical.geometry(env)
        self_peak=max(self_peak,a)
        object_peak=max(object_peak,b)
        excluded |= bad
        if (not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all()
                or abs(env.data.time-start_time-(tick+1)*DT)>1e-8):
            numerical_failure='Non-finite state or solver time reset'
            break
        sample=hold_probe.snapshot(env,origin,rotation,command)
        _,actual_rotation=env.pose()
        sample.update(pinch_load(env,finger),command=command.tolist(),object_rotation=actual_rotation.tolist())
        samples.append(sample)
        if sample['displacement_m']>.12 or sample['object_position_m'][2]<.10 or excluded:
            break
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
    if warnings:
        numerical_failure=numerical_failure or 'MuJoCo solver warning'
    effective=command_kinematics(initial_command,samples,env.config)
    command_ok=bool(effective['peak_velocity_rad_s']<=.8+1e-8 and effective['peak_acceleration_rad_s2']<=2+1e-8)
    maximum_displacement=max(s['displacement_m'] for s in [initial,*samples])
    peak_normal_force=max(s['summed_normal_force_n'] for s in [initial,*samples])
    feasible=bool(len(samples)==ticks and not numerical_failure and not excluded and command_ok
        and self_peak<=.002 and object_peak<=.002 and maximum_displacement<.04
        and peak_normal_force<=10*initial['object_weight_n'])
    last=samples[-10:]
    angular_span=max((float(np.rad2deg(rotation_error(np.array(s['object_rotation']),
        np.array(last[-1]['object_rotation'])))) for s in last),default=None)
    position_span=max((float(np.linalg.norm(np.array(s['object_position_m'])-last[-1]['object_position_m']))
                       for s in last),default=None)
    stable=bool(feasible and len(last)==10 and angular_span<2 and position_span<.002
                and all(s['resting'] and s['loaded_opposition'] for s in last))
    return dict(initial=initial,end=samples[-1] if samples else initial,samples=samples,
        simulated_seconds=float(env.data.time-start_time),complete=len(samples)==ticks,
        numerical_failure=numerical_failure,warnings=warnings,feasible_diagnostic=feasible,
        stable_loaded_pinch=stable,acceptance_met=False,command_limits_met=command_ok,
        effective_command_kinematics=effective,peak_self_penetration_m=self_peak,
        peak_object_penetration_m=object_peak,excluded_object_support=excluded,
        maximum_displacement_m=maximum_displacement,peak_normal_force_sum_n=peak_normal_force,
        final_500ms_orientation_span_deg=angular_span,final_500ms_position_span_m=position_span,
        loaded_opposition_samples=sum(s['loaded_opposition'] for s in samples))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',acceptance_met=False,
        purpose='Fixed IK-directed physical grasp approaches; not RL, reset-asset or deployment acceptance',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_script_sha256=hashlib.sha256(Path(physical.__file__).read_bytes()).hexdigest(),
        retarget_script_sha256=hashlib.sha256(Path(retarget.__file__).read_bytes()).hexdigest(),
        model_sha256=ModelSpec.load().source_sha256,geometry_sha256=hold_probe.geometry_hash(),
        controller=CONTROLLER_ID,runtime=dict(system=platform.system(),python=platform.python_version(),
            mujoco=mujoco.__version__,numpy=np.__version__),
        protocol=dict(seeds=list(SEEDS),extra_physical_hold_seconds=5.,
            pinch_pairs=['thumb/index','thumb/middle'],axial_stations_m=[-.040,-.015],
            clearances_m=list(CLEARANCES),maximum_ik_fits=16,maximum_nfev_per_fit=60,
            approach_seconds=APPROACH_SECONDS,hold_seconds=HOLD_SECONDS,maximum_physical_rollouts=20,
            collisions_audited_every_physics_step_s=.002,force_samples_s=DT,
            loaded_pair_minimum_normal_fraction_weight=.05,opposed_normal_dot_threshold=-.5,
            maximum_total_normal_fraction_weight=10.,maximum_displacement_m=.04,
            static_penetration_allowance_m=.0005,actual_penetration_threshold_m=.002,
            state_injections_during_approach=0,randomization=0.,all_candidates_reported=True),
        limitations=['Distal mesh-cap proxies are not calibrated fleshy contact points.',
            'Fixed initial-pose IK does not track the freely moving object during approach.',
            'Opposed loaded contacts plus stationary hold do not establish force closure or disturbance robustness.',
            'All four resets are development cases, with no independent randomization or real-robot validation.',
            'The force ceilings are diagnostic design choices, not hardware safety guarantees.'],cases=[])
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    hand=retarget.HandKinematics()
    def save():
        physical.save_report(args.output/'probe.json',report)
    save()
    try:
        for seed in SEEDS:
            info,command,preparation=physical.prepare(env,seed,1,5.)
            state=physical.branch_state(env)
            position,rotation=env.pose()
            actual_q=env.data.qpos[env.qadr[:16]].copy()
            tips,proximal=hand.points(actual_q)
            quaternion=np.empty(4)
            mujoco.mju_mat2Quat(quaternion,rotation.ravel())
            case=dict(seed=seed,reset_info=info,preparation=physical.brief(preparation),
                initial_state=state.tolist(),initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                baseline=approach(env,state,command,command[:16],'index'),candidates=[])
            report['cases'].append(case)
            save()
            for finger,index,station in PAIR_CONFIGS:
                for clearance in CLEARANCES:
                    targets=pinch_targets(tips,position,rotation,index,station,clearance)
                    fit,actual_tips=hand.fit(targets,proximal,actual_q,(position,quaternion),self_contact_aware=True)
                    object_depth=float(hand.contact_depths(position,quaternion).max())
                    self_depth=float(hand.self_contact_depths().max())
                    result=approach(env,state,command,fit.x,finger)
                    row=dict(finger=finger,clearance_m=clearance,target_qpos_rad=fit.x.tolist(),
                        fingertip_targets_m=targets.tolist(),optimizer_success=bool(fit.success),
                        optimizer_nfev=int(fit.nfev),tip_errors_m=np.linalg.norm(actual_tips-targets,axis=1).tolist(),
                        static_object_penetration_m=object_depth,static_self_penetration_m=self_depth,
                        physical=result)
                    case['candidates'].append(row)
                    save()
                    print(json.dumps(dict(seed=seed,finger=finger,clearance_m=clearance,
                        static_self=self_depth,static_object=object_depth,
                        loaded_samples=result['loaded_opposition_samples'],stable=result['stable_loaded_pinch'],
                        feasible=result['feasible_diagnostic'])),flush=True)
        runs=[r['physical'] for c in report['cases'] for r in c['candidates']]
        report['summary']=dict(cases=len(report['cases']),candidate_approaches=len(runs),
            static_ik_converged=sum(r['optimizer_success'] for c in report['cases'] for r in c['candidates']),
            stable_loaded_pinches=sum(r['stable_loaded_pinch'] for r in runs),
            feasible_approaches=sum(r['feasible_diagnostic'] for r in runs),
            approaches_with_any_loaded_opposition=sum(r['loaded_opposition_samples']>0 for r in runs),
            baseline_stable_loaded_pinches=sum(c['baseline']['stable_loaded_pinch'] for c in report['cases']),
            all_command_limits_met=all(r['command_limits_met'] for r in [*runs,*[c['baseline'] for c in report['cases']]]),
            numerical_failures=sum(r['numerical_failure'] is not None for r in runs))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        save()
        hand.close()
        env.close()


if __name__=='__main__':
    main()
