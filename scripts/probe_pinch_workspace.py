"""Bounded pinch placement/IK-prior scan and separate manual-release trials.

An empty hand physically reaches each frozen candidate. A new trial then
initializes a manually placed object once, preserving the actual hand state.
This does not model human insertion, autonomous pickup or a deployable policy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.manipulation import DT, ManipulationEnv, TaskConfig
from orca_feetech.model import ModelSpec
from orca_feetech.reference_control import BrakingActionCodec, CONTROLLER_ID
import probe_grasp_acquisition as grasp
import probe_physical_trajectory as physical
import probe_reference_hold as hold_probe
import probe_reference_retarget as retarget
from probe_reference_replay import command_kinematics


SOURCE_SEED=1990000
GRID_X=(.16,.18,.20,.22)
GRID_Z=(.17,.19,.21)
PROXIMAL_WEIGHTS=(.25,0.)
MAX_RELEASE_CANDIDATES=4


def empty_reach(env,target):
    target=np.asarray(target,float)
    if (target.shape!=(16,) or not np.isfinite(target).all()
            or np.any(target<env.lower[:16]) or np.any(target>env.upper[:16])):
        raise ValueError('Expected a finite 16-joint target within ROM')
    # This is an independent empty-hand preparation episode. The unused free
    # object falls onto the floor far outside the hand, with unchanged physics.
    env.data=mujoco.MjData(env.model)
    command=env.hand_spec.neutral.copy()
    env.data.qpos[env.qadr]=command
    env.data.ctrl[env.actuators]=command
    env.data.qpos[env.oq:env.oq+7]=[.6,0,.8,1,0,0,0]
    mujoco.mj_forward(env.model,env.data)
    codec=BrakingActionCodec(env.config,command)
    samples=[]
    self_peak=0.
    numerical_failure=None
    for tick in range(130):  # 0.5s neutral, 5s approach, 1s hold.
        phase=float(np.clip(((tick+1)*DT-.5)/5.,0,1))
        blend=phase**3*(10+phase*(-15+6*phase))
        desired=command[:16]+blend*(target-command[:16])
        actual_command=codec.step(np.clip((desired-codec.q[:16])/(env.config.max_velocity*DT),-1,1))
        env.data.ctrl[env.actuators]=actual_command
        for _ in range(25):
            mujoco.mj_step(env.model,env.data)
            self_peak=max(self_peak,max((c['penetration_m'] for c in
                retarget.hand_self_contacts(env.model,env.data)),default=0.))
        mujoco.mj_forward(env.model,env.data)
        self_peak=max(self_peak,max((c['penetration_m'] for c in
            retarget.hand_self_contacts(env.model,env.data)),default=0.))
        if (not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all()
                or abs(env.data.time-(tick+1)*DT)>1e-8):
            numerical_failure='Non-finite state or solver time reset'
            break
        samples.append(dict(time_s=float(env.data.time),command=actual_command.tolist(),
            actual_hand_qpos_rad=env.data.qpos[env.qadr].tolist()))
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
    if warnings:
        numerical_failure=numerical_failure or 'MuJoCo solver warning'
    effective=command_kinematics(command,samples,env.config)
    command_ok=bool(effective['peak_velocity_rad_s']<=.8+1e-8 and effective['peak_acceleration_rad_s2']<=2+1e-8)
    dofs=env.model.jnt_dofadr[env.model.actuator_trnid[env.actuators,0]]
    qvel=env.data.qvel[dofs].copy()
    error=float(np.sqrt(np.mean((env.data.qpos[env.qadr[:16]]-target)**2)))
    reached=bool(len(samples)==130 and not numerical_failure and self_peak<=.002 and command_ok
                 and error<.03 and np.max(np.abs(qvel[:16]))<.1)
    return dict(reached=reached,acceptance_met=False,samples=samples,
        simulated_seconds=float(env.data.time),peak_self_penetration_m=self_peak,
        numerical_failure=numerical_failure,warnings=warnings,command_limits_met=command_ok,
        effective_command_kinematics=effective,target_error_rms_rad=error,
        final_command_rad=codec.q.tolist(),final_hand_qpos_rad=env.data.qpos[env.qadr].tolist(),
        final_hand_qvel_rad_s=qvel.tolist(),unused_object_kept_outside_hand=True)


def manual_release_state(env,position,rotation):
    position=np.asarray(position,float)
    rotation=np.asarray(rotation,float)
    if (position.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(position).all()
            or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7)
            or abs(np.linalg.det(rotation)-1)>1e-7):
        raise ValueError('Expected a finite position and proper rotation')
    old=physical.branch_state(env)
    hand_q=env.data.qpos[env.qadr].copy()
    dofs=env.model.jnt_dofadr[env.model.actuator_trnid[env.actuators,0]]
    hand_velocity=env.data.qvel[dofs].copy()
    # A NEW manually placed/released-object episode, not a correction to an
    # object trajectory already in progress. Preserve the actually reached hand.
    env.data=mujoco.MjData(env.model)
    mujoco.mj_setState(env.model,env.data,old,physical.STATE_SPEC)
    quaternion=np.empty(4)
    mujoco.mju_mat2Quat(quaternion,rotation.ravel())
    env.data.qpos[env.oq:env.oq+7]=np.r_[position,quaternion]
    env.data.qvel[env.ov:env.ov+6]=0.  # Human-supported object is released at rest.
    env.data.qacc_warmstart[:]=0.  # New contacts: clear only numerical warmstart.
    mujoco.mj_forward(env.model,env.data)
    if (not np.array_equal(hand_q,env.data.qpos[env.qadr])
            or not np.array_equal(hand_velocity,env.data.qvel[dofs])):
        raise RuntimeError('Manual release initialization changed the reached hand')
    return physical.branch_state(env)


def select_candidates(rows):
    selected=[]
    positions=set()
    for index,row in enumerate(rows):
        key=tuple(row['position_m'])
        if row['static_screen_passed'] and key not in positions:
            selected.append(index)
            positions.add(key)
            if len(selected)==MAX_RELEASE_CANDIDATES:
                break
    return selected


def refinement_rows(previous):
    return [r for r in previous['fits'] if not r['optimizer_success'] and r['optimizer_nfev']==60
        and r['selected_tip_max_error_m']<=.002 and r['static_object_penetration_m']<=.002
        and r['static_self_penetration_m']<=.002][:4]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refine-report',type=Path,
                   help='Retry at most four geometrically screened, budget-exhausted fits at fixed 120-nfev limit')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',acceptance_met=False,
        purpose='Bounded placement/prior ablation and separate manual-release diagnostics; no training acceptance',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        retarget_script_sha256=hashlib.sha256(Path(retarget.__file__).read_bytes()).hexdigest(),
        grasp_script_sha256=hashlib.sha256(Path(grasp.__file__).read_bytes()).hexdigest(),
        model_sha256=ModelSpec.load().source_sha256,geometry_sha256=hold_probe.geometry_hash(),controller=CONTROLLER_ID,
        runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
        protocol=dict(source_seed=SOURCE_SEED,source_extra_hold_seconds=5.,grid_x_m=list(GRID_X),grid_z_m=list(GRID_Z),
            y_and_rotation='Frozen from source physical reset',proximal_weights=list(PROXIMAL_WEIGHTS),
            clearance_m=-.0005,maximum_ik_fits=52,maximum_nfev_per_fit=60,
            static_screen='Converged; BOTH selected tip residuals <=2mm, hand/self and object overlap <=2mm; no tower',
            selection='First four screened rows at distinct positions in fixed current/grid-x/grid-z/finger/weight order, frozen before physics',
            maximum_empty_reaches=4,empty_reach_seconds=6.5,maximum_manual_release_trials=4,release_seconds=7.,
            actual_release_object_initialized_once=True,actual_reached_hand_preserved=True,
            model_or_collision_filters_changed=False,randomization=0.),
        limitations=['One-start soft IK and a coarse grid do not establish the full shared workspace or global impossibility.',
            'Manual insertion is not simulated; empty reach and release are distinct episodes.',
            'An initialized object can overlap soft collision surfaces by up to the diagnostic threshold; this is not proof of human placeability.',
            'A stable loaded pinch is not force closure, a rotation skill or a validated training reset asset.'],
        fits=[],selected_indices=[],physical_trials=[])
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    hand=retarget.HandKinematics()
    def save():
        physical.save_report(args.output/'probe.json',report)
    save()
    try:
        previous=None
        if args.refine_report:
            raw=args.refine_report.read_bytes()
            previous=json.loads(raw)
            if (previous.get('status')!='complete' or previous.get('model_sha256')!=report['model_sha256']
                    or previous.get('geometry_sha256')!=report['geometry_sha256']
                    or previous.get('protocol',{}).get('maximum_nfev_per_fit')!=60):
                raise ValueError('Expected a matching completed 60-nfev workspace report')
            source_position=np.asarray(previous['source_position_m'])
            rotation=np.asarray(previous['source_rotation'])
            q=np.asarray(previous['source_hand_qpos_rad'])
            info=previous['source_reset_info']
            preparation=previous['source_preparation']
            report['refinement_input_sha256']=hashlib.sha256(raw).hexdigest()
            report['protocol'].update(maximum_ik_fits=4,maximum_nfev_per_fit=120,
                refinement='Only original <=2mm geometries with 60-nfev exhaustion; same original start/objective, no extra starts')
        else:
            info,_,preparation=physical.prepare(env,SOURCE_SEED,1,5.)
            source_position,rotation=env.pose()
            q=env.data.qpos[env.qadr[:16]].copy()
        tips,proximal=hand.points(q)
        quaternion=np.empty(4)
        mujoco.mju_mat2Quat(quaternion,rotation.ravel())
        report.update(source_reset_info=info,source_preparation=physical.brief(preparation),
            source_position_m=source_position.tolist(),source_rotation=rotation.tolist(),source_hand_qpos_rad=q.tolist())
        tasks=[]
        if previous is not None:
            for row in refinement_rows(previous):
                index=1 if row['finger']=='index' else 2
                tasks.append((row['kind'],np.asarray(row['position_m']),row['finger'],index,
                              np.asarray(row['target_tips_m']),row['proximal_weight']))
        else:
            placements=[('current',source_position.copy())]+[('grid',np.array([x,source_position[1],z])) for x in GRID_X for z in GRID_Z]
            for kind,position in placements:
                for finger,index,station in grasp.PAIR_CONFIGS:
                    targets=grasp.pinch_targets(tips,position,rotation,index,station,-.0005)
                    for weight in PROXIMAL_WEIGHTS:
                        tasks.append((kind,position,finger,index,targets,weight))
        for kind,position,finger,index,targets,weight in tasks:
            fit,actual=hand.fit(targets,proximal,q,(position,quaternion),
                self_contact_aware=True,proximal_weight=weight,
                optimizer_max_nfev=report['protocol']['maximum_nfev_per_fit'])
            object_depth=float(hand.contact_depths(position,quaternion).max())
            self_depth=float(hand.self_contact_depths().max())
            errors=np.linalg.norm(actual-targets,axis=1)
            tower=any(c['body']=='right_tower' for c in hand.contacts(position,quaternion))
            passed=bool(fit.success and max(errors[0],errors[index])<=.002
                        and object_depth<=.002 and self_depth<=.002 and not tower)
            report['fits'].append(dict(kind=kind,position_m=position.tolist(),finger=finger,
                proximal_weight=weight,target_qpos_rad=fit.x.tolist(),target_tips_m=targets.tolist(),
                tip_errors_m=errors.tolist(),selected_tip_max_error_m=float(max(errors[0],errors[index])),
                optimizer_success=bool(fit.success),optimizer_nfev=int(fit.nfev),
                static_object_penetration_m=object_depth,static_self_penetration_m=self_depth,
                static_screen_passed=passed))
            save()
        report['selected_indices']=select_candidates(report['fits'])
        save()
        for index in report['selected_indices']:
            row=report['fits'][index]
            target=np.asarray(row['target_qpos_rad'])
            reach=empty_reach(env,target)
            trial=dict(fit_index=index,empty_reach=reach,release=None,status='empty_reach_failed')
            report['physical_trials'].append(trial)
            save()
            if reach['reached']:
                state=manual_release_state(env,row['position_m'],rotation)
                a,b,bad=physical.geometry(env)
                trial['release_initial_geometry']=dict(self_penetration_m=a,object_penetration_m=b,excluded_support=bad)
                trial['release_initial_state_sha256']=hashlib.sha256(state.tobytes()).hexdigest()
                if max(a,b)<=.002 and not bad:
                    trial['release']=grasp.approach(env,state,np.asarray(reach['final_command_rad']),target,row['finger'])
                    trial['status']='complete'
                else:
                    trial['status']='release_initial_geometry_rejected'
            save()
            print(json.dumps(dict(fit_index=index,status=trial['status'],stable=bool(trial['release'] and trial['release']['stable_loaded_pinch']))),flush=True)
        report['summary']=dict(fits=len(report['fits']),static_screen_passed=sum(r['static_screen_passed'] for r in report['fits']),
            screened_current=sum(r['static_screen_passed'] and r['kind']=='current' for r in report['fits']),
            screened_anchored=sum(r['static_screen_passed'] and r['proximal_weight']==.25 for r in report['fits']),
            screened_no_proximal_prior=sum(r['static_screen_passed'] and r['proximal_weight']==0 for r in report['fits']),
            frozen_selected=len(report['selected_indices']),empty_reaches_passed=sum(t['empty_reach']['reached'] for t in report['physical_trials']),
            release_trials_completed=sum(t['status']=='complete' for t in report['physical_trials']),
            stable_loaded_releases=sum(bool(t['release'] and t['release']['stable_loaded_pinch']) for t in report['physical_trials']))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        env.close()
        hand.close()


if __name__=='__main__':
    main()
