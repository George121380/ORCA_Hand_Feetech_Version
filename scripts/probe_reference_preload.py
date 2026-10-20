"""Bounded position-target preload fit and independent nominal physical holds.

Fits commands at fixed synthetic geometry, then releases all physical degrees
of freedom for validation. This is neither a reachable reset nor a policy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import scipy
from scipy.optimize import least_squares

from orca_feetech import contact_support
from orca_feetech.contact_support import gravity_support, hand_object_contacts
from orca_feetech.manipulation import ManipulationEnv, TaskConfig
from probe_reference_hold import geometry_hash, hold, validate_reference


BIAS_LIMIT=.15
REGULARIZATION=.02
MAX_NFEV=60
MAX_FORWARD_CALLS=1200


def instantaneous(env,q,pose,target):
    """Deterministic forward solve only: no physics time integration."""
    actual=env.hand_spec.neutral.copy()
    actual[:16]=q
    command=actual.copy()
    command[:16]=target
    mujoco.mj_resetData(env.model,env.data)
    env.data.qpos[env.qadr]=actual
    env.data.qpos[env.oq:env.oq+7]=pose
    env.data.ctrl[env.actuators]=command
    mujoco.mj_forward(env.model,env.data)
    contacts,excluded=hand_object_contacts(env.model,env.data,env.geoms)
    wrench=np.zeros(6)
    center=env.data.xipos[env.body]
    for contact in contacts:
        force=contact['solver_force_world_n']
        wrench+=np.r_[force,contact['solver_torque_world_nm']
                      +np.cross(contact['position_m']-center,force)]
    mass=env.model.body_mass[env.body]
    expected=np.r_[-mass*env.model.opt.gravity,np.zeros(3)]
    residual=(wrench-expected)*np.r_[np.ones(3),np.ones(3)/.1]/np.linalg.norm(expected[:3])
    if (excluded or not np.isfinite(residual).all() or not np.isfinite(env.data.qacc).all()
            or any(w.number for w in env.data.warning)):
        raise ValueError('Unexpected excluded support or non-finite static solve')
    return residual,contacts


def solve_preload(env,q,pose):
    q=np.asarray(q,float)
    calls=0
    def objective(target):
        nonlocal calls
        calls+=1
        if calls>MAX_FORWARD_CALLS:
            raise RuntimeError('Static forward-call budget exceeded')
        residual,_=instantaneous(env,q,pose,target)
        return np.r_[residual,REGULARIZATION*(target-q)/BIAS_LIMIT]
    initial=objective(q)
    fit=least_squares(objective,q,bounds=(np.maximum(env.lower[:16],q-BIAS_LIMIT),
        np.minimum(env.upper[:16],q+BIAS_LIMIT)),max_nfev=MAX_NFEV,
        ftol=1e-8,xtol=1e-8,gtol=1e-8)
    # Selection uses only the predeclared static objective, never hold results.
    target=fit.x if np.linalg.norm(fit.fun)<np.linalg.norm(initial) else q.copy()
    final=objective(target)
    _,contacts=instantaneous(env,q,pose,target)
    support=gravity_support(contacts,env.data.xipos[env.body],
        env.model.body_mass[env.body],env.model.opt.gravity)
    return dict(target_rad=target.tolist(),bias_rad=(target-q).tolist(),
        maximum_absolute_bias_rad=float(np.max(np.abs(target-q))),
        optimizer_success=bool(fit.success),optimizer_status=int(fit.status),
        optimizer_message=str(fit.message),optimizer_nfev=int(fit.nfev),
        objective_forward_calls=calls,total_forward_calls=calls+1,
        initial_objective_norm=float(np.linalg.norm(initial)),
        selected_objective_norm=float(np.linalg.norm(final)),
        initial_object_wrench_relative_residual=float(np.linalg.norm(initial[:6])),
        selected_object_wrench_relative_residual=float(np.linalg.norm(final[:6])),
        instantaneous_hand_acceleration_max_rad_s2=float(np.max(np.abs(env.data.qacc[
            env.model.jnt_dofadr[env.model.actuator_trnid[env.actuators[:16],0]]]))),
        instantaneous_object_linear_acceleration_m_s2=float(np.linalg.norm(env.data.qacc[env.ov:env.ov+3])),
        instantaneous_object_angular_acceleration_rad_s2=float(np.linalg.norm(env.data.qacc[env.ov+3:env.ov+6])),
        selected_support=support)


def validate_inputs(reference_raw,holds_raw,audit_raw):
    holds=json.loads(holds_raw)
    audit=json.loads(audit_raw)
    reference=json.loads(reference_raw)
    reference_hash=hashlib.sha256(reference_raw).hexdigest()
    if (holds.get('status')!='complete' or audit.get('status')!='complete'
            or holds.get('reference_sha256')!=reference_hash
            or audit.get('reference_sha256')!=reference_hash
            or audit.get('hold_report_sha256')!=hashlib.sha256(holds_raw).hexdigest()):
        raise ValueError('Frozen audit/hold reports must match the reference exactly')
    frames=[r['frame'] for r in reference['trials']]
    for report in (reference,holds,audit):
        other=[r['frame'] for r in report['trials']]
        if len(set(other))!=len(other) or set(other)!=set(frames):
            raise ValueError('Duplicate or mismatched frame sets')
    return reference,holds,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--hold-report',type=Path,required=True)
    p.add_argument('--support-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raw,hold_raw,audit_raw=(p.read_bytes() for p in (args.reference,args.hold_report,args.support_audit))
    reference,holds,audit=validate_inputs(raw,hold_raw,audit_raw)
    selected={r['frame'] for r in audit['trials'] if r['refined']['gravity_wrench_feasible_5pct']}
    if not selected:
        raise ValueError('No cone-feasible refined frames in the frozen audit')
    frozen={r['frame']:r['refined'] for r in holds['trials']}
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        validate_reference(reference,env)
        if (audit.get('model_sha256')!=env.hand_spec.source_sha256
                or audit.get('geometry_sha256')!=geometry_hash()):
            raise ValueError('Audit model/geometry contract mismatch')
        args.output.mkdir(parents=True,exist_ok=False)
        report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',
            purpose='Bounded synthetic preload diagnostic, not rotation, reachable reset or policy acceptance',
            is_physics_rollout=True,acceptance_met=False,randomization_applied=False,
            reference_sha256=hashlib.sha256(raw).hexdigest(),
            hold_report_sha256=hashlib.sha256(hold_raw).hexdigest(),
            support_audit_sha256=hashlib.sha256(audit_raw).hexdigest(),
            model_sha256=env.hand_spec.source_sha256,geometry_sha256=geometry_hash(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            hold_script_sha256=hashlib.sha256(Path(hold.__code__.co_filename).read_bytes()).hexdigest(),
            support_module_sha256=hashlib.sha256(Path(contact_support.__file__).read_bytes()).hexdigest(),
            source_url=reference.get('source_url'),reference_attribution=reference.get('reference_attribution'),
            runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__,
                         numpy=np.__version__,scipy=scipy.__version__),
            protocol=dict(selection='All refined frames marked gravity_wrench_feasible_5pct in frozen audit',
                selected_frames=sorted(selected),bias_limit_rad=BIAS_LIMIT,
                objective='gravity-wrench residual (torque / 0.1m), divided by weight; append 0.02*bias/0.15',
                max_nfev_per_frame=MAX_NFEV,max_objective_forward_calls_per_frame=MAX_FORWARD_CALLS,
                numerical_jacobian_calls_counted_separately=True,optimizer_initialization='One start: zero bias',
                hold_seconds=1.,hold_control='Constant position targets, applied at synthetic initialization',
                initial_actual_state='Same reference joint/object positions; zero velocity only at initialization',
                state_rewrites_during_hold=False,external_wrenches_applied=False,
                actuator_gains_forcelimits_and_physics_unchanged=True,
                position_gains=env.model.actuator_gainprm[env.actuators,0].tolist(),
                actuator_force_ranges=env.model.actuator_forcerange[env.actuators].tolist(),
                physics_timestep_s=float(env.model.opt.timestep)),
            limitations=['Static objective does not constrain hand acceleration or ensure equilibrium.',
                'Synthetic initial preload is not a reachable approach or rate-limited command trajectory.',
                'No delay, randomization, hardware action, trained policy or acceptance claim.'],trials=[])
        def save():
            temp=args.output/'probe.tmp'
            temp.write_text(json.dumps(report,indent=2)+'\n')
            temp.replace(args.output/'probe.json')
        save()
        try:
            for row in reference['trials']:
                if row['frame'] not in selected:
                    continue
                q=np.asarray(row['qpos'])
                pose=np.r_[row['object_position_m'],row['object_quaternion_wxyz']]
                fit=solve_preload(env,q,pose)
                baseline=hold(env,q,pose,1.)
                optimized=hold(env,q,pose,1.,target=fit['target_rad'])
                report['trials'].append(dict(frame=row['frame'],actual_qpos_rad=q.tolist(),
                    object_pose_wxyz=pose.tolist(),fit=fit,baseline=baseline,preloaded=optimized,
                    frozen_baseline={k:frozen[row['frame']][k] for k in
                        ('retained_at_end','resting_last_200ms','shaft_preserved_last_200ms','full_pose_preserved_last_200ms')}))
                save()
                print(json.dumps(dict(frame=row['frame'],completed=len(report['trials']),
                    residual=fit['selected_object_wrench_relative_residual'],
                    baseline_shaft_held=baseline['shaft_preserved_last_200ms'],
                    preloaded_shaft_held=optimized['shaft_preserved_last_200ms'])),flush=True)
        except BaseException as error:
            report.update(status='failed',error=repr(error))
            save()
            raise
        report['summary']=dict(frames=len(selected),
            frozen_baseline_outcome_mismatches=sum(any(r['baseline'][k]!=value for k,value in
                r['frozen_baseline'].items()) for r in report['trials']),
            optimizer_converged=sum(r['fit']['optimizer_success'] for r in report['trials']),
            initial_wrench_within_5pct=sum(r['fit']['initial_object_wrench_relative_residual']<=.05 for r in report['trials']),
            selected_wrench_within_5pct=sum(r['fit']['selected_object_wrench_relative_residual']<=.05 for r in report['trials']),
            median_initial_wrench_residual=float(np.median([r['fit']['initial_object_wrench_relative_residual'] for r in report['trials']])),
            median_selected_wrench_residual=float(np.median([r['fit']['selected_object_wrench_relative_residual'] for r in report['trials']])))
        for mode in ('baseline','preloaded'):
            runs=[r[mode] for r in report['trials']]
            report['summary'][mode]={k:sum(r[k] for r in runs) for k in
                ('retained_at_end','resting_last_200ms','shaft_preserved_last_200ms','full_pose_preserved_last_200ms')}
            report['summary'][mode].update(numerical_failures=sum(r['numerical_failure'] is not None for r in runs),
                mean_end_shaft_change_deg=float(np.mean([r['end']['shaft_change_deg'] for r in runs])),
                mean_end_displacement_m=float(np.mean([r['end']['displacement_m'] for r in runs])))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report['summary'],indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
