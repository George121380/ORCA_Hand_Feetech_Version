"""Bounded v8 native-reset/hold and frozen-grasp placement sensitivity probes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.manipulation import DT,rotation_error
from orca_feetech.manipulation_v8 import ManipulationV8Env,TaskConfigV8,task_contract
import probe_grasp_acquisition as grasp
import probe_physical_trajectory as physical
import probe_pinch_workspace as workspace
import probe_reference_hold as hold_probe
from probe_reference_replay import command_kinematics


OFFSETS=([0,0,0],[.001,0,0],[-.001,0,0],[0,.001,0],[0,-.001,0],[0,0,.001],[0,0,-.001])


def native_hold(env,seed):
    _,initial=env.reset(seed=seed)
    position,rotation=env.pose()
    command=env.codec.q.copy()
    initial_hand_speed=float(np.max(np.abs(env.data.qvel[env.hand_dofs])))
    initial_wrist=float(env.data.qpos[env.qadr[-1]])
    samples=[]
    self_peak,object_peak,excluded=physical.geometry(env)
    first_terminal=None
    for tick in range(140):
        _,_,done,truncated,info=env.step(np.zeros(16))
        a,b,bad=physical.geometry(env)
        self_peak=max(self_peak,a)
        object_peak=max(object_peak,b)
        excluded|=bad
        samples.append(dict(time_s=(tick+1)*DT,command=env.codec.q.tolist(),info=info,
            hand_contact=env.contact(),
            pose_drift_deg=float(np.rad2deg(rotation_error(env.pose()[1],rotation))),
            displacement_m=float(np.linalg.norm(env.pose()[0]-position)),
            wrist_rad=float(env.data.qpos[env.qadr[-1]])))
        if done or truncated:
            first_terminal=first_terminal or dict(tick=tick+1,info=info)
        # A rotation success does not stop this separate hold diagnostic.
        # No physical state, velocity, control or goal is rewritten.
        if info['dropped'] or info['vision_age']>.25:
            break
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
    numerical=bool(warnings or not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all())
    effective=command_kinematics(command,samples,env.config)
    last=samples[-10:]
    # These are diagnostics, not the formal rotational task's success gate.
    retained=bool(len(samples)==140 and not numerical and not excluded and
                  all(not s['info']['dropped'] for s in samples) and all(s['hand_contact'] for s in last))
    held=bool(retained and self_peak<=.002 and object_peak<=.002 and
        max(s['displacement_m'] for s in samples)<.04 and
        all(s['pose_drift_deg']<8 and s['info']['linear_speed_mps']<.015
            and s['info']['angular_speed_rps']<.3 for s in last))
    return dict(seed=seed,randomization=env.config.randomization,initial=initial,
        initial_actual_hand_max_speed_rad_s=initial_hand_speed,initial_actual_wrist_rad=initial_wrist,
        samples=samples,first_task_terminal=first_terminal,full_7s_retained=retained,held_diagnostic=held,
        peak_self_penetration_m=self_peak,peak_object_penetration_m=object_peak,excluded_support=excluded,
        effective_command_kinematics=effective,numerical_failure=numerical,warnings=warnings,
        final_pose_drift_deg=samples[-1]['pose_drift_deg'],acceptance_met=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.workspace_report.read_bytes()
    source=json.loads(raw)
    if (source.get('status')!='complete' or source.get('geometry_sha256')!=hold_probe.geometry_hash()
            or len(source.get('selected_indices',[]))!=2):
        raise ValueError('Expected the two frozen refined-workspace candidates')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),acceptance_met=False,
        contract=task_contract(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        workspace_report_sha256=hashlib.sha256(raw).hexdigest(),model_sha256=source['model_sha256'],
        geometry_sha256=source['geometry_sha256'],runtime=dict(system=platform.system(),
            python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
        protocol=dict(native_seeds=list(range(2001000,2001008)),randomization=[0.,.3,1.],
            hold_seconds=7.,manual_offsets_m=OFFSETS,manual_randomization=0.,
            manual_candidate_indices=source['selected_indices'],maximum_native_holds=24,maximum_manual_releases=14,
            geometry_audit='Native hold at 20Hz; manual release every 2ms',
            continued_after_rotation_success='Hold diagnostic only; record first terminal, stop on drop/vision timeout',
            manual_insertion_simulated=False,training_or_deployment_acceptance=False),
        native=[],manual=[])
    def save():
        physical.save_report(args.output/'probe.json',report)
    save()
    try:
        for strength in (0.,.3,1.):
            env=ManipulationV8Env(TaskConfigV8(object='pen',angle=10,randomization=strength))
            if env.hand_spec.source_sha256!=source['model_sha256']:
                raise ValueError('Frozen model source mismatch')
            try:
                for seed in range(2001000,2001008):
                    report['native'].append(native_hold(env,seed))
                    save()
            finally:
                env.close()
        env=ManipulationV8Env(TaskConfigV8(object='pen',angle=10,randomization=0))
        try:
            for index in source['selected_indices']:
                row=source['fits'][index]
                reach=workspace.empty_reach(env,row['target_qpos_rad'])
                reached_state=physical.branch_state(env)
                for offset in OFFSETS:
                    trial=dict(fit_index=index,offset_m=offset,empty_reach_passed=reach['reached'],
                        release=None,status='empty_reach_failed')
                    report['manual'].append(trial)
                    if reach['reached']:
                        mujoco.mj_setState(env.model,env.data,reached_state,physical.STATE_SPEC)
                        mujoco.mj_forward(env.model,env.data)
                        state=workspace.manual_release_state(env,np.asarray(row['position_m'])+offset,
                                                            source['source_rotation'])
                        a,b,bad=physical.geometry(env)
                        trial['initial_geometry']=dict(self_penetration_m=a,object_penetration_m=b,excluded_support=bad)
                        if max(a,b)<=.002 and not bad:
                            trial['release']=grasp.approach(env,state,np.asarray(reach['final_command_rad']),
                                                           row['target_qpos_rad'],row['finger'])
                            trial['status']='complete'
                        else:
                            trial['status']='initial_geometry_rejected'
                    save()
        finally:
            env.close()
        report['summary']=dict(native=[dict(randomization=s,cases=8,
            full_7s_retained=sum(r['full_7s_retained'] for r in report['native'] if r['randomization']==s),
            held_diagnostic=sum(r['held_diagnostic'] for r in report['native'] if r['randomization']==s),
            mean_final_drift_deg=float(np.mean([r['final_pose_drift_deg'] for r in report['native'] if r['randomization']==s])))
            for s in (0.,.3,1.)],manual_releases_completed=sum(r['status']=='complete' for r in report['manual']),
            manual_stable_loaded_pinches=sum(bool(r['release'] and r['release']['stable_loaded_pinch']) for r in report['manual']))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise


if __name__=='__main__':
    main()
