"""Paired legacy/corrected wrist diagnostics on frozen pinch candidates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.manipulation import ManipulationEnv,TaskConfig
from orca_feetech import wrist_diagnostic
from orca_feetech.wrist_diagnostic import NeutralWristDiagnosticEnv
import probe_grasp_acquisition as grasp
import probe_physical_trajectory as physical
import probe_pinch_workspace as workspace
import probe_reference_hold as hold_probe


def wrist_state(env):
    return dict(actual_rad=float(env.data.qpos[env.qadr[-1]]),
        commanded_rad=float(env.data.ctrl[env.actuators[-1]]),neutral_rad=float(env.hand_spec.neutral[-1]),
        reference_rad=float(env.model.qpos0[env.qadr[-1]]),equality_constant=float(env.model.eq_data[0,0]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raw=args.workspace_report.read_bytes()
    source=json.loads(raw)
    if (source.get('status')!='complete' or source.get('geometry_sha256')!=hold_probe.geometry_hash()
            or len(source.get('selected_indices',[]))>4):
        raise ValueError('Expected a completed bounded workspace report')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',acceptance_met=False,
        purpose='Paired wrist-reference bug diagnosis; corrected branch cannot train or deploy a policy',
        workspace_report_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        correction_module_sha256=hashlib.sha256(Path(wrist_diagnostic.__file__).read_bytes()).hexdigest(),
        model_sha256=source['model_sha256'],geometry_sha256=source['geometry_sha256'],
        documentation='https://mujoco.readthedocs.io/en/3.13.0/XMLreference.html#equality-joint',
        runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
        protocol=dict(candidate_indices=source['selected_indices'],selection='Exactly frozen workspace candidates; no new optimization or selection',
            ordinary_reset_seeds=list(range(1991000,1991004)),randomization=0.,
            legacy_model_unchanged=True,correction='constant = desired neutral - model.qpos0[wrist]',
            manual_insertion_simulated=False),branches=[])
    def save():
        physical.save_report(args.output/'probe.json',report)
    save()
    try:
        for name,env_type in (('legacy_v7',ManipulationEnv),('corrected_diagnostic',NeutralWristDiagnosticEnv)):
            env=env_type(TaskConfig(object='pen',angle=10,randomization=0))
            if env.hand_spec.source_sha256!=source['model_sha256']:
                raise ValueError('Workspace/model source mismatch')
            branch=dict(name=name,model_contract=env.wrist_contract() if name=='corrected_diagnostic' else 'legacy-v7',
                        native_resets=[],manual_trials=[])
            report['branches'].append(branch)
            save()
            try:
                for seed in range(1991000,1991004):
                    info,command,_=physical.prepare(env,seed,1,0.)
                    initial_wrist=wrist_state(env)
                    state=physical.branch_state(env)
                    run=grasp.approach(env,state,command,command[:16],'index')
                    branch['native_resets'].append(dict(seed=seed,reset_info=info,initial_wrist=initial_wrist,
                        end_wrist=wrist_state(env),run=run))
                    save()
                rotation=np.asarray(source['source_rotation'])
                for index in source['selected_indices']:
                    row=source['fits'][index]
                    reach=workspace.empty_reach(env,row['target_qpos_rad'])
                    trial=dict(fit_index=index,empty_reach=reach,reached_wrist=wrist_state(env),
                               release=None,status='empty_reach_failed')
                    branch['manual_trials'].append(trial)
                    save()
                    if reach['reached']:
                        state=workspace.manual_release_state(env,row['position_m'],rotation)
                        a,b,bad=physical.geometry(env)
                        trial['release_initial_geometry']=dict(self_penetration_m=a,object_penetration_m=b,excluded_support=bad)
                        if max(a,b)<=.002 and not bad:
                            trial['release']=grasp.approach(env,state,np.asarray(reach['final_command_rad']),
                                                           row['target_qpos_rad'],row['finger'])
                            trial['status']='complete'
                        else:
                            trial['status']='release_initial_geometry_rejected'
                    save()
            finally:
                env.close()
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        report['summary']=[dict(branch=b['name'],
            native_mean_actual_wrist_rad=float(np.mean([r['end_wrist']['actual_rad'] for r in b['native_resets']])),
            native_mean_final_pose_drift_deg=float(np.mean([r['run']['end']['rotation_change_deg'] for r in b['native_resets']])),
            empty_reaches=sum(t['empty_reach']['reached'] for t in b['manual_trials']),
            release_initial_geometry_passed=sum(t['status']=='complete' for t in b['manual_trials']),
            stable_loaded_pinches=sum(bool(t['release'] and t['release']['stable_loaded_pinch']) for t in b['manual_trials']))
            for b in report['branches']]
        save()
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise


if __name__=='__main__':
    main()
