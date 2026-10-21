"""At most112 physical preparations/rollouts on28 new training states. No retries."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from manipulation import write
from probe_rotation_authority import contacts
from probe_rotation_contact_ik import anchors_from_prefix,propose,run_continuation
from probe_rotation_goal_response import paired_signature
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,PREFIX_SHA256
from orca_feetech.rotation_demonstrations import training_states,demonstration_admission
from orca_feetech.terminal_grip import terminal_knots


ROOT=Path(__file__).resolve().parents[1]


def protocol():
    path=ROOT/'configs/training/pen-rotate-demonstrations-v1.json'
    plan=json.loads(path.read_text())
    if (plan['rotation_fingerprint']!=RotationConfig().fingerprint()
            or plan['prefix_checkpoint_sha256']!=PREFIX_SHA256
            or plan['training_seeds']!=dict(start=3111000,count=28)
            or plan['reserved_development_seeds']!=dict(start=3211000,count=28)
            or plan['reserved_formal_seeds']!=dict(start=3311000,count=100)
            or plan['condition']!=dict(normal_targets=True,lift_m=0.)
            or plan['middle_abd_deltas_rad']!=[0.,-.08,.08]
            or plan['maximum_cases']!=112 or plan['maximum_continuations']!=84
            or plan['maximum_ik_fits']!=112 or plan['max_nfev_per_fit']!=60
            or plan['terminal_knot_weights']!=[0.,0.,.5,1.]
            or plan['knot_seconds']!=[1.25,2.5,3.75,5.]
            or plan['imitation_admission']!=dict(minimum_successful_states=8,
                minimum_states_per_direction=2,minimum_placements=4) or plan['extensions']!=0):
        raise ValueError('Changed demonstration protocol: register a separate experiment')
    return plan,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan,plan_sha=protocol()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cases').mkdir()
    report=dict(status='running',kind='rotation_training_demonstrations_v1',
        plan=plan,plan_sha256=plan_sha,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),is_formal_evaluation=False,
        acceptance_met=False,continuous_rotation_accepted=False,states=[],cases=[],active_case=None)
    def save():
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'probe.json',report)
    def skipped(state,role,reason):
        report['cases'].append(dict(index=len(report['cases']),role=role,**state,
            skipped=True,reason=reason,native_task_success=False))
        save()
    save()
    env=PenRotateEnv(RotationConfig())
    try:
        for state in training_states():
            number=len(report['cases'])
            if number>=plan['maximum_cases']:
                raise RuntimeError('Demonstration case budget exhausted before preparation')
            report['active_case']=dict(index=number,role='planning-prefix',**state)
            save()
            _,initial=env.reset(seed=state['seed'],options=dict(
                placement_index=state['placement_index'],direction=state['direction']))
            signature=paired_signature(env.carrier)
            prep=dict(index=number,role='planning-prefix',**state,initial=initial,prefix_signature=signature,
                qpos=env.carrier.data.qpos.tolist(),qvel=env.carrier.data.qvel.tolist(),
                command=env.carrier.codec.q.tolist(),contacts=contacts(env))
            path=args.output/'cases'/f'case-{number:04d}.json'
            write(path,prep)
            report['cases'].append(dict(prep,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            entry=dict(**state,prefix_signature=signature)
            report['states'].append(entry)
            if not initial['prefix_grasp_success']:
                entry['planning_failure']='grasp_prefix_failed'
            else:
                try:
                    entry['anchors']=anchors_from_prefix(env)
                    entry['proposal']=propose(env,entry['anchors'],plan['condition'])
                except ValueError as error:
                    entry['planning_failure']=repr(error)
            if paired_signature(env.carrier)!=signature:
                raise RuntimeError('Planning changed live physics/controller/camera/RNG/model state')
            initial_command=env.carrier.codec.q[:16].copy()
            lower,upper=env.carrier.lower[:16].copy(),env.carrier.upper[:16].copy()
            save()
            for amount in plan['middle_abd_deltas_rad']:
                role=f'middle-abd-delta-{amount:+.2f}'
                if entry.get('planning_failure'):
                    skipped(state,role,entry['planning_failure'])
                    continue
                delta=np.zeros(16)
                delta[7]=amount
                try:
                    knots=terminal_knots(entry['proposal']['command_knots'],delta,initial_command,lower,upper)
                except ValueError as error:
                    skipped(state,role,repr(error))
                    continue
                run_continuation(env,state,knots,signature,role,args.output,report,save,
                    maximum_cases=plan['maximum_cases'],maximum_continuations=plan['maximum_continuations'])
        if len(report['cases'])!=112 or len(report['states'])!=28:
            raise RuntimeError('Incomplete fixed seed/variant allocation')
        report.update(status='complete',active_case=None,finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(demonstration_admission(report['cases']),allocated_cases=len(report['cases']),
                physical_cases=sum('path' in c for c in report['cases']),
                continuations=sum('end' in c for c in report['cases']),
                skipped_variants=sum(c.get('skipped',False) for c in report['cases']),
                failed_prefix_states=sum(not c['initial']['prefix_grasp_success']
                    for c in report['cases'] if c['role']=='planning-prefix')))
        save()
        print(json.dumps(report['summary'],indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
