"""Isolated2187-posture screen and at most7 actual preparation/rollout cases.

Refine one frozen partial development trajectory; no native gate changes,
live snapshot injection, neural training, automatic extension or hardware.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from orca_feetech.contact_support import hand_object_contacts
from orca_feetech.model import JOINT_NAMES
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,PREFIX_SHA256
from orca_feetech.rotation_trajectory import trajectory_cost
from orca_feetech.terminal_grip import terminal_knots,static_grip_score
from manipulation import write
from probe_rotation_contact_ik import hypothetical_geometry,run_continuation
from probe_rotation_goal_response import paired_signature


ROOT=Path(__file__).resolve().parents[1]
NAMES=('thumb_mcp','thumb_abd','thumb_pip','thumb_dip','middle_abd','middle_mcp','middle_pip')
PARENT_SHA='78308c3fcc376e1f18fa3d663da0076ed971a187df12ded7f1410c7c8a0dd274'


def normal_measurement(carrier,data):
    rows,excluded=hand_object_contacts(carrier.model,data,carrier.geoms)
    groups={body:[] for body in ('right_thumb_dp','right_middle_ip')}
    forces={body:0. for body in groups}
    for c in rows:
        if c['body'] not in groups:
            continue
        normal=c['sign']*c['frame'][0]
        force=max(0.,float(c['solver_force_world_n']@normal))
        if force>1e-6:
            groups[c['body']].append(normal)
            forces[c['body']]+=force
    dots=[float(a@b) for a in groups['right_thumb_dp'] for b in groups['right_middle_ip']]
    return dict(normal_dot=min(dots,default=None),normal_force_n=forces,excluded_bodies=excluded)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan_path=ROOT/'configs/training/pen-rotate-terminal-grip-diagnostic-v1.json'
    plan=json.loads(plan_path.read_text())
    if (plan['parent_report_sha256']!=PARENT_SHA or plan['rotation_fingerprint']!=RotationConfig().fingerprint()
            or plan['prefix_checkpoint_sha256']!=PREFIX_SHA256 or plan['source_case']!=15
            or plan['source_seed']!=2511002 or plan['source_placement_index']!=2
            or tuple(plan['grid_joint_names'])!=NAMES or plan['grid_values_rad']!=[-.08,0.,.08]
            or plan['maximum_static_postures']!=2187 or plan['maximum_candidates']!=3
            or plan['maximum_cases']!=7 or plan['maximum_continuations']!=6
            or plan['terminal_knot_weights']!=[0.,0.,.5,1.] or plan['total_command_delta_limit_rad']!=.3
            or plan['extensions']!=0
            or [(s['seed'],s['placement_index'],s['direction']) for s in plan['transfers']]
            !=[(2511000,0,1),(2511001,1,1)]):
        raise ValueError('Changed terminal refinement protocol; register another experiment')
    raw=args.parent_report.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:
        raise ValueError('Require immutable contact-IK report')
    parent=json.loads(raw)
    source_brief=parent['cases'][15]
    source_path=args.parent_report.parent/'cases/case-0015.json'
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=source_brief['sha256']:
        raise ValueError('Frozen source trajectory hash mismatch')
    source=json.loads(source_path.read_text())
    if (source['seed']!=2511002 or source['role']!='points-normals-center-fixed'
            or source['native_task_success'] or not source['full_uncensored_duration']):
        raise ValueError('Source is not the predeclared partial event')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cases').mkdir()
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),plan=plan,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),source_case_sha256=source_brief['sha256'],
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acceptance_met=False,continuous_rotation_accepted=False,cases=[],selected_static=[],
        candidate_results=[],transfers=[],selected_physical=None,active_case=None)
    def save():
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'probe.json',report)
    save()
    env=PenRotateEnv(RotationConfig())
    carrier=env.carrier
    states={s['seed']:s for s in parent['states']}
    state=dict(seed=2511002,placement_index=2,direction=1)
    def execute(knots,current_state,role,delta):
        row=run_continuation(env,current_state,knots,states[current_state['seed']]['prefix_signature'],
            role,args.output,report,save,maximum_cases=7,maximum_continuations=6)
        data=json.loads(Path(row['path']).read_text())
        samples=[dict(s,handoff_displacement_m=float(np.linalg.norm(np.asarray(s['position'])-data['start_position'])))
                 for s in data['samples']]
        row['selection_score']=trajectory_cost(samples,delta,complete=row['full_uncensored_duration'],
                                                command_limits_met=row['command_limits_met'])
        save()
        return row
    try:
        report['active_case']=dict(index=0,role='static-model-prefix',**state)
        save()
        obs,initial=env.reset(seed=2511002,options=dict(placement_index=2,direction=1))
        signature=paired_signature(carrier)
        prep=dict(index=0,role='static-model-prefix',**state,initial=initial,prefix_signature=signature,
            qpos_all=carrier.data.qpos.tolist(),qvel_all=carrier.data.qvel.tolist(),command=carrier.codec.q.tolist())
        path=args.output/'cases/case-0000.json'
        write(path,prep)
        report['cases'].append(dict(prep,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if not initial['prefix_grasp_success'] or signature!=states[2511002]['prefix_signature']:
            raise RuntimeError('Physical source preparation differs from parent; no retry')
        scratch=mujoco.MjData(carrier.model)
        last=source['samples'][-1]
        base_q=np.asarray(last['qpos_all'])
        scratch.qpos[:]=base_q
        scratch.qvel[:]=last['qvel_all']
        scratch.ctrl[carrier.actuators]=np.clip(np.asarray(last['command'])+carrier.bias,carrier.lower,carrier.upper)
        reference_geometry=hypothetical_geometry(carrier,scratch,last['position'],last['rotation'])
        reference=normal_measurement(carrier,scratch)
        report['static_reference']=dict(reference,geometry=reference_geometry)
        if reference['normal_dot'] is None or abs(reference['normal_dot']-source['end']['minimum_thumb_target_normal_dot'])>1e-8:
            raise RuntimeError('Isolated recorded-pose contact-normal regression differs; no static search')
        indices=np.array([JOINT_NAMES.index(name) for name in NAMES])
        original=np.asarray(source['command_knots'])
        start_command=np.asarray(source['start_command'])[:16]
        static=[]
        for number,values in enumerate(itertools.product((-.08,0.,.08),repeat=7)):
            delta=np.zeros(16)
            delta[indices]=values
            row=dict(index=number,delta_rad=delta.tolist(),admitted=False)
            try:
                terminal_knots(original,delta,start_command,carrier.lower[:16],carrier.upper[:16])
                changed=base_q[carrier.qadr[indices]]+delta[indices]
                if (np.any(changed<carrier.lower[indices]-1e-10)
                        or np.any(changed>carrier.upper[indices]+1e-10)):
                    raise ValueError('Modified actual joints exceed original ROM')
                scratch.qpos[:]=base_q
                scratch.qpos[carrier.qadr[indices]]=changed
                geometry=hypothetical_geometry(carrier,scratch,last['position'],last['rotation'])
                measurement=normal_measurement(carrier,scratch)
                score=static_grip_score(measurement['normal_dot'],geometry,delta)
                row.update(geometry=geometry,measurement=measurement,score=score,admitted=score is not None)
            except ValueError as error:
                row['reason']=str(error)
            static.append(row)
            if (number+1)%100==0:
                report['static_progress']=dict(tested=number+1,maximum=2187)
                save()
        assert len(static)==2187
        if paired_signature(carrier)!=signature:
            raise RuntimeError('Static screen mutated live physics/controller/camera/RNG/model state')
        grid_path=args.output/'static-grid.json'
        write(grid_path,static)
        ranked=sorted((s for s in static if s['admitted']),key=lambda s:(s['score'],s['index']))
        report.update(static_progress=dict(tested=2187,maximum=2187),
            static_grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
            static_admitted=len(ranked),static_live_state_unchanged=True,selected_static=ranked[:3],
            static_selection_frozen_at=datetime.now(timezone.utc).isoformat())
        save()
        baseline=execute(original,state,'frozen-partial-reference',np.zeros(16))
        if baseline['end']!=source['end']:
            raise RuntimeError('Full physical reference does not exactly reproduce parent case15')
        report['reference_exactly_reproduced']=True
        save()
        for candidate in report['selected_static']:
            delta=np.asarray(candidate['delta_rad'])
            knots=terminal_knots(original,delta,start_command,carrier.lower[:16],carrier.upper[:16])
            row=execute(knots,state,f'static-candidate-{candidate["index"]}',delta)
            report['candidate_results'].append(dict(static_index=candidate['index'],delta_rad=delta.tolist(),case_index=row['index'],
                score=row['selection_score'],native_task_success=row['native_task_success']))
            save()
        if report['candidate_results']:
            best=min(report['candidate_results'],key=lambda c:(c['score'],c['case_index']))
            report['selected_physical']=dict(best,frozen_before_transfers=True)
            save()
            delta=np.asarray(best['delta_rad'])
            for transfer in plan['transfers']:
                template=next(p for p in states[transfer['seed']]['proposals']
                              if p['condition']['name']=='points-normals-center-fixed')
                old_case=next(c for c in parent['cases'] if c['seed']==transfer['seed']
                              and c['role']=='points-normals-center-fixed')
                try:
                    knots=terminal_knots(template['command_knots'],delta,np.asarray(old_case['start_command'])[:16],
                                         carrier.lower[:16],carrier.upper[:16])
                except ValueError as error:
                    report['transfers'].append(dict(**transfer,skipped=True,reason=str(error),native_task_success=False))
                    save()
                    continue
                row=execute(knots,transfer,'selected-delta-development-transfer',delta)
                report['transfers'].append(dict(seed=transfer['seed'],case_index=row['index'],native_task_success=row['native_task_success']))
                save()
        assert len(report['cases'])<=7
        report.update(status='complete',active_case=None,finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(physical_cases=len(report['cases']),static_postures=2187,static_admitted=len(ranked),
                physically_tested_candidates=len(report['candidate_results']),
                candidate_successes=sum(c['native_task_success'] for c in report['candidate_results']),
                transfer_successes=sum(c['native_task_success'] for c in report['transfers']),
                transfer_skips=sum(c.get('skipped',False) for c in report['transfers']),
                no_static_candidate=not bool(ranked),formal_or_continuous_acceptance=False))
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
