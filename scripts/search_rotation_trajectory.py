"""One-state nonlinear trajectory search using fresh physical grasp prefixes.

Fixed161 search cases and3 frozen-plan replays. No state injection, training,
formal test, hardware action, retries or continuous-rotation acceptance credit.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from orca_feetech.manipulation import DT
from orca_feetech.model import ModelSpec
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,PREFIX_SHA256
from orca_feetech.rotation_trajectory import trajectory_target,trajectory_cost
from manipulation import write
from probe_rotation_authority import contacts
from probe_rotation_goal_response import paired_signature


ROOT=Path(__file__).resolve().parents[1]


def load_plan():
    path=ROOT/'configs/training/pen-rotate-trajectory-diagnostic-v1.json'
    plan=json.loads(path.read_text())
    expected=dict(search_seed=2511000,placement_index=0,direction=1,optimizer_seed=3161000,
        population=32,generations=5,elites=8,offset_limit_rad=.18,initial_sigma_rad=.07,
        sigma_floor_rad=.012,knot_seconds=[2.5,5.],horizon_seconds=7.,search_cases=161,
        maximum_cases=164,extensions=0)
    if (any(plan[k]!=v for k,v in expected.items()) or plan['prefix_checkpoint_sha256']!=PREFIX_SHA256
            or plan['rotation_fingerprint']!=RotationConfig().fingerprint()
            or [(r['seed'],r['placement_index'],r['direction']) for r in plan['replays']]
            !=[(2511000,0,1),(2511001,1,1),(2511002,2,1)]):
        raise ValueError('Changed search budget, physics or evaluation; register a new experiment')
    return plan,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan,plan_sha=load_plan()
    raw=args.parent_report.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=plan['parent_report_sha256']:
        raise ValueError('Require immutable33-case authority report for the physical prefix signature')
    parent=json.loads(raw)
    if parent['checkpoint_metadata']['fingerprint']!=plan['rotation_fingerprint']:
        raise ValueError('Parent physical/control/observation contract differs')
    expected_signature=parent['cases'][0]['prefix_signature']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cases').mkdir()
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),plan=plan,
        plan_sha256=plan_sha,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_source_sha256=ModelSpec.load().source_sha256,
        rotation_fingerprint=RotationConfig().fingerprint(),prefix_checkpoint_sha256=PREFIX_SHA256,
        acceptance_met=False,continuous_rotation_accepted=False,cases=[],generations=[],
        selected=None,replays=[],active_case=None,progress=None)
    def save():
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'search.json',report)
    save()
    env=PenRotateEnv(RotationConfig())
    def run_case(role,offsets,seed=2511000,bucket=0,direction=1):
        number=len(report['cases'])
        if number>=164:
            raise RuntimeError('Predeclared total case budget exhausted')
        path=args.output/'cases'/f'case-{number:04d}.json'
        report['active_case']=dict(index=number,role=role,seed=seed,path=str(path))
        save()
        obs,initial=env.reset(seed=seed,options=dict(placement_index=bucket,direction=direction))
        signature=paired_signature(env.carrier)
        start_p,start_r=env.pose()
        initial_command=env.carrier.codec.q.copy()
        previous_command=initial_command.copy()
        previous_velocity=env.carrier.codec.velocity.copy()
        release_origin=env.carrier.origin.copy()
        start_time=float(env.carrier.data.time)
        row=dict(index=number,role=role,seed=seed,placement_index=bucket,direction=direction,
            offsets_rad=np.asarray(offsets).tolist(),initial=initial,prefix_signature=signature,
            start_position=start_p.tolist(),start_rotation=start_r.tolist(),goal_rotation=env.goal.tolist(),
            start_command=initial_command.tolist(),inherited_command_velocity=previous_velocity.tolist(),
            release_origin=release_origin.tolist(),start_time_s=start_time,initial_contacts=contacts(env),
            samples=[],peak_command_velocity_rad_s=0.,peak_command_acceleration_rad_s2=0.)
        write(path,row)
        try:
            if seed==2511000 and signature!=expected_signature:
                raise RuntimeError('Independent prefix differs from immutable parent; stop without retry')
            if not initial['prefix_grasp_success']:
                # No action, retry or successful-prefix filtering after failure.
                info=initial
            else:
                for tick in range(140):
                    target=trajectory_target(initial_command,offsets,(tick+1)*DT,
                        env.carrier.codec.lower,env.carrier.codec.upper)
                    action=np.clip((target-env.carrier.codec.q[:16])/(env.config.max_velocity*DT),-1,1)
                    input_observation=obs.copy()
                    obs,reward,done,truncated,info=env.step(action)
                    position,rotation=env.pose()
                    command=env.carrier.codec.q.copy()
                    velocity=(command[:16]-previous_command[:16])/DT
                    row['peak_command_velocity_rad_s']=max(row['peak_command_velocity_rad_s'],float(max(abs(velocity))))
                    row['peak_command_acceleration_rad_s2']=max(row['peak_command_acceleration_rad_s2'],
                        float(max(abs(velocity-previous_velocity)))/DT)
                    limits_met=bool(np.all(command>=env.carrier.codec.lower-1e-10)
                        and np.all(command<=env.carrier.codec.upper+1e-10)
                        and abs(command[-1]-env.carrier.hand_spec.neutral[-1])<1e-10
                        and row['peak_command_velocity_rad_s']<=.8+1e-9
                        and row['peak_command_acceleration_rad_s2']<=2.+1e-9)
                    row['samples'].append(dict(tick=tick+1,observation=input_observation.tolist(),
                        action=action.tolist(),position_target=target.tolist(),command=command.tolist(),
                        command_limits_met=limits_met,position=position.tolist(),rotation=rotation.tolist(),
                        handoff_displacement_m=float(np.linalg.norm(position-start_p)),
                        hand_qpos=env.carrier.data.qpos[env.carrier.qadr].tolist(),
                        hand_qvel=env.carrier.data.qvel[env.carrier.hand_dofs].tolist(),contacts=contacts(env),**info))
                    previous_command,previous_velocity=command,velocity
                    if (tick+1)%20==0:
                        write(path,row)
                    if done or truncated or not limits_met:
                        break
            np.testing.assert_array_equal(env.carrier.origin,release_origin)
            complete=bool(initial['prefix_grasp_success'] and len(row['samples'])==140
                and info['termination_reason'] in ('horizon','success') and not info['dropped'])
            command_ok=all(s['command_limits_met'] for s in row['samples'])
            elapsed=float(env.carrier.data.time-start_time)
            if complete and not np.isclose(elapsed,7.,atol=1e-9,rtol=0):
                raise RuntimeError('Complete trajectory changed the original7s physical duration')
            row.update(end=info,final_observation=obs.tolist(),simulated_seconds=elapsed,
                full_uncensored_duration=complete,command_limits_met=command_ok,
                score=trajectory_cost(row['samples'],offsets,complete=complete,command_limits_met=command_ok),
                native_task_success=bool(complete and command_ok and info['is_success']))
            write(path,row)
        except BaseException as error:
            row['error']=repr(error)
            write(path,row)
            raise
        brief={k:v for k,v in row.items() if k not in ('samples','final_observation','initial_contacts')}
        brief.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['cases'].append(brief)
        report['active_case']=None
        report['progress']=dict(completed_cases=len(report['cases']),maximum_cases=164,
            native_task_successes=sum(c['native_task_success'] for c in report['cases']),
            full_uncensored_cases=sum(c['full_uncensored_duration'] for c in report['cases']))
        save()
        print(json.dumps(dict(case=number,role=role,score=row['score'],ticks=len(row['samples']),
            end_error_deg=info['angle_error_deg'],reason=info['termination_reason'],
            native_task_success=row['native_task_success'])),flush=True)
        return brief
    try:
        zero=np.zeros((2,16))
        baseline=run_case('zero-offset-reference',zero)
        best=zero.copy()
        best_row=baseline
        mean=np.zeros(32)
        sigma=np.full(32,.07)
        rng=np.random.default_rng(3161000)
        for generation in range(5):
            candidates=np.clip(rng.normal(mean,sigma,(32,32)),-.18,.18)
            candidates[0]=best.ravel()
            rows=[]
            for index,candidate in enumerate(candidates):
                row=run_case(f'generation-{generation}-candidate-{index}',candidate.reshape(2,16))
                rows.append(row)
                if row['score']<best_row['score']:
                    best,best_row=candidate.reshape(2,16).copy(),row
            scores=[r['score'] for r in rows]
            elite=candidates[np.argsort(scores,kind='stable')[:8]]
            mean=elite.mean(axis=0)
            sigma=np.maximum(elite.std(axis=0),.012)
            report['generations'].append(dict(generation=generation,best_case=best_row['index'],
                best_score=best_row['score'],best_error_deg=best_row['end']['angle_error_deg'],
                native_task_successes=sum(r['native_task_success'] for r in rows),
                full_uncensored_cases=sum(r['full_uncensored_duration'] for r in rows),
                next_mean=mean.tolist(),next_sigma=sigma.tolist()))
            save()
        assert len(report['cases'])==161
        report['selected']=dict(offsets_rad=best.tolist(),search_case=best_row['index'],
            score=best_row['score'],selected_at=datetime.now(timezone.utc).isoformat(),
            frozen_before_replays=True)
        save()
        for replay in plan['replays']:
            row=run_case(replay['role'],best,replay['seed'],replay['placement_index'],replay['direction'])
            if replay['seed']==2511000:
                if row['end']!=best_row['end'] or row['score']!=best_row['score']:
                    raise RuntimeError('Identical physical prefix and selected trajectory failed exact replay')
            report['replays'].append(row)
            save()
        assert len(report['cases'])==164
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(search_cases=161,replay_cases=3,
                selected_search_success=best_row['native_task_success'],
                identical_replay_success=report['replays'][0]['native_task_success'],
                non_search_development_transfer_successes=sum(r['native_task_success'] for r in report['replays'][1:]),
                formal_or_continuous_acceptance=False))
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
