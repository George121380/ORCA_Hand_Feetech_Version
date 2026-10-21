"""At most33 local motor pulses and one physically validated fitted pulse.

Every case executes its own real frozen grasp prefix. No state injection,
velocity clearing, training, hardware action or full-task acceptance credit.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from orca_feetech.contact_support import hand_object_contacts
from orca_feetech.control_authority import pose_delta,local_pose_metrics,fit_bounded_response
from orca_feetech.manipulation import yaw_rotation
from orca_feetech.model import JOINT_NAMES
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,load_rotation_config
from orca_feetech.training_supervision import checkpoint_steps
from manipulation import write
from probe_grasp_rotation_readiness import physical_control_signature
from probe_rotation_goal_response import counterfactual_goal,paired_signature


ROOT=Path(__file__).resolve().parents[1]


def contacts(env):
    rows,excluded=hand_object_contacts(env.carrier.model,env.carrier.data,env.carrier.geoms)
    return dict(excluded=excluded,hand_contacts=[dict(body=c['body'],position_m=c['position_m'].tolist(),
        normal_world=(c['sign']*c['frame'][0]).tolist(),force_world_n=c['solver_force_world_n'].tolist(),
        torque_world_nm=c['solver_torque_world_nm'].tolist(),dimension=c['dimension'],
        friction=c['friction'].tolist(),penetration_m=c['penetration_m']) for c in rows])


def complete_case(row,ticks):
    return bool(row['initial']['prefix_grasp_success'] and len(row['samples'])==ticks
                and row['end']['termination_reason'] is None and not row['end']['dropped'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    plan_path=ROOT/'configs/training/pen-rotate-authority-diagnostic.json'
    plan=json.loads(plan_path.read_text())
    if (plan['maximum_cases']!=34 or plan['pulse_ticks']!=10 or plan['pulse_action_amplitude']!=.04
            or plan['seed']!=2511000 or plan['placement_index']!=0 or plan['direction']!=1):
        raise ValueError('Local authority protocol changed; register a separate experiment')
    cfg,meta=load_rotation_config(args.checkpoint)
    if (cfg.fingerprint()!=RotationConfig().fingerprint() or checkpoint_steps(args.checkpoint,check_weights=True)!=100000
            or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=plan['source_checkpoint_sha256']):
        raise ValueError('Require the predeclared frozen100k checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),acceptance_met=False,
        continuous_rotation_accepted=False,plan=plan,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_metadata=meta,
        checkpoint_sha256=plan['source_checkpoint_sha256'],cases=[],fit=None,validation=None,
        limits='One development state,0.5s response only; no7s rotation or hardware success claim')
    policy=PPO.load(args.checkpoint,device='cpu')
    env=PenRotateEnv(cfg)
    expected=None
    amplitude=plan['pulse_action_amplitude']
    ticks=plan['pulse_ticks']
    def save():
        write(args.output/'probe.json',report)
    def run_case(name,pulse):
        nonlocal expected
        obs,initial=env.reset(seed=plan['seed'],options=dict(placement_index=0,direction=1))
        signature=paired_signature(env.carrier)
        expected=signature if expected is None else expected
        if signature!=expected:
            raise RuntimeError('Pulse changed the independently executed physical prefix distribution')
        start_position,start_rotation=env.pose()
        row=dict(name=name,residual_action=np.asarray(pulse).tolist(),initial=initial,
            prefix_signature=signature,start_position=start_position.tolist(),start_rotation=start_rotation.tolist(),
            start_time_s=float(env.carrier.data.time),initial_contacts=contacts(env),samples=[])
        report['cases'].append(row)
        for tick in range(ticks):
            before=physical_control_signature(env.carrier)
            neutral_obs,_=counterfactual_goal(obs,env.goal,env.goal_capture_rotation)
            base=policy.predict(neutral_obs,deterministic=True)[0]
            if before!=physical_control_signature(env.carrier):
                raise RuntimeError('Actor prediction mutated the physical/control/camera/RNG state')
            action=base+np.asarray(pulse)
            obs,reward,done,truncated,info=env.step(action)
            position,rotation=env.pose()
            row['samples'].append(dict(tick=tick+1,action=action.tolist(),neutral_action=base.tolist(),
                position=position.tolist(),rotation=rotation.tolist(),command=env.carrier.codec.q.tolist(),
                hand_qpos=env.carrier.data.qpos[env.carrier.qadr].tolist(),
                hand_qvel=env.carrier.data.qvel[env.carrier.hand_dofs].tolist(),contacts=contacts(env),**info))
            if done or truncated:
                break
        row.update(end=info,end_position=position.tolist(),end_rotation=rotation.tolist(),
                   end_time_s=float(env.carrier.data.time))
        row['full_uncensored_duration']=complete_case(row,ticks)
        if row['full_uncensored_duration'] and not np.isclose(row['end_time_s']-row['start_time_s'],.5,atol=1e-10,rtol=0):
            raise RuntimeError('Uncensored pulse duration differs from the registered0.5s')
        save()
        return row
    try:
        reference=run_case('reference',np.zeros(16))
        pairs=[]
        for motor in range(16):
            pulse=np.zeros(16)
            pulse[motor]=amplitude
            positive=run_case(f'{JOINT_NAMES[motor]}-positive',pulse)
            negative=run_case(f'{JOINT_NAMES[motor]}-negative',-pulse)
            pairs.append((positive,negative))
        matrix=np.full((6,16),np.nan)
        linearity=[]
        if reference['full_uncensored_duration']:
            ref_p=np.array(reference['end_position'])
            ref_r=np.array(reference['end_rotation'])
            for motor,(positive,negative) in enumerate(pairs):
                valid=positive['full_uncensored_duration'] and negative['full_uncensored_duration']
                detail=dict(motor=JOINT_NAMES[motor],uncensored_pair=bool(valid))
                if valid:
                    plus=pose_delta(positive['end_position'],positive['end_rotation'],ref_p,ref_r)
                    minus=pose_delta(negative['end_position'],negative['end_rotation'],ref_p,ref_r)
                    matrix[:,motor]=(plus-minus)/(2*amplitude)
                    detail.update(positive_response=plus.tolist(),negative_response=minus.tolist(),
                                  even_nonlinear_response=((plus+minus)/2).tolist())
                linearity.append(detail)
            start_p=np.array(reference['start_position'])
            start_r=np.array(reference['start_rotation'])
            target_r=yaw_rotation(np.deg2rad(2.))@start_r
            desired=pose_delta(start_p,target_r,ref_p,ref_r)
            fit=fit_bounded_response(matrix,desired)
            predicted=np.array(fit['predicted_delta'])
            pred_p=ref_p+predicted[:3]
            pred_r=Rotation.from_rotvec(predicted[3:]).as_matrix()@ref_r
            prediction=local_pose_metrics(pred_p,pred_r,start_p,start_r)
            fit.update(jacobian=[[None if not np.isfinite(v) else float(v) for v in row] for row in matrix],
                       columns=linearity,desired_delta=desired.tolist(),predicted_pose=prediction)
            report['fit']=fit
            save()
            if fit['solver_success'] and prediction['local_pose_tracking_met']:
                validation=run_case('one-fitted-validation',fit['residual_action'])
                metrics=local_pose_metrics(validation['end_position'],validation['end_rotation'],start_p,start_r)
                report['validation']=dict(full_uncensored_duration=validation['full_uncensored_duration'],
                    pose=metrics,local_pose_tracking_verified=bool(validation['full_uncensored_duration']
                        and metrics['local_pose_tracking_met']),end=validation['end'])
            else:
                report['validation']=dict(skipped=True,reason='Bounded fitted prediction did not meet predeclared local target')
        else:
            report['validation']=dict(skipped=True,reason='Reference did not complete the full valid0.5s; no fit or retry')
        assert len(report['cases'])<=34
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(cases=len(report['cases']),prefix_successes=sum(r['initial']['prefix_grasp_success'] for r in report['cases']),
                full_uncensored_cases=sum(r['full_uncensored_duration'] for r in report['cases']),
                censored_case_names=[r['name'] for r in report['cases'] if not r['full_uncensored_duration']],
                all_prefixes_exactly_matched=True,rank=None if report['fit'] is None else report['fit']['rank'],
                validation=report['validation']))
        save()
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
