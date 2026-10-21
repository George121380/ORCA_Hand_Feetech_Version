"""One new physical validation from a fixed33-case local response report.

This is a separate1deg experiment, not the skipped2deg validation. No retraining,
state injection, retries, hardware actions or full-task acceptance credit.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from orca_feetech.control_authority import pose_delta,local_pose_metrics,fit_bounded_response
from orca_feetech.manipulation import yaw_rotation
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,load_rotation_config
from orca_feetech.training_supervision import checkpoint_steps
from manipulation import write
from probe_grasp_rotation_readiness import physical_control_signature
from probe_rotation_authority import contacts,complete_case
from probe_rotation_goal_response import counterfactual_goal,paired_signature


ROOT=Path(__file__).resolve().parents[1]
PARENT_SHA='97c13437b0a5755dc46c2af6c204f5396020a27597a0093ff3690e68c54dc842'
CHECKPOINT_SHA='a64f95f2b23029efcc02ad77237d019c3d72d0e228bdd913b858bd502bf090fb'


def load_parent(path,plan):
    if (plan['parent_report_sha256']!=PARENT_SHA or plan['source_checkpoint_sha256']!=CHECKPOINT_SHA
            or plan['local_target_deg']!=1. or plan['maximum_cases']!=1 or plan['extensions']!=0
            or plan['pulse_ticks']!=10 or plan['pulse_seconds']!=.5
            or plan['residual_action_bound']!=.04 or plan['fit_regularization']!=.1
            or plan['seed']!=2511000 or plan['placement_index']!=0 or plan['direction']!=1):
        raise ValueError('Changed local protocol; register a separate experiment')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:
        raise ValueError('Require the exact immutable33-case parent report')
    parent=json.loads(raw)
    if (parent['status']!='complete' or len(parent['cases'])!=33
            or not parent['validation'].get('skipped') or parent['checkpoint_sha256']!=CHECKPOINT_SHA
            or not parent['summary']['all_prefixes_exactly_matched']
            or not all(case['full_uncensored_duration'] for case in parent['cases'])):
        raise ValueError('Parent is not the preserved complete33-case experiment with skipped2deg validation')
    return parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--parent-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan_path=ROOT/'configs/training/pen-rotate-local-validation-v2.json'
    plan=json.loads(plan_path.read_text())
    parent=load_parent(args.parent_report,plan)
    cfg,metadata=load_rotation_config(args.checkpoint)
    if (cfg.fingerprint()!=RotationConfig().fingerprint()
            or checkpoint_steps(args.checkpoint,check_weights=True)!=100000
            or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA):
        raise ValueError('Require the unchanged default rotation config and fixed100k checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    reference=parent['cases'][0]
    start_p=np.asarray(reference['start_position'])
    start_r=np.asarray(reference['start_rotation'])
    ref_p=np.asarray(reference['end_position'])
    ref_r=np.asarray(reference['end_rotation'])
    desired=pose_delta(start_p,yaw_rotation(np.deg2rad(1.))@start_r,ref_p,ref_r)
    fit=fit_bounded_response(np.asarray(parent['fit']['jacobian'],dtype=float),desired)
    predicted=np.asarray(fit['predicted_delta'])
    predicted_p=ref_p+predicted[:3]
    predicted_r=Rotation.from_rotvec(predicted[3:]).as_matrix()@ref_r
    prediction=local_pose_metrics(predicted_p,predicted_r,start_p,start_r,target_deg=1.)
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),plan=plan,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_metadata=metadata,checkpoint_sha256=CHECKPOINT_SHA,
        parent_report_sha256=PARENT_SHA,parent_two_degree_validation=parent['validation'],
        fit=dict(fit,desired_delta=desired.tolist(),predicted_pose=prediction),cases=[],
        acceptance_met=False,continuous_rotation_accepted=False)
    def save():
        write(args.output/'validation.json',report)
    save()
    if not fit['solver_success'] or not prediction['local_pose_tracking_met']:
        report.update(status='complete',skipped=True,reason='Predicted1deg pose failed predeclared admission')
        save()
        print(json.dumps(report['fit'],indent=2))
        return
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    policy=PPO.load(args.checkpoint,device='cpu')
    env=PenRotateEnv(cfg)
    try:
        obs,initial=env.reset(seed=2511000,options=dict(placement_index=0,direction=1))
        position,rotation=env.pose()
        signature=paired_signature(env.carrier)
        row=dict(initial=initial,prefix_signature=signature,start_position=position.tolist(),
            start_rotation=rotation.tolist(),start_time_s=float(env.carrier.data.time),
            residual_action=fit['residual_action'],initial_contacts=contacts(env),samples=[])
        report['cases'].append(row)
        save()
        if signature!=reference['prefix_signature']:
            raise RuntimeError('Independently executed prefix did not exactly match parent; no retry')
        np.testing.assert_array_equal(position,start_p)
        np.testing.assert_array_equal(rotation,start_r)
        pulse=np.asarray(fit['residual_action'])
        for tick in range(10):
            before=physical_control_signature(env.carrier)
            neutral_obs,_=counterfactual_goal(obs,env.goal,env.goal_capture_rotation)
            base=policy.predict(neutral_obs,deterministic=True)[0]
            if physical_control_signature(env.carrier)!=before:
                raise RuntimeError('Prediction mutated physical/control/camera/RNG state')
            action=base+pulse
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
        row['full_uncensored_duration']=complete_case(row,10)
        if row['full_uncensored_duration'] and not np.isclose(row['end_time_s']-row['start_time_s'],.5,atol=1e-10,rtol=0):
            raise RuntimeError('Duration changed from the registered0.5s')
        metrics=local_pose_metrics(position,rotation,start_p,start_r,target_deg=1.)
        discrepancy=pose_delta(position,rotation,predicted_p,predicted_r)
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(cases=1,all_prefixes_exactly_matched=True,
                full_uncensored_duration=row['full_uncensored_duration'],actual_pose=metrics,
                local_pose_tracking_verified=bool(row['full_uncensored_duration'] and metrics['local_pose_tracking_met']),
                model_position_error_m=float(np.linalg.norm(discrepancy[:3])),
                model_rotation_error_deg=float(np.rad2deg(np.linalg.norm(discrepancy[3:]))),
                end=info))
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
