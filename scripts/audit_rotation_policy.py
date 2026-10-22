"""One fixed100k checkpoint on the28 existing BC development states; no training.

Records every original observation/action and native gate. The BC endpoint
comparison is paired; these reused development seeds never grant acceptance.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from manipulation import write
from orca_feetech.manipulation import rotation_error
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,load_rotation_config
from orca_feetech.training_supervision import checkpoint_steps


ROOT=Path(__file__).resolve().parents[1]


def failure_summary(samples,end,*,horizon=140):
    """Overlapping explanations of the ORIGINAL gate, never alternative success."""
    if not end['prefix_grasp_success']:
        return dict(reasons=['prefix_failed'],tail_ticks=0)
    tail=samples[-20:]
    reasons=[]
    if len(samples)!=horizon or end['termination_reason'] not in ('horizon','success'):
        reasons.append('native_constraint_or_early_end')
    counts=dict(pose=sum(x['info']['angle_error_deg']<8 for x in tail),
        contact=sum(x['contact'] and x['info']['loaded_opposition'] for x in tail),
        rest=sum(x['info']['resting'] for x in tail),
        native_stable=sum(x['info']['stable'] for x in tail))
    for key in ('pose','contact','rest'):
        if len(tail)!=20 or counts[key]!=20:
            reasons.append('incomplete_final_1s_'+key)
    span=distance=None
    if tail:
        last=tail[-1]
        span=max(float(np.rad2deg(rotation_error(np.asarray(x['rotation']),
            np.asarray(last['rotation'])))) for x in tail)
        distance=max(float(np.linalg.norm(np.asarray(x['position'])-last['position'])) for x in tail)
        if span>=2:
            reasons.append('final_1s_orientation_span')
        if distance>=.002:
            reasons.append('final_1s_position_span')
    return dict(reasons=reasons,tail_ticks=len(tail),tail_counts=counts,
        final_1s_orientation_span_deg=span,final_1s_position_span_m=distance,
        native_success=bool(end['is_success']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--bc-metrics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan_path=ROOT/'configs/training/pen-rotate-bc-ppo-midpoint-audit-v1.json'
    plan=json.loads(plan_path.read_text())
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest!=plan['checkpoint_sha256'] or checkpoint_steps(args.checkpoint,check_weights=True)!=100000:
        raise ValueError('Require the fixed step-selected100k checkpoint')
    config,meta=load_rotation_config(args.checkpoint)
    if config.fingerprint()!=RotationConfig().fingerprint():
        raise ValueError('Original physical/reward/native task contract changed')
    reference=json.loads(args.bc_metrics.read_text())
    if hashlib.sha256(args.bc_metrics.read_bytes()).hexdigest()!=plan['bc_metrics_sha256']:
        raise ValueError('BC reference changed')
    if (plan['seed']!=3211000 or plan['episodes']!=28 or plan['extensions']!=0
            or len(reference['trials'])!=28):
        raise ValueError('Changed fixed development allocation')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cases').mkdir()
    report=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),plan=plan,
        checkpoint_sha256=digest,checkpoint_metadata=meta,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        is_formal_evaluation=False,acceptance_met=False,continuous_rotation_accepted=False,cases=[])
    def save():
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'audit.json',report)
    save()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    model=PPO.load(args.checkpoint,device='cpu')
    env=PenRotateEnv(config)
    try:
        for i,ref in enumerate(reference['trials']):
            state=dict(seed=3211000+i,placement_index=i%7,direction=-1 if (i//7)%2==0 else 1)
            if any(ref[k]!=v for k,v in state.items()):
                raise ValueError('BC seed order changed')
            observation,initial=env.reset(seed=state['seed'],options=dict(
                placement_index=state['placement_index'],direction=state['direction']))
            paired=dict(initial_position=initial['object_position'],initial_rotation=env.pose()[1].tolist(),
                goal_rotation=env.goal.tolist(),prefix_grasp_success=initial['prefix_grasp_success'],
                prefix_steps=initial['prefix_steps'],prefix_end_reason=initial['prefix_end_reason'])
            if any(paired[k]!=ref[k] for k in paired):
                raise RuntimeError('PPO and BC prefixes/goals are not exactly matched')
            row=dict(**state,paired=paired,initial=initial,samples=[])
            total=0.
            for tick in range(config.horizon):
                action=model.predict(observation,deterministic=True)[0]
                previous=observation.copy()
                observation,reward,done,truncated,info=env.step(action)
                position,rotation=env.pose()
                row['samples'].append(dict(tick=tick+1,observation=previous.tolist(),action=action.tolist(),
                    position=position.tolist(),rotation=rotation.tolist(),contact=bool(env.carrier.contact()),
                    command=env.carrier.codec.q[:16].tolist(),reward=reward,info=info))
                total+=reward
                if done or truncated:
                    break
            row.update(end=info,reward=total,diagnosis=failure_summary(row['samples'],info),
                bc_success=ref['is_success'],bc_final_error_deg=ref['angle_error_deg'])
            path=args.output/'cases'/f'case-{i:04d}.json'
            write(path,row)
            report['cases'].append(dict(**state,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                end=info,diagnosis=row['diagnosis'],bc_success=ref['is_success'],
                bc_final_error_deg=ref['angle_error_deg']))
            save()
        rows=report['cases']
        failures=[r for r in rows if not r['end']['is_success']]
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(bc_successes=sum(r['bc_success'] for r in rows),
                ppo_successes=sum(r['end']['is_success'] for r in rows),episodes=28,
                gained=sum(not r['bc_success'] and r['end']['is_success'] for r in rows),
                lost=sum(r['bc_success'] and not r['end']['is_success'] for r in rows),
                drops=sum(r['end']['dropped'] for r in rows),
                successes_per_direction={str(d):sum(r['end']['is_success'] and r['direction']==d for r in rows)
                    for d in (-1,1)},
                mean_final_error_deg=float(np.mean([r['end']['angle_error_deg'] for r in rows])),
                overlapping_failure_counts=dict(Counter(x for r in failures for x in r['diagnosis']['reasons'])),
                exact_prefix_goal_pairing=True))
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
