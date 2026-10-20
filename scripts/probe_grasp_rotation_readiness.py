"""Development-only goal response after a real learned seven-second grasp.

No learning, formal rotation credit, pose injection, velocity clearing, retry,
or successful-prefix selection. Every failed prefix remains a failed case.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from orca_feetech.manipulation import yaw_rotation
from orca_feetech.pen_grasp import PenGraspEnv,load_grasp_config
from orca_feetech.training_supervision import checkpoint_steps
from manipulation import write
from probe_grasp_exploration import reset_signature


CONDITIONS=('release_goal','new_zero_goal','yaw_minus15','yaw_plus15')
SEEDS=tuple(range(2111000,2111014))


def physical_control_signature(env):
    state=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    h=hashlib.sha256(state.tobytes())
    for value in (env.codec.q,env.codec.velocity,np.asarray(env.queue),env.bias,env.encoder_bias,
                  env.measured_pos,env.measured_rot):
        h.update(np.asarray(value).tobytes())
    h.update(json.dumps(dict(rng=env.np_random.bit_generator.state,steps=env.steps,
        age=env.age,valid=env.valid,dropout_left=env.dropout_left,delay=env.delay),sort_keys=True).encode())
    return h.hexdigest()


def handoff_goal(env,condition):
    before=physical_control_signature(env)
    rotation=env.pose()[1]
    if condition!='release_goal':
        angle={'new_zero_goal':0.,'yaw_minus15':-15.,'yaw_plus15':15.}[condition]
        env.goal=yaw_rotation(np.deg2rad(angle))@rotation
    obs=env._observation()  # Keep the preceding three real history frames.
    after=physical_control_signature(env)
    if before!=after:
        raise RuntimeError('Goal handoff changed physical/control/sensor/RNG state')
    return obs,before


def rotation_gate(info,hand_contact,reason=None):
    return bool(not reason and not info['dropped'] and info['vision_age']<=.25
        and info['angle_error_deg']<8 and info['displacement_m']<.04 and hand_contact
        and info['linear_speed_mps']<.08 and info['angular_speed_rps']<1.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    cfg,meta=load_grasp_config(args.checkpoint)
    job=json.loads((args.source_job/'job.json').read_text())
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if (job['status']!='complete' or not job['acceptance_met'] or job['selected_sha256']!=digest
            or cfg.randomization!=.3):
        raise ValueError('Require the frozen accepted0.3 grasp checkpoint and its completed source job')
    assert checkpoint_steps(args.checkpoint,check_weights=True)==200704
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',acceptance_met=False,source_revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),checkpoint_sha256=digest,checkpoint_metadata=meta,
        source_job=str(args.source_job),source_grasp_test=job['test'],cases=[],
        protocol=dict(seeds=SEEDS,conditions=CONDITIONS,maximum_cases=56,formal_seeds_used=False,
            prefix='One physical reach/manual release and140 deterministic source-grasp actions; failures count, no retries',
            continuation='Up to140 additional physical actions, no physical/control/RNG reset; retain cumulative geometry/force/displacement audit',
            goals='Release reference, current rotation, or world-Z +/-15deg from actual prefix-end rotation',
            observation='Same296 deployable inputs; retain history across goal change',
            rotation_gate='FullSO3<8deg, contact,v<.08m/s,w<1rad/s,displacement<4cm,10 consecutive ticks; prefix success never counts as rotation',
            credit='Record first reached and final retained goal separately; no formal rotation or sim2real acceptance'))
    def save():
        write(args.output/'probe.json',report)
    policy=PPO.load(args.checkpoint,device='cpu')
    env=PenGraspEnv(cfg)
    try:
        for episode,seed in enumerate(SEEDS):
            expected_initial=expected_prefix=None
            for condition in CONDITIONS:
                obs,initial=env.reset(seed=seed,options={'placement_index':episode%7})
                signature=reset_signature(env)
                expected_initial=signature if expected_initial is None else expected_initial
                if signature!=expected_initial:
                    raise RuntimeError('Paired initial state mismatch')
                row=dict(seed=seed,condition=condition,initial=initial,initial_signature=signature,
                         prefix_samples=[],continuation_samples=[])
                report['cases'].append(row)
                for _ in range(140):
                    action=policy.predict(obs,deterministic=True)[0]
                    obs,reward,done,truncated,info=env.step(action)
                    row['prefix_samples'].append(dict(action=action.tolist(),**info))
                    if done or truncated:
                        break
                row['prefix_success']=bool(info['is_success'])
                row.update(first_rotation_hold_tick=None,final_rotation_hold=False,
                           last_grasp_stable=False,end=info)
                if not row['prefix_success']:
                    row['end_reason']='prefix_failed'
                    save()
                    continue
                signature=reset_signature(env)
                expected_prefix=signature if expected_prefix is None else expected_prefix
                if signature!=expected_prefix:
                    raise RuntimeError('Paired learned prefix state mismatch')
                row['prefix_end_signature']=signature
                row['prefix_end_time_s']=float(env.data.time)
                row['prefix_end_rotation']=env.pose()[1].tolist()
                obs,handoff=handoff_goal(env,condition)
                row.update(handoff_physical_control_signature=handoff,goal=env.goal.tolist())
                held=0
                for tick in range(140):
                    action=policy.predict(obs,deterministic=True)[0]
                    info=env._advance(action)  # Physical continuation, no grasp done/success shortcut.
                    reason=env._constraint_reason() or ('dropped' if info['dropped'] else
                        'vision_timeout' if env.age>.25 else None)
                    valid=rotation_gate(info,env.contact(),reason)
                    held=held+1 if valid else 0
                    if held>=10 and row['first_rotation_hold_tick'] is None:
                        row['first_rotation_hold_tick']=tick+1
                    info['source_grasp_completed']=info.pop('is_success')
                    info['source_grasp_stable']=info.pop('stable')
                    info['rotation_gate_now']=valid
                    row['continuation_samples'].append(dict(tick=tick+1,action=action.tolist(),
                        command=env.codec.q.tolist(),rotation=env.pose()[1].tolist(),**info))
                    obs=env._observation()
                    row.update(end=info,end_reason=reason or 'continuation_horizon',
                        final_rotation_hold=bool(held>=10 and not reason),
                        last_grasp_stable=bool(info['source_grasp_stable'] and not reason),
                        end_rotation=env.pose()[1].tolist(),
                        final_time_s=float(env.data.time))
                    if reason:
                        break
                save()
        report['summary']=[]
        for condition in CONDITIONS:
            rows=[r for r in report['cases'] if r['condition']==condition]
            report['summary'].append(dict(condition=condition,cases=len(rows),
                prefix_successes=sum(r['prefix_success'] for r in rows),
                ever_rotation_holds=sum(r['first_rotation_hold_tick'] is not None for r in rows),
                final_rotation_holds=sum(r['final_rotation_hold'] for r in rows),
                last_grasp_stable=sum(r['last_grasp_stable'] for r in rows),
                reasons={k:sum(r['end_reason']==k for r in rows) for k in sorted({r['end_reason'] for r in rows})}))
        report['status']='complete'
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
