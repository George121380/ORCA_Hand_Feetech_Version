"""Paired development-only grasp exploration/early-policy diagnostic.

Four conditions, or mean versus hold, on exactly14 development resets.
No training, source-policy selection or formal seeds; the active pilot is unchanged.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.pen_grasp import GraspConfig,PenGraspEnv,load_grasp_config,grasp_contract
from orca_feetech.training_supervision import checkpoint_steps
from manipulation import write


CONDITIONS=('zero','noise_log_std_minus4','noise_log_std_minus2','frozen_early_mean')
SEEDS=tuple(range(2111000,2111014))


def reset_signature(env):
    state=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    digest=hashlib.sha256(state.tobytes())
    for name in ('body_mass','body_inertia','geom_friction','actuator_gainprm','actuator_biasprm','dof_damping'):
        digest.update(getattr(env.model,name).tobytes())
    for value in (env.bias,env.encoder_bias,env.codec.q,env.codec.velocity,np.asarray(env.queue),
                  env.measured_pos,env.measured_rot,env.goal,np.asarray(env.observer.frames)):
        digest.update(np.asarray(value).tobytes())
    digest.update(json.dumps(dict(delay=env.delay,rng=env.np_random.bit_generator.state),sort_keys=True).encode())
    return digest.hexdigest()


def failure_flags(samples):
    end=samples[-1]
    if end['termination_reason']!='horizon':
        return dict(hard_failure=end['termination_reason']!='success')
    last=samples[-10:]
    return dict(hard_failure=False,loaded_opposition_failed=not all(s['loaded_opposition'] for s in last),
        rest_failed=not all(s['resting'] for s in last),contact_failed=not all(s['hand_contact'] for s in last),
        orientation_span_failed=end['final_500ms_orientation_span_deg'] is None or end['final_500ms_orientation_span_deg']>=2,
        position_span_failed=end['final_500ms_position_span_m'] is None or end['final_500ms_position_span_m']>=.002)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='Frozen checkpoint copy, never a live best_model path')
    parser.add_argument('--checkpoint-steps',type=int,choices=(40000,100000),default=40000,
                        help='Freeze by this declared step count, never evaluation score')
    parser.add_argument('--mean-vs-hold',action='store_true',help='Only zero action and frozen policy mean;28 episodes')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    config,meta=load_grasp_config(args.checkpoint)
    if config.fingerprint()!=GraspConfig(randomization=.3).fingerprint():
        raise ValueError('Expected the frozen0.3 randomized grasp contract')
    actual=checkpoint_steps(args.checkpoint,check_weights=True)
    if actual!=args.checkpoint_steps:
        raise ValueError(f'Expected the frozen{args.checkpoint_steps}-step checkpoint, got{actual}')
    conditions=('zero','frozen_early_mean') if args.mean_vs_hold else CONDITIONS
    raw=args.checkpoint.read_bytes()
    policy=PPO.load(args.checkpoint,device='cpu')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),acceptance_met=False,
        task_contract=grasp_contract(),config_fingerprint=config.fingerprint(),checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_steps=actual,checkpoint_metadata=meta,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
        protocol=dict(seeds=SEEDS,conditions=conditions,maximum_episodes=len(SEEDS)*len(conditions),
            placement='episode%7;two per bucket',declared_checkpoint_steps=args.checkpoint_steps,
            noise='none' if args.mean_vs_hold else 'same140x16 standard-normal array per seed; scale0/exp(-4)/exp(-2)',
            actor_noise_rng_seed=None if args.mean_vs_hold else 'environment_seed+1000000; independent from environment RNG',
            initial_state_model_control_rng_required_identical=True,active_training_changed=False,
            formal_seeds_used=False),cases=[])
    def save():
        write(args.output/'probe.json',report)
    save()
    env=PenGraspEnv(config)
    try:
        for episode,seed in enumerate(SEEDS):
            noise=np.random.default_rng(seed+1000000).standard_normal((140,16))
            expected=None
            for condition in conditions:
                obs,initial=env.reset(seed=seed,options={'placement_index':episode%7})
                signature=reset_signature(env)
                expected=signature if expected is None else expected
                if signature!=expected:
                    raise RuntimeError('Paired reset/model/controller/RNG state mismatch')
                command=env.codec.q.copy()
                row=dict(seed=seed,condition=condition,initial=initial,reset_signature=signature,samples=[])
                report['cases'].append(row)
                for tick in range(140):
                    action=(policy.predict(obs,deterministic=True)[0] if condition=='frozen_early_mean' else
                        noise[tick]*np.exp(-4) if condition=='noise_log_std_minus4' else
                        noise[tick]*np.exp(-2) if condition=='noise_log_std_minus2' else np.zeros(16))
                    obs,reward,done,truncated,info=env.step(action)
                    row['samples'].append(dict(tick=tick+1,reward=reward,action=np.asarray(action).tolist(),
                        command=env.codec.q.tolist(),hand_contact=env.contact(),**info))
                    if done or truncated:
                        break
                row.update(end=info,failed_requirements=failure_flags(row['samples']),
                    peak_joint_command_delta_rad=max(float(np.max(np.abs(np.asarray(s['command'])[:16]-command[:16])))
                                                     for s in row['samples']),
                    mean_absolute_action=float(np.mean(np.abs([s['action'] for s in row['samples']]))))
                save()
        report['summary']=[]
        for condition in conditions:
            rows=[r for r in report['cases'] if r['condition']==condition]
            reasons=sorted({r['end']['termination_reason'] for r in rows})
            report['summary'].append(dict(condition=condition,episodes=len(rows),
                successes=sum(r['end']['is_success'] for r in rows),drops=sum(r['end']['dropped'] for r in rows),
                mean_steps=float(np.mean([len(r['samples']) for r in rows])),
                mean_absolute_action=float(np.mean([r['mean_absolute_action'] for r in rows])),
                mean_peak_command_delta_rad=float(np.mean([r['peak_joint_command_delta_rad'] for r in rows])),
                reasons={reason:sum(r['end']['termination_reason']==reason for r in rows) for reason in reasons},
                failed_requirements={key:sum(r['failed_requirements'].get(key,False) for r in rows)
                    for key in sorted({k for r in rows for k in r['failed_requirements']})}))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
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
