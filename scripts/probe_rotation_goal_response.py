"""Fourteen development rollouts: paired +/- goals after the real frozen grasp.

Also audit deterministic actor responses to changing only active goal entries
in each actual observation. Counterfactual actions never enter the simulator.
No training, formal seeds, pose injection, retries or acceptance credit.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from orca_feetech.manipulation import FRAME_SIZE,HISTORY,yaw_rotation
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,load_rotation_config
from orca_feetech.training_supervision import checkpoint_steps
from manipulation import write
from probe_grasp_rotation_readiness import physical_control_signature


SEEDS=tuple(range(2511000,2511007))
GOAL=slice(47,56)


def counterfactual_goal(observation,current,alternative):
    """Keep preparation history; replace only frames carrying the active goal."""
    frames=np.asarray(observation).reshape(HISTORY,FRAME_SIZE).copy()
    matched=np.all(np.isclose(frames[:,GOAL],np.asarray(current).reshape(9),rtol=0,atol=1e-6),axis=1)
    if not matched[-1]:
        raise ValueError('Latest observation does not carry the declared current goal')
    frames[matched,GOAL]=np.asarray(alternative).reshape(9)
    return frames.ravel(),matched


def rotation_components(start,end,direction):
    # R maps object into world; end @ start.T is the WORLD rotation increment.
    vector=Rotation.from_matrix(np.asarray(end)@np.asarray(start).T).as_rotvec(degrees=True)
    return dict(world_rotvec_deg=vector.tolist(),signed_goal_yaw_progress_deg=float(direction*vector[2]),
                off_axis_rotation_deg=float(np.linalg.norm(vector[:2])),total_rotation_deg=float(np.linalg.norm(vector)))


def paired_signature(carrier):
    h=hashlib.sha256(physical_control_signature(carrier).encode())
    for name in ('body_mass','body_inertia','geom_friction','actuator_gainprm','actuator_biasprm','dof_damping','eq_data'):
        h.update(getattr(carrier.model,name).tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    cfg,meta=load_rotation_config(args.checkpoint)
    if cfg.fingerprint()!=RotationConfig().fingerprint() or checkpoint_steps(args.checkpoint,check_weights=True)!=100000:
        raise ValueError('Require the frozen100000-step15deg/r0.3 rotation checkpoint, selected by step, not score')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),acceptance_met=False,
        continuous_rotation_accepted=False,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_steps=100000,checkpoint_metadata=meta,source_revision=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(seeds=SEEDS,directions=[-1,1],maximum_episodes=14,formal_seeds_used=False,
            prefix='Actual empty reach, manual release,140 frozen grasp actions; all failures count without retry',
            pairing='Same seed/placement for both directions; exact physical/controller/camera/RNG/model signature required',
            action_audit='Counterfactual copy changes only active G entries; real preparation history kept, alternate action never applied',
            credit='Actor sensitivity is not physical controllability; world-Z progress and off-axis motion from actual dynamics, no acceptance'),
        cases=[],pairs=[])
    policy=PPO.load(args.checkpoint,device='cpu')
    env=PenRotateEnv(cfg)
    try:
        for index,seed in enumerate(SEEDS):
            expected=None
            pair=[]
            for direction in (-1,1):
                obs,initial=env.reset(seed=seed,options=dict(placement_index=index,direction=direction))
                signature=paired_signature(env.carrier)
                expected=signature if expected is None else expected
                if expected!=signature:
                    raise RuntimeError('Opposite-goal physical preparation mismatch')
                start=env.pose()[1].copy()
                opposite=yaw_rotation(np.deg2rad(-direction*cfg.angle))@env.goal_capture_rotation
                row=dict(seed=seed,direction=direction,initial=initial,initial_observation=obs.tolist(),
                    prefix_physical_signature=signature,start_rotation=start.tolist(),goal=env.goal.tolist(),samples=[])
                report['cases'].append(row)
                pair.append(row)
                for tick in range(cfg.horizon):
                    before=physical_control_signature(env.carrier)
                    alternate,mask=counterfactual_goal(obs,env.goal,opposite)
                    action=policy.predict(obs,deterministic=True)[0]
                    counter=policy.predict(alternate,deterministic=True)[0]
                    source=env.prefix.predict(obs,deterministic=True)[0]
                    source_counter=env.prefix.predict(alternate,deterministic=True)[0]
                    if before!=physical_control_signature(env.carrier):
                        raise RuntimeError('Actor audit mutated physical/controller/camera/RNG state')
                    obs,reward,done,truncated,info=env.step(action)
                    rotation=env.pose()[1]
                    row['samples'].append(dict(tick=tick+1,action=action.tolist(),counterfactual_action=counter.tolist(),
                        active_goal_frames=mask.tolist(),action_goal_delta_norm=float(np.linalg.norm(action-counter)),
                        source_action=source.tolist(),source_counterfactual_action=source_counter.tolist(),
                        source_goal_delta_norm=float(np.linalg.norm(source-source_counter)),
                        command=env.carrier.codec.q.tolist(),rotation=rotation.tolist(),reward=reward,**info))
                    if done or truncated:
                        break
                row.update(end=info,end_rotation=rotation.tolist(),**rotation_components(start,rotation,direction))
                write(args.output/'probe.json',report)
            # Observations should differ only in the latest goal; preparation
            # history is real and must be identical for the two directions.
            negative,positive=(np.array(r['initial_observation'],dtype=np.float32) for r in pair)
            changed,_=counterfactual_goal(negative,np.array(pair[0]['goal']),np.array(pair[1]['goal']))
            if not np.array_equal(changed,positive):
                raise RuntimeError('Goal pairing changed non-goal observation fields or preparation history')
            report['pairs'].append(dict(seed=seed,prefix_signature=expected,
                same_prefix_success=pair[0]['initial']['prefix_grasp_success']==pair[1]['initial']['prefix_grasp_success'],
                only_active_goal_observation_changed=True,
                initial_goal_action_delta_norm=pair[0]['samples'][0]['action_goal_delta_norm'],
                source_initial_goal_action_delta_norm=pair[0]['samples'][0]['source_goal_delta_norm'],
                signed_yaw_progress_deg=[r['signed_goal_yaw_progress_deg'] for r in pair]))
        valid=[r for r in report['cases'] if r['initial']['prefix_grasp_success']]
        samples=[s for r in valid for s in r['samples']]
        def mean(values):
            return float(np.mean(values)) if values else None
        report['summary']=dict(episodes=len(report['cases']),prefix_successes=len(valid),
            successes=sum(r['end']['is_success'] for r in report['cases']),drops=sum(r['end']['dropped'] for r in report['cases']),
            valid_prefix_diagnostics=dict(cases=len(valid),
                mean_signed_goal_yaw_progress_deg=mean([r['signed_goal_yaw_progress_deg'] for r in valid]),
                mean_off_axis_rotation_deg=mean([r['off_axis_rotation_deg'] for r in valid]),
                mean_final_error_deg=mean([r['end']['angle_error_deg'] for r in valid]),
                mean_action_norm=mean([float(np.linalg.norm(s['action'])) for s in samples]),
                mean_goal_action_delta_norm=mean([s['action_goal_delta_norm'] for s in samples]),
                mean_source_goal_action_delta_norm=mean([s['source_goal_delta_norm'] for s in samples])),
            limits='Goal-action sensitivity does not prove correct direction, local reachability or a causal learning failure mechanism')
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
        write(args.output/'probe.json',report)
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        write(args.output/'probe.json',report)
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
