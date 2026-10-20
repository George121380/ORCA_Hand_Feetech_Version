"""Reconstruct the seed0 untrained actor using the exact shared PPO factory.

No learning and no formal seeds. This is a reconstructed reference, not a
previously recorded initial checkpoint from the completed pilot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from orca_feetech.pen_grasp import PenGraspEnv,GraspConfig,load_grasp_config
from orca_feetech.training_supervision import checkpoint_steps
from manipulation import fresh_ppo,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    cfg,meta=load_grasp_config(args.template)
    if (cfg.fingerprint()!=GraspConfig(randomization=.3).fingerprint() or meta['seed']!=0
            or meta['num_envs']!=8 or meta['initial_log_std']!=-4):
        raise ValueError('Expected seed0,8-environment low-noise grasp template')
    recipe=SimpleNamespace(**{k:meta[k] for k in ('seed','num_envs','learning_rate','entropy_coef',
                                                'target_kl','initial_log_std')},
        device='cpu',output=args.output,zero_initial_action_mean=False)
    args.output.mkdir(parents=True,exist_ok=False)
    env=DummyVecEnv([lambda:Monitor(PenGraspEnv(cfg)) for _ in range(recipe.num_envs)])
    try:
        model=fresh_ppo(env,recipe)
        assert model.num_timesteps==0
        model.save(args.output/'policy.zip')
        meta.update(git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            device='cpu',resume=None,trained=False,requested_steps=0,actual_steps=0,
            zero_initial_action_mean=False,initialization_reference='reconstructed with shared fresh_ppo factory; no learning',
            template_checkpoint=str(args.template),template_config_sha256=hashlib.sha256(
                (args.template.parent/'config.json').read_bytes()).hexdigest())
        write(args.output/'config.json',meta)
        assert checkpoint_steps(args.output/'policy.zip',check_weights=True)==0
        write(args.output/'freeze.json',dict(checkpoint_sha256=hashlib.sha256(
            (args.output/'policy.zip').read_bytes()).hexdigest(),source_revision=meta['git_revision'],
            recipe=vars(recipe)|{'output':str(recipe.output)},no_learning=True,
            caveat='Reconstruction, not an original pretraining snapshot',formal_seeds_used=False))
        print(json.dumps(dict(output=str(args.output),steps=0,source=meta['git_revision'])))
    finally:
        env.close()


if __name__=='__main__':
    main()
