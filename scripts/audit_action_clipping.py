"""Quantify Gaussian clipping on development-only deterministic-rollout states."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv


def main():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    if args.seed < 20000 or args.episodes < 1 or args.output.exists():
        parser.error("Use development seeds >=20000, positive episodes and a new output")
    config, _ = load_config(args.checkpoint)
    torch.set_num_threads(1)
    policy = PPO.load(args.checkpoint, device="cpu")
    if policy.policy.squash_output or type(policy.policy.action_dist) is not DiagGaussianDistribution:
        parser.error("Audit expects an unsquashed diagonal Gaussian")
    means, stds, probabilities, trials = [], [], [], []
    env = ManipulationEnv(config)
    try:
        for seed in range(args.seed, args.seed+args.episodes):
            obs, info = env.reset(seed=seed)
            for step in range(config.horizon):
                with torch.no_grad():
                    dist = policy.policy.get_distribution(torch.from_numpy(obs[None])).distribution
                    mean, std = dist.mean[0], dist.stddev[0]
                    probability = dist.cdf(torch.full_like(dist.mean, -1))[0] + 1 \
                        - dist.cdf(torch.full_like(dist.mean, 1))[0]
                means.append(mean.numpy())
                stds.append(std.numpy())
                probabilities.append(probability.numpy())
                obs, _, done, truncated, info = env.step(np.clip(mean.numpy(), -1, 1))
                if done or truncated:
                    break
            trials.append(dict(seed=seed, steps=step+1, success=info["is_success"]))
    finally:
        env.close()
    mean, std = np.array(means), np.array(stds)
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(args.checkpoint.resolve()),
                  source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  purpose="Development-state theoretical Gaussian clip mass, not a formal policy test",
                  states=len(means), mean_outside_bounds_fraction=float(np.mean(np.abs(mean) > 1)),
                  mean_beyond_three_std_fraction=float(np.mean(np.abs(mean) > 1+3*std)),
                  expected_sample_clip_fraction=float(np.mean(probabilities)),
                  max_abs_latent_mean=float(np.abs(mean).max()), mean_std=float(std.mean()),
                  latent_abs_mean_percentiles=np.percentile(np.abs(mean), [50, 90, 99]).tolist(),
                  trials=trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
