"""Calibrate gSDE only on training seeds, then verify deterministic-policy parity."""
import argparse
import json
from pathlib import Path

import numpy as np

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv
from orca_feetech.policy_exploration import migrate_sde_checkpoint
from orca_feetech.training_supervision import checkpoint_steps


def main():
    import torch
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-frequency", type=int, default=4)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--calibration-episodes", type=int, default=16)
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.calibration_episodes <= 32
            or not 1 <= args.sample_frequency <= 20
            or not 0 <= args.calibration_seed < args.calibration_seed+args.calibration_episodes <= 10000):
        parser.error("Use a new output and at most32 training seeds below10000")
    config, _ = load_config(args.source)
    checkpoint_steps(args.source, check_weights=True)
    torch.set_num_threads(1)
    model = PPO.load(args.source, device="cpu")
    if model.use_sde:
        parser.error("Expected an ordinary Gaussian source")
    env = ManipulationEnv(config)
    observations = []
    try:
        for seed in range(args.calibration_seed, args.calibration_seed+args.calibration_episodes):
            obs, _ = env.reset(seed=seed)
            observations.append(obs.copy())
            for step in range(config.horizon):
                obs, _, done, truncated, _ = env.step(model.predict(obs, deterministic=True)[0])
                if (step+1) % 20 == 0:
                    observations.append(obs.copy())
                if done or truncated:
                    break
    finally:
        env.close()
    proof = migrate_sde_checkpoint(args.source, args.output, np.asarray(observations),
                                   sample_frequency=args.sample_frequency)
    proof.update(calibration_seed=args.calibration_seed, calibration_episodes=args.calibration_episodes,
                 calibration_purpose="Training observations only; no held-out data or new truth inputs")
    metadata_path = args.output/"config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sde_migration"] = proof
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
