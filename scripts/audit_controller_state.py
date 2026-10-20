"""Audit command-limit filter state on development seeds, without changing dynamics."""
import argparse
import json
from pathlib import Path

from orca_feetech.checkpoint import load_config, load_v6_config_for_migration
from orca_feetech.manipulation import ManipulationEnv, HISTORY, FRAME_SIZE
from orca_feetech.policy_migration import legacy_columns


def main():
    import torch
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-v6", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    if args.seed < 20000 or args.episodes < 1 or args.output.exists():
        parser.error("Use development seeds >=20000, positive episodes, and a new output path")
    torch.set_num_threads(1)
    config, _ = (load_v6_config_for_migration if args.legacy_v6 else load_config)(args.checkpoint)
    policy = PPO.load(args.checkpoint, device="cpu")
    env = ManipulationEnv(config)
    records = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            obs, info = env.reset(seed=seed)
            saturated = joint_steps = longest = consecutive = 0
            for step in range(config.horizon):
                velocity, q = env.codec.velocity, env.codec.q[:16]
                outward = (((q <= env.lower[:16]+1e-8) & (velocity < -.1))
                            | ((q >= env.upper[:16]-1e-8) & (velocity > .1)))
                count = int(outward.sum())
                saturated += bool(count)
                joint_steps += count
                consecutive = consecutive+1 if count else 0
                longest = max(longest, consecutive)
                assert obs.shape == (HISTORY * FRAME_SIZE,)
                inputs = obs[legacy_columns()] if args.legacy_v6 else obs
                action = policy.predict(inputs, deterministic=True)[0]
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            records.append(dict(seed=seed, success=info["is_success"], steps=step+1,
                                windup_steps=saturated, windup_joint_steps=joint_steps,
                                longest_windup=longest))
    finally:
        env.close()
    groups = []
    for success in (True, False):
        group = [r for r in records if r["success"] == success]
        if group:
            groups.append(dict(success=success, episodes=len(group),
                               windup_episode_count=sum(r["windup_steps"] > 0 for r in group),
                               windup_step_fraction=sum(r["windup_steps"] for r in group)
                               / sum(r["steps"] for r in group),
                               max_consecutive_windup=max(r["longest_windup"] for r in group)))
    result = dict(checkpoint=str(args.checkpoint.resolve()), legacy_v6=args.legacy_v6,
                  purpose="Development-only audit; correlation is not evidence of causation",
                  summary=groups, episodes=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
