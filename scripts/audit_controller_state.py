"""Audit command-limit filter state on development seeds, without changing dynamics."""
import argparse
import json
from pathlib import Path

import numpy as np

from orca_feetech.checkpoint import load_config, load_v6_config_for_migration
from orca_feetech.manipulation import ManipulationEnv, ActionCodec, DT, HISTORY, FRAME_SIZE
from orca_feetech.policy_migration import legacy_columns


class LimitProjectionProbeCodec(ActionCodec):
    """Simulation diagnostic only; does not change the canonical/deployed codec.

    At a clipped position target, project the internal filter velocity onto
    the realized command increment. This changes the controller dynamics;
    its results cannot qualify a checkpoint for canonical-task acceptance.
    """
    def step(self, action, valid=True):
        before = self.q[:16].copy()
        command = super().step(action, valid=valid)
        outward = (((command[:16] <= self.lower[:16]) & (self.velocity < 0))
                   | ((command[:16] >= self.upper[:16]) & (self.velocity > 0)))
        self.velocity[outward] = ((command[:16]-before)/DT)[outward]
        return command


def main():
    import torch
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-v6", action="store_true")
    parser.add_argument("--project-limit-velocity", action="store_true",
                        help="Development-only counterfactual, never canonical acceptance")
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
            initial_position = info["object_position"]
            if args.project_limit_velocity:
                env.codec = LimitProjectionProbeCodec(config, env.codec.q)
            saturated = joint_steps = longest = consecutive = 0
            minimum_error = info["angle_error_deg"]
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
                minimum_error = min(minimum_error, info["angle_error_deg"])
                if done or truncated:
                    break
            reason = ("success" if info["is_success"] else "dropped" if info["dropped"] else
                      "vision_timeout" if info["vision_age"] > .25 else "horizon")
            records.append(dict(seed=seed, success=info["is_success"], steps=step+1,
                                windup_steps=saturated, windup_joint_steps=joint_steps,
                                longest_windup=longest, direction=env.direction,
                                initial_position=initial_position, end_reason=reason,
                                minimum_error_deg=minimum_error,
                                final_error_deg=info["angle_error_deg"],
                                shaft_error_deg=info["shaft_error_deg"], dropped=info["dropped"]))
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
                  project_limit_velocity=args.project_limit_velocity, is_formal_evaluation=False,
                  purpose="Development-only controller diagnostic; never canonical task acceptance",
                  success_count=sum(r["success"] for r in records),
                  drop_count=sum(r["dropped"] for r in records),
                  mean_final_error_deg=float(np.mean([r["final_error_deg"] for r in records])),
                  summary=groups, episodes=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
