"""Paired development-only angle/subgoal probes; never formal stage acceptance."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv


def probe_configs(config, angles, subgoals=(1,), horizon_per_subgoal=None):
    if (len(set(angles)) != len(angles) or len(set(subgoals)) != len(subgoals)
            or not 1 <= len(angles)*len(subgoals) <= 8):
        raise ValueError("Use1..8 distinct angle/subgoal combinations")
    horizon = config.horizon if horizon_per_subgoal is None else horizon_per_subgoal
    if not 20 <= horizon <= 240:
        raise ValueError("Use20..240 control ticks per subgoal")
    return [replace(config,angle=angle,turns=count,horizon=horizon*count)
            for angle in angles for count in subgoals]


def main():
    import torch
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--subgoals", type=int, nargs="+", default=[1],
                        help="Number of angle increments, not full360-degree revolutions")
    parser.add_argument("--horizon-per-subgoal", type=int)
    args = parser.parse_args()
    if args.seed < 20000 or not 1 <= args.episodes <= 200:
        parser.error("Use1..200 episodes and development seeds >=20000")
    config, meta = load_config(args.checkpoint)
    configs = probe_configs(config,args.angles,args.subgoals,args.horizon_per_subgoal)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy = PPO.load(args.checkpoint, device="cpu")
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  purpose="Development-only course design; not formal acceptance or checkpoint selection",
                  source_checkpoint=str(args.checkpoint.resolve()),
                  source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  source_fingerprint=meta["fingerprint"], seed=args.seed,
                  episodes_per_angle=args.episodes, is_formal_evaluation=False, angles=[])
    for cfg in configs:
        env = ManipulationEnv(cfg)
        trials = []
        try:
            for seed in range(args.seed, args.seed+args.episodes):
                obs, info = env.reset(seed=seed)
                initial = dict(initial_position=info["object_position"], direction=env.direction,
                               initial_rotation=env.pose()[1].tolist())
                minimum = info["angle_error_deg"]
                streak = longest = 0
                completion_steps = []
                for step in range(cfg.horizon):
                    action = policy.predict(obs, deterministic=True)[0]
                    obs, _, done, truncated, info = env.step(action)
                    minimum = min(minimum, info["angle_error_deg"])
                    streak = streak+1 if info["stable"] and not info["dropped"] else 0
                    longest = max(longest, streak)
                    if info["completed_turns"] > len(completion_steps):
                        completion_steps.append(step+1)
                    if done or truncated:
                        break
                reason = ("success" if info["is_success"] else "dropped" if info["dropped"] else
                          "vision_timeout" if info["vision_age"] > .25 else "horizon")
                trials.append(dict(seed=seed, steps=step+1, end_reason=reason,
                                   minimum_error_deg=minimum, longest_stable_steps=longest,
                                   subgoal_completion_steps=completion_steps,
                                   **initial, **info))
        finally:
            env.close()
        metrics = dict(angle=cfg.angle, subgoals=cfg.turns, scheduled_yaw_deg=cfg.angle*cfg.turns,
                       probe_task=asdict(cfg), episodes=len(trials),
                       success_rate=sum(t["is_success"] for t in trials)/len(trials),
                       drop_rate=sum(t["dropped"] for t in trials)/len(trials),
                       mean_final_error_deg=float(np.mean([t["angle_error_deg"] for t in trials])),
                       end_reasons=dict(Counter(t["end_reason"] for t in trials)),
                       completed_subgoals_histogram=dict(Counter(t["completed_turns"] for t in trials)),
                       trials=trials)
        report["angles"].append(metrics)
        temporary = args.output / "probe.tmp"
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(args.output / "probe.json")
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("probe_task", "trials")}), flush=True)


if __name__ == "__main__":
    main()
