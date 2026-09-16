"""Train or evaluate contact-based pen reorientation on ORCA v1.

Examples:
  .venv/bin/python scripts/pen_task.py train --steps 120000 --output runs/pen-ppo
  .venv/bin/python scripts/pen_task.py evaluate --checkpoint runs/pen-ppo/policy.zip --video --output runs/pen-eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.pen import TASK_ID, PenReorientationEnv, build_pen_scene


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + "\n")


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    torch.set_num_threads(1)
    if args.resume:
        previous = json.loads((args.resume.parent / "config.json").read_text())
        if (previous["task_id"] != TASK_ID or previous["upstream"] != provenance()
                or previous["turn_degrees"] != args.angle or tuple(previous["joint_names"]) != JOINT_NAMES):
            raise ValueError("Resume checkpoint task/model/angle/lock mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    build_pen_scene()
    config = {"task_id": TASK_ID, "joint_names": JOINT_NAMES, "turn_degrees": args.angle,
              "requested_steps": args.steps, "seed": args.seed, "num_envs": args.num_envs,
              "resume_from": str(args.resume) if args.resume else None,
              "observations": "69D simulator state: finger q,dq,previous command; pen relative position,axis,velocity; goal axis; six hand-contact flags",
              "actions": "16 finger setpoint increments, bounded to 0.8 rad/s, fixed wrist",
              "acceptance": "axis error <10 deg, displacement <4 cm, hand contact, linear speed <0.1 m/s and angular speed <1 rad/s, held for 0.5 s",
              "upstream": provenance()}
    json_write(args.output / "config.json", config)

    def factory(index):
        def make():
            return Monitor(PenReorientationEnv(turn_degrees=args.angle), str(args.output / f"env{index}.monitor.csv"),
                           info_keywords=("is_success", "angle_error_deg", "dropped"))
        return make

    class Progress(BaseCallback):
        def _on_step(self):
            if self.n_calls % 512 == 0:
                json_write(args.output / "progress.json", {"steps": self.num_timesteps,
                    "recent_episodes": list(self.model.ep_info_buffer)})
            return True

    env = SubprocVecEnv([factory(i) for i in range(args.num_envs)])
    env.seed(args.seed)
    model = PPO.load(args.resume, env=env, device="cpu") if args.resume else PPO("MlpPolicy", env, seed=args.seed, device="cpu", n_steps=256, batch_size=256,
                learning_rate=3e-4, gamma=.99, gae_lambda=.95, ent_coef=.005,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}}, verbose=1)
    try:
        model.learn(total_timesteps=args.steps, callback=[Progress(), CheckpointCallback(
            save_freq=max(1, 25000 // args.num_envs), save_path=str(args.output), name_prefix="checkpoint")],
            log_interval=10, reset_num_timesteps=args.resume is None)
        config["actual_steps"] = model.num_timesteps
        json_write(args.output / "config.json", config)
        model.save(args.output / "policy.zip")
    finally:
        env.close()


def evaluate(args):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    model = None
    if args.checkpoint:
        config = json.loads((args.checkpoint.parent / "config.json").read_text())
        if config["task_id"] != TASK_ID or config["upstream"] != provenance() or tuple(config["joint_names"]) != JOINT_NAMES:
            raise ValueError("Pen checkpoint task/model/lock mismatch")
        if args.angle != config["turn_degrees"]:
            raise ValueError("Evaluation angle differs from checkpoint; choose its trained angle")
        model = PPO.load(args.checkpoint, device="cpu")
    env = PenReorientationEnv(turn_degrees=args.angle)
    records, traces = [], []
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.output / "evaluation.mp4", fps=20, macro_block_size=1)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            rng = np.random.default_rng(args.seed + episode)
            minimum_angle = 180.
            for step in range(env.horizon):
                action = model.predict(obs, deterministic=True)[0] if model else (
                    rng.uniform(-1, 1, 16) if args.random else np.zeros(16))
                obs, reward, terminated, truncated, info = env.step(action)
                minimum_angle = min(minimum_angle, info["angle_error_deg"])
                if episode < 3:
                    traces.append({"episode": episode, "time_s": info["elapsed_s"],
                                   "angle_error_deg": info["angle_error_deg"],
                                   "displacement_m": info["displacement_m"], "hand_contact": info["hand_contact"],
                                   "is_success": info["is_success"], "dropped": info["dropped"]})
                    if writer:
                        import cv2
                        frame = env.render()
                        label = f'{"PPO" if model else "Random" if args.random else "Hold"} | trial {episode+1} | angle error {info["angle_error_deg"]:.1f} deg'
                        cv2.rectangle(frame, (0, 0), (640, 33), (20, 32, 38), -1)
                        cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .53, (245, 245, 245), 1)
                        writer.append_data(frame)
                if terminated or truncated:
                    break
            records.append({"seed": args.seed + episode, "success": info["is_success"],
                            "dropped": info["dropped"], "final_angle_error_deg": info["angle_error_deg"],
                            "best_angle_error_deg": minimum_angle, "duration_s": info["elapsed_s"]})
    finally:
        env.close()
        if writer:
            writer.close()
    summary = {"task_id": TASK_ID, "policy": "ppo" if model else "random" if args.random else "hold",
               "turn_degrees": args.angle, "episodes": args.episodes,
               "success_rate": float(np.mean([r["success"] for r in records])),
               "drop_rate": float(np.mean([r["dropped"] for r in records])),
               "mean_final_angle_error_deg": float(np.mean([r["final_angle_error_deg"] for r in records])),
               "mean_best_angle_error_deg": float(np.mean([r["best_angle_error_deg"] for r in records])),
               "mean_episode_duration_s": float(np.mean([r["duration_s"] for r in records])),
               "results": records}
    json_write(args.output / "metrics.json", summary)
    json_write(args.output / "traces.json", traces)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=120000)
    t.add_argument("--num-envs", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", type=Path)
    e = sub.add_parser("evaluate")
    choice = e.add_mutually_exclusive_group()
    choice.add_argument("--checkpoint", type=Path)
    choice.add_argument("--random", action="store_true")
    e.add_argument("--episodes", type=int, default=32)
    e.add_argument("--seed", type=int, default=20000)
    e.add_argument("--video", action="store_true")
    for cmd in (t, e):
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--angle", type=float, default=90)
    args = parser.parse_args()
    if args.command == "train" and (args.steps <= 0 or args.num_envs <= 0):
        parser.error("Steps and num-envs must be positive")
    if args.command == "evaluate" and args.episodes <= 0:
        parser.error("Episodes must be positive")
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
