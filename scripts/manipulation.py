"""Train/evaluate the printable-object task; long training belongs on Linux/4090."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv, TaskConfig, TASK_ID, HISTORY, FRAME_SIZE
from orca_feetech.model import JOINT_NAMES
from orca_feetech.objects import specification, export_objects
from orca_feetech.paths import provenance


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)




def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run setup_4090.sh and inspect the NVIDIA driver")
    config = TaskConfig(object=args.object, angle=args.angle, turns=args.turns,
                        horizon=args.horizon, randomization=args.randomization,
                        mass_kg=args.mass_kg, friction=args.friction,
                        delay_max_steps=args.delay_max_ms // 50,
                        command_bias_deg=args.command_bias_deg, encoder_bias_deg=args.encoder_bias_deg)
    if args.device_config:
        from orca_feetech.hardware import load_device
        _, _, lo, hi = load_device(args.device_config)
        config.lower, config.upper = lo.tolist(), hi.tolist()
        config.__post_init__()
    if args.resume:
        previous, _ = load_config(args.resume)
        # The curriculum may change angle, turns, horizon and randomization.
        # Physical/observation/action definitions must remain fixed.
        keys = ("object", "mass_kg", "friction", "max_velocity", "max_acceleration", "lower", "upper")
        if any(asdict(previous)[key] != asdict(config)[key] for key in keys):
            raise ValueError("Resume changes physical/action contract; use the same measured config")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = dict(task_id=TASK_ID, task=asdict(config), fingerprint=config.fingerprint(),
                    joint_names=JOINT_NAMES, history=HISTORY, frame_size=FRAME_SIZE,
                    upstream=provenance(), geometry=specification(config.object), seed=args.seed,
                    num_envs=args.num_envs, device=args.device, requested_steps=args.steps,
                    resume=str(args.resume) if args.resume else None, git_revision=revision,
                    platform=platform.platform(), torch_version=torch.__version__,
                    cuda=torch.version.cuda, physics="native MuJoCo CPU; PPO on selected device",
                    learning_rate=args.learning_rate, entropy_coef=args.entropy_coef,
                    target_kl=args.target_kl, initial_log_std=args.initial_log_std,
                    reset_log_std=args.reset_log_std,
                    training_negative_goal_probability=args.negative_goal_probability,
                    trained=False)
    write(args.output / "config.json", metadata)

    def factory(index, evaluation=False):
        def make():
            return Monitor(ManipulationEnv(config, negative_goal_probability=(
                .5 if evaluation else args.negative_goal_probability)),
                           str(args.output / ("eval" if evaluation else f"env{index}")),
                           info_keywords=("is_success", "dropped", "angle_error_deg", "completed_turns"))
        return make

    env = evaluation = None
    model = None
    try:
        env = (SubprocVecEnv([factory(i) for i in range(args.num_envs)], start_method="spawn")
               if args.num_envs > 1 else DummyVecEnv([factory(0)]))
        env.seed(args.seed)
        evaluation = DummyVecEnv([factory(0, True)])
        evaluation.seed(10000 + args.seed)
        model = (PPO.load(args.resume, env=env, device=args.device,
                          batch_size=min(1024, 256 * args.num_envs), learning_rate=args.learning_rate,
                          ent_coef=args.entropy_coef, target_kl=args.target_kl) if args.resume else
                 PPO("MlpPolicy", env, seed=args.seed, device=args.device, n_steps=256,
                     batch_size=min(1024, 256 * args.num_envs), n_epochs=5, learning_rate=args.learning_rate,
                     ent_coef=args.entropy_coef, gamma=.99, gae_lambda=.95, target_kl=args.target_kl,
                     policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                                    "log_std_init": args.initial_log_std},
                     tensorboard_log=str(args.output / "tb"), verbose=1))
        if args.reset_log_std is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(args.reset_log_std)
        model.tensorboard_log = str(args.output / "tb")
        started = time.monotonic()
        initial_steps = model.num_timesteps

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 256 == 0:
                    write(args.output / "progress.json",
                          dict(total_steps=self.num_timesteps,
                               elapsed_s=time.monotonic() - started,
                               steps_per_second=(self.num_timesteps-initial_steps) /
                               max(.001, time.monotonic()-started),
                               recent_episodes=list(self.model.ep_info_buffer)))
                return True

        model.learn(total_timesteps=args.steps, reset_num_timesteps=args.resume is None,
                    callback=[Progress(), CheckpointCallback(
                        save_freq=max(1, args.checkpoint_every // args.num_envs),
                        save_path=str(args.output), name_prefix="checkpoint"),
                        EvalCallback(evaluation, best_model_save_path=str(args.output),
                                     log_path=str(args.output / "evaluation"),
                                     eval_freq=max(1, args.eval_every // args.num_envs),
                                     n_eval_episodes=20, deterministic=True)])
        model.save(args.output / "policy.zip")
        metadata.update(actual_steps=model.num_timesteps, trained=True)
        write(args.output / "config.json", metadata)
    except KeyboardInterrupt:
        if model is not None:
            model.save(args.output / "interrupted.zip")
        raise
    finally:
        if env is not None:
            env.close()
        if evaluation is not None:
            evaluation.close()


def wilson(successes, n):
    z = 1.96
    p = successes / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [float(center-half), float(center+half)]


def evaluate(args):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    config, meta = load_config(args.checkpoint)
    if args.seed < 20000:
        raise ValueError("Use held-out evaluation seeds >=20000")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "config.json", meta)
    policy = PPO.load(args.checkpoint, device="cpu")
    env = ManipulationEnv(config)
    records = []
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.output / "evaluation.mp4", fps=20, macro_block_size=1)
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            initial = dict(direction=env.direction, initial_position=info["object_position"],
                           initial_rotation=env.pose()[1].tolist(), goal_rotation=env.goal.tolist(),
                           reset_attempts=info["reset_attempts"], settle_seconds=info["settle_seconds"],
                           reset_linear_speed_mps=info["linear_speed_mps"],
                           reset_angular_speed_rps=info["angular_speed_rps"])
            rng = np.random.default_rng(args.seed + episode)
            total = 0.
            for step in range(config.horizon):
                action = (np.zeros(16) if args.baseline == "hold" else
                          rng.uniform(-1, 1, 16) if args.baseline == "random" else
                          policy.predict(obs, deterministic=True)[0])
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if writer and episode < 5:
                    import cv2
                    frame = env.render()
                    cv2.putText(frame, f"seed {args.seed+episode} | {info['angle_error_deg']:.1f} deg | "
                                f"turns {info['completed_turns']} | drop {info['dropped']}",
                                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    writer.append_data(frame)
                if terminated or truncated:
                    break
            reason = ("success" if info["is_success"] else "dropped" if info["dropped"] else
                      "vision_timeout" if info["vision_age"] > .25 else "horizon")
            records.append(dict(seed=args.seed+episode, reward=total, steps=step+1,
                                end_reason=reason, **initial, **info))
    finally:
        env.close()
        if writer:
            writer.close()
    successes = sum(r["is_success"] for r in records)
    success_rate = successes / len(records)
    drop_rate = sum(r["dropped"] for r in records) / len(records)
    result = dict(policy=args.baseline or str(args.checkpoint), episodes=len(records),
                  success_rate=success_rate, success_95ci=wilson(successes, len(records)),
                  drop_rate=drop_rate,
                  mean_final_error_deg=float(np.mean([r["angle_error_deg"] for r in records])),
                  acceptance_met=len(records) >= 100 and success_rate >= .8 and drop_rate <= .1,
                  trials=records)
    write(args.output / "metrics.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--object", choices=["cube", "pen"], default="cube")
    t.add_argument("--angle", type=float, default=15)
    t.add_argument("--turns", type=int, default=1)
    t.add_argument("--horizon", type=int, default=240)
    t.add_argument("--randomization", type=float, default=1.)
    t.add_argument("--mass-kg", type=float)
    t.add_argument("--friction", type=float, default=1.)
    t.add_argument("--delay-max-ms", type=int, choices=[0, 50, 100, 150, 200], default=100)
    t.add_argument("--command-bias-deg", type=float, default=2.)
    t.add_argument("--encoder-bias-deg", type=float, default=1.5)
    t.add_argument("--device-config", type=Path, help="Completed hardware YAML: train within measured ROM")
    t.add_argument("--steps", type=int, default=3000000)
    t.add_argument("--num-envs", type=int, default=16)
    t.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--entropy-coef", type=float, default=.01)
    t.add_argument("--target-kl", type=float, default=.03)
    t.add_argument("--initial-log-std", type=float, default=0.)
    t.add_argument("--reset-log-std", type=float, help="Explicit exploration reset for a new fine-tuning stage")
    t.add_argument("--negative-goal-probability", type=float, default=.5,
                   help="Training-only negative-direction frequency; evaluations remain balanced")
    t.add_argument("--resume", type=Path)
    t.add_argument("--checkpoint-every", type=int, default=100000)
    t.add_argument("--eval-every", type=int, default=250000)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=20000)
    e.add_argument("--baseline", choices=["hold", "random"])
    e.add_argument("--video", action="store_true")
    for cmd in (t, e):
        cmd.add_argument("--output", type=Path, required=True)
    x = sub.add_parser("export-objects")
    x.add_argument("--output", type=Path, default=Path("assets/objects/v4"))
    args = p.parse_args()
    if args.command == "export-objects":
        export_objects(args.output)
    elif args.command == "train":
        if min(args.steps, args.num_envs, args.checkpoint_every, args.eval_every) < 1:
            p.error("Training budgets must be positive")
        if not 0 <= args.negative_goal_probability <= 1:
            p.error("Goal probability must be in [0,1]")
        if (not 0 < args.learning_rate <= .01 or not 0 <= args.entropy_coef <= 1
                or not 0 < args.target_kl <= 1 or not -5 <= args.initial_log_std <= 2
                or args.reset_log_std is not None and not -5 <= args.reset_log_std <= 2):
            p.error("Invalid PPO optimization/exploration parameters")
        train(args)
    else:
        if args.episodes < 1:
            p.error("Episodes must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
