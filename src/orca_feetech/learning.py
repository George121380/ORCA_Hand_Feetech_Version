from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .backends import SimBackend
from .model import JOINT_NAMES, MODEL_ID, ModelSpec, poses
from .paths import provenance


class GestureTrackingEnv(gym.Env):
    """Goal-conditioned control of the official ORCA v1 MuJoCo hand.

    Actions are absolute position setpoints over the gesture envelope. Joints
    constant across gestures remain fixed; all 16 finger channels are retained.
    This task deliberately uses state observations, not RGB observations.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(self, render_mode=None, horizon=100, moving_goal=False):
        self.backend = SimBackend(render_mode=render_mode, rate=20)
        self.render_mode = render_mode
        self.spec = ModelSpec.load()
        self.bank = np.stack([poses()[n] for n in ("open", "fist", "pinch")])
        self.action_low = self.bank[:, :16].min(axis=0)
        self.action_high = self.bank[:, :16].max(axis=0)
        self.action_space = gym.spaces.Box(-1, 1, (16,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (64,), dtype=np.float32)
        self.horizon, self.moving_goal = horizon, moving_goal
        self.goal = self.bank[0].copy()
        self.previous = self.spec.neutral[:16].copy()
        self.steps = self.stable_steps = 0
        self.succeeded = False

    def _normalize(self, q):
        span = self.spec.upper[:16] - self.spec.lower[:16]
        return 2 * (q - self.spec.lower[:16]) / span - 1

    def _observation(self):
        q = self.backend.read().positions[:16]
        velocity = self.backend.env.data.qvel[self.backend.vadr][:16]
        return np.concatenate([self._normalize(q), np.clip(velocity / 5, -5, 5),
                               self._normalize(self.goal[:16]), self._normalize(self.previous)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        weights = self.np_random.dirichlet(np.ones(3))
        initial = weights @ self.bank
        if "goal" in options:
            self.goal = self.spec.clip(options["goal"])
        elif self.np_random.random() < 0.5:
            self.goal = self.bank[self.np_random.integers(3)].copy()
        else:
            self.goal = self.np_random.dirichlet(np.ones(3)) @ self.bank
        self.goal[-1] = self.spec.neutral[-1]
        self.start_goal = self.goal.copy()
        self.end_goal = self.np_random.dirichlet(np.ones(3)) @ self.bank
        self.backend.reset(initial)
        self.previous = initial[:16].copy()
        self.steps = self.stable_steps = 0
        self.succeeded = False
        return self._observation(), {"goal": self.goal.copy()}

    def decode(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (16,) or not np.isfinite(a).all():
            raise ValueError("Expected 16 finite PPO action values")
        q = self.spec.neutral.copy()
        q[:16] = self.action_low + (np.clip(a, -1, 1) + 1) / 2 * (self.action_high - self.action_low)
        return q

    def baseline_action(self):
        span = self.action_high - self.action_low
        return np.divide(2 * (self.goal[:16] - self.action_low), span,
                         out=np.ones(16), where=span > 1e-9) - 1

    def step(self, action):
        command = self.decode(action)
        # Position setpoints share the deployment velocity bound.
        delta = np.clip(command[:16] - self.previous, -0.8 * self.backend.dt, 0.8 * self.backend.dt)
        command[:16] = self.previous + delta
        self.backend.write(command)
        self.previous = command[:16].copy()
        self.steps += 1
        if self.moving_goal:
            u = min(1.0, self.steps / self.horizon)
            self.goal = (1 - u) * self.start_goal + u * self.end_goal
        q = self.backend.read().positions
        error = q[:16] - self.goal[:16]
        mae = float(np.rad2deg(np.abs(error)).mean())
        rmse = float(np.rad2deg(np.sqrt(np.mean(error**2))))
        self.stable_steps = self.stable_steps + 1 if mae <= 5 else 0
        self.succeeded |= self.stable_steps * self.backend.dt >= 0.5
        reward = float(np.exp(-8 * np.mean(error**2)) - 0.01 * np.mean(delta**2))
        info = {"mae_deg": mae, "rmse_deg": rmse, "is_success": self.succeeded}
        return self._observation(), reward, False, self.steps >= self.horizon, info

    def render(self):
        return self.backend.render()

    def close(self):
        self.backend.close()


def train(output, *, steps=300_000, seed=0, num_envs=4, device="cpu", resume=None,
          moving_goal=False):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    torch.set_num_threads(1)
    if steps <= 0 or num_envs <= 0:
        raise ValueError("steps and num_envs must be positive")
    output = Path(output)
    if (output / "config.json").exists() and resume is None:
        raise ValueError("Training output already exists; use a new directory or --resume")
    if resume:
        validate_checkpoint(resume)
    output.mkdir(parents=True, exist_ok=True)
    def factory():
        return GestureTrackingEnv(moving_goal=moving_goal)
    vec_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    env = vec_cls([factory for _ in range(num_envs)])
    env.seed(seed)
    evaluation = GestureTrackingEnv(moving_goal=moving_goal)
    evaluation.reset(seed=10_000 + seed)
    config = {"model_id": MODEL_ID, "joint_names": JOINT_NAMES, "observation":
              "q_normalized, dq/5, goal_normalized, previous_command_normalized",
              "normalization": "fixed model limits, no running statistics",
              "seed": seed, "steps": steps, "num_envs": num_envs, "device": device,
              "moving_goal": moving_goal, "upstream": provenance()}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if resume:
        model = PPO.load(resume, env=env, device=device, tensorboard_log=str(output / "tensorboard"))
    else:
        model = PPO("MlpPolicy", env, seed=seed, device=device, n_steps=256,
                    batch_size=256, learning_rate=3e-4, gamma=0.98, gae_lambda=0.95,
                    policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
                    tensorboard_log=str(output / "tensorboard"), verbose=1)
    callbacks = [
        CheckpointCallback(save_freq=max(1, 25_000 // num_envs), save_path=str(output), name_prefix="checkpoint"),
        EvalCallback(evaluation, best_model_save_path=str(output), log_path=str(output),
                     eval_freq=max(1, 10_000 // num_envs), n_eval_episodes=10, deterministic=True),
    ]
    try:
        model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=resume is None, log_interval=10)
    finally:
        model.save(output / "policy.zip")
        env.close()
        evaluation.close()
    return output / "policy.zip"


def validate_checkpoint(checkpoint):
    path = Path(checkpoint).resolve()
    config_path = path.parent / "config.json"
    if not config_path.exists():
        raise ValueError("Checkpoint needs its adjacent config.json (model/order/normalization)")
    config = json.loads(config_path.read_text())
    if config["model_id"] != MODEL_ID or tuple(config["joint_names"]) != JOINT_NAMES:
        raise ValueError("Checkpoint embodiment mismatch")
    if config["upstream"] != provenance():
        raise ValueError("Checkpoint was trained with a different upstream lock")
    return config


def evaluate(output, *, checkpoint=None, episodes=100, seed=20_000, video=False,
             baseline=False, random_policy=False, moving_goal=False):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        validate_checkpoint(checkpoint)
    model = PPO.load(checkpoint, device="cpu") if checkpoint else None
    if not (model or baseline or random_policy):
        raise ValueError("Select --checkpoint, --baseline or --random")
    env = GestureTrackingEnv(render_mode="rgb_array" if video else None, moving_goal=moving_goal)
    rng = np.random.default_rng(seed)
    rows, traces = [], []
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(output / "evaluation.mp4", fps=20, macro_block_size=1)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            errors = []
            for step in range(env.horizon):
                action = (model.predict(obs, deterministic=True)[0] if model else
                          env.baseline_action() if baseline else rng.uniform(-1, 1, 16))
                obs, reward, terminated, truncated, info = env.step(action)
                errors.append(info["mae_deg"])
                if episode < 3:
                    traces.append({"episode": episode, "time_s": (step + 1) * env.backend.dt,
                                   "target": env.goal.tolist(), "actual": env.backend.read().positions.tolist(),
                                   **info})
                    if writer:
                        writer.append_data(env.render())
                if terminated or truncated:
                    break
            rows.append({"seed": seed + episode, "success": bool(info["is_success"]),
                         "mean_mae_deg": float(np.mean(errors)), "final_mae_deg": errors[-1]})
    finally:
        env.close()
        if writer:
            writer.close()
    summary = {"policy": "ppo" if model else "baseline" if baseline else "random",
               "checkpoint": str(checkpoint) if checkpoint else None, "episodes": episodes,
               "success_rate": float(np.mean([r["success"] for r in rows])),
               "mean_mae_deg": float(np.mean([r["mean_mae_deg"] for r in rows])),
               "final_mae_deg": float(np.mean([r["final_mae_deg"] for r in rows])),
               "moving_goal": moving_goal,
               "acceptance": "MAE <= 5 degrees for 0.5 seconds within each 5-second episode",
               "results": rows}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "traces.json").write_text(json.dumps(traces) + "\n")
    plot_evaluation(output, traces)
    return {k: v for k, v in summary.items() if k != "results"}


def plot_evaluation(output, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for episode in sorted({r["episode"] for r in traces}):
        records = [r for r in traces if r["episode"] == episode]
        axes[0].plot([r["time_s"] for r in records], [r["mae_deg"] for r in records], label=f"Episode {episode + 1}")
    axes[0].axhline(5, color="black", linestyle="--", linewidth=1, label="5 degree threshold")
    axes[0].set(title="Gesture tracking error", xlabel="Time (s)", ylabel="Mean absolute error (degrees)")
    axes[0].legend()
    records = [r for r in traces if r["episode"] == 0]
    index = JOINT_NAMES.index("index_mcp")
    for field, label in (("target", "Target"), ("actual", "Simulated feedback")):
        axes[1].plot([r["time_s"] for r in records],
                     [np.rad2deg(r[field][index]) for r in records], label=label)
    axes[1].set(title="Index MCP — first evaluation episode", xlabel="Time (s)", ylabel="Joint position (degrees)")
    axes[1].legend()
    fig.savefig(Path(output) / "tracking.png", dpi=160)
    plt.close(fig)
