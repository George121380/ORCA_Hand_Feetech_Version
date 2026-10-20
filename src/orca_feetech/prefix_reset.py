"""Training-only physical prefix curriculum; canonical evaluation is unchanged."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np

from .checkpoint import load_config
from .manipulation import ManipulationEnv
from .training_supervision import checkpoint_steps

PREFIX_INFO = ("training_prefix_used", "training_prefix_steps", "training_prefix_attempts",
               "training_prefix_rollout_steps")


def validate_prefix_checkpoint(checkpoint, config):
    checkpoint = Path(checkpoint).resolve()
    previous, _ = load_config(checkpoint)
    if (config.object != "pen" or config.angle != 30 or config.turns != 2
            or config.horizon != 480 or config.randomization != 1
            or previous.turns != 1 or previous.horizon != 240
            or any(asdict(previous)[key] != value for key,value in asdict(config).items()
                   if key not in ("turns", "horizon"))):
        raise ValueError("Prefix curriculum requires matched pen30/r1 single-to-double subgoal contracts")
    steps = checkpoint_steps(checkpoint,check_weights=True)
    return dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((checkpoint.parent/"config.json").read_bytes()).hexdigest(),
                source_steps=steps,maximum_prefix_steps=240,maximum_attempts=5,
                actor_observation_unchanged=True,evaluation_assistance=False)


def load_frozen_teacher(checkpoint, config):
    from stable_baselines3 import PPO
    validate_prefix_checkpoint(checkpoint,config)
    teacher = PPO.load(checkpoint,device="cpu")
    if teacher.observation_space.shape != (296,) or teacher.action_space.shape != (16,):
        raise ValueError("Prefix teacher must use the canonical raw observation and action")
    teacher.policy.set_training_mode(False)
    teacher.policy.requires_grad_(False)
    return teacher


def make_training_environment(config,*,evaluation=False,negative_goal_probability=.5,
                              minimum_goal_angle=None,prefix_checkpoint=None,
                              prefix_probability=.75,seed=0):
    """Evaluation ignores every training-only distribution/teacher option."""
    env = ManipulationEnv(config,negative_goal_probability=.5 if evaluation else negative_goal_probability,
                          minimum_goal_angle=None if evaluation else minimum_goal_angle)
    if not evaluation and prefix_checkpoint is not None:
        env = PhysicalPrefixReset(env,load_frozen_teacher(prefix_checkpoint,config),
                                  probability=prefix_probability,seed=seed)
    return env


class PhysicalPrefixReset(gym.Wrapper):
    """Start some training episodes after one real frozen-policy subgoal.

    The returned observation is the exact last observation produced by
    env.step, with its real history, time budget, origin, goal and controller
    state. No simulated state is teleported, zeroed, rebased or supplied as
    an additional actor input. Failed prefix attempts have a fixed cap and
    fall back to an ordinary neutral reset.
    """
    def __init__(self,env,teacher,probability=.75,seed=0):
        super().__init__(env)
        if not 0 <= probability <= 1:
            raise ValueError("Prefix probability must lie in[0,1]")
        self.teacher, self.probability = teacher, probability
        self.rng = np.random.default_rng(np.random.SeedSequence([seed,0x504658]))
        self.prefix_used = self.prefix_steps = self.prefix_attempts = self.prefix_rollout_steps = 0

    def annotate(self,info):
        return dict(info,**dict(zip(PREFIX_INFO,(self.prefix_used,self.prefix_steps,
                                               self.prefix_attempts,self.prefix_rollout_steps))))

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed,0x504658]))
        self.prefix_used = self.prefix_steps = self.prefix_attempts = self.prefix_rollout_steps = 0
        obs, info = self.env.reset(seed=seed,options=options)
        if self.rng.random() >= self.probability:
            return obs,self.annotate(info)
        for attempt in range(5):
            if attempt:
                obs, info = self.env.reset(options=options)
            self.prefix_attempts += 1
            for _ in range(240):
                action = self.teacher.predict(obs,deterministic=True)[0]
                obs, _, done, truncated, info = self.env.step(action)
                self.prefix_rollout_steps += 1
                if info["completed_turns"] == 1 and not (done or truncated or info["dropped"]):
                    self.prefix_used = 1
                    self.prefix_steps = self.env.unwrapped.steps
                    # step's info describes the just-completed old goal;
                    # reset must describe the current, second target instead.
                    return obs,self.annotate(self.env.unwrapped.info())
                if done or truncated:
                    break
        obs, info = self.env.reset(options=options)
        return obs,self.annotate(info)

    def step(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        return obs,reward,done,truncated,self.annotate(info)
