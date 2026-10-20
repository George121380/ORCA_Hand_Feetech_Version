"""A training-only squared-hinge penalty for unbounded Gaussian action means."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from .checkpoint import load_config
from .training_supervision import checkpoint_steps


class ActionMeanPenaltyPolicy(ActorCriticPolicy):
    def __init__(self, *args, mean_penalty_coef=.01, **kwargs):
        if not np.isfinite(mean_penalty_coef) or not 0 < mean_penalty_coef <= 1:
            raise ValueError("Mean penalty coefficient must be in (0,1]")
        self.mean_penalty_coef = float(mean_penalty_coef)
        self._regularizing_mean = False
        self.last_action_mean_stats = {}
        super().__init__(*args, **kwargs)
        if (self.use_sde or self.squash_output or type(self.action_dist) is not DiagGaussianDistribution
                or not np.all(self.action_space.low == -1) or not np.all(self.action_space.high == 1)):
            raise ValueError("Mean penalty requires the normalized unsquashed diagonal Gaussian")

    def evaluate_actions(self, obs, actions):
        self._regularizing_mean = True
        try:
            return super().evaluate_actions(obs, actions)
        finally:
            self._regularizing_mean = False

    def _get_action_dist_from_latent(self, latent_pi):
        distribution = super()._get_action_dist_from_latent(latent_pi)
        if self._regularizing_mean:
            mean = distribution.distribution.mean
            detached = mean.detach()
            excess = torch.relu(detached.abs()-1)
            self.last_action_mean_stats = dict(
                mean_penalty=float((self.mean_penalty_coef * excess.square().mean()).cpu()),
                mean_outside_fraction=float((detached.abs() > 1).float().mean().cpu()),
                max_abs_action_mean=float(detached.abs().max().cpu()))
            if mean.requires_grad:
                # Exact first-order gradient of lambda*mean(relu(abs(mu)-1)^2).
                # Inject during PPO's evaluate_actions backward, before SB3's
                # ordinary gradient clipping/Adam step. Forward probabilities,
                # PPO ratios, rollout sampling and inference remain unchanged.
                correction = 2*self.mean_penalty_coef/mean.numel() * excess * detached.sign()
                mean.register_hook(lambda gradient: gradient + correction)
        return distribution


def regularize_checkpoint(source, destination, coefficient=.01):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    _, metadata = load_config(source)
    checkpoint_steps(source, check_weights=True)
    torch.set_num_threads(1)
    old = PPO.load(source, device="cpu")
    if (old.observation_space.shape != (296,) or old.action_space.shape != (16,)
            or isinstance(old.policy, ActionMeanPenaltyPolicy) or old.use_sde or old.policy.squash_output
            or type(old.policy.action_dist) is not DiagGaussianDistribution):
        raise ValueError("Expected an unregularized diagonal Gaussian source")
    policy_kwargs = dict(old.policy_kwargs, mean_penalty_coef=coefficient)
    new = PPO.load(source, device="cpu", custom_objects={
        "policy_class": ActionMeanPenaltyPolicy, "policy_kwargs": policy_kwargs})
    old.policy.set_training_mode(False)
    new.policy.set_training_mode(False)
    obs = torch.from_numpy(np.random.default_rng(771).normal(size=(64, 296)).astype(np.float32))
    with torch.no_grad():
        a = old.policy.get_distribution(obs).distribution
        b = new.policy.get_distribution(obs).distribution
        errors = dict(action_mean=float((a.mean-b.mean).abs().max()),
                      action_std=float((a.stddev-b.stddev).abs().max()),
                      value=float((old.policy.predict_values(obs)-new.policy.predict_values(obs)).abs().max()))
    if not all(np.isfinite(error) and error <= 1e-6 for error in errors.values()):
        raise ValueError(f"Policy initialization changed: {errors}")
    proof = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 timestamp=datetime.now(timezone.utc).isoformat(), optimizer="preserved",
                 mean_penalty_coef=coefficient, verification_samples=64, max_absolute_errors=errors,
                 additional_training_steps=0)
    metadata.update(policy_class="ActionMeanPenaltyPolicy", action_mean_penalty_coef=coefficient,
                    regularization_migration=proof)
    destination.mkdir(parents=True, exist_ok=False)
    new.save(destination / "policy.zip")
    (destination / "config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return proof
