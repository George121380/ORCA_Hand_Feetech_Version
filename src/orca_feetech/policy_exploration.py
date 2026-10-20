"""Explicit Gaussian-to-gSDE migration with deterministic-policy parity."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .checkpoint import load_config
from .training_supervision import checkpoint_steps


def migrate_sde_checkpoint(source, destination, calibration, *, sample_frequency=4):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if type(sample_frequency) is not int or not 1 <= sample_frequency <= 20:
        raise ValueError("gSDE sample frequency must be 1..20 control steps")
    observations = np.asarray(calibration, dtype=np.float32)
    if (observations.ndim != 2 or observations.shape[1] != 296
            or not 16 <= len(observations) <= 4096 or not np.isfinite(observations).all()):
        raise ValueError("Calibration requires 16..4096 finite raw296 observations")
    _, metadata = load_config(source)
    checkpoint_steps(source, check_weights=True)
    torch.set_num_threads(1)
    old = PPO.load(source, device="cpu")
    if (type(old.policy) is not ActorCriticPolicy or old.use_sde or old.policy.squash_output
            or type(old.policy.action_dist) is not DiagGaussianDistribution
            or type(old.policy.features_extractor) is not FlattenExtractor
            or not old.policy.share_features_extractor
            or not np.all(old.action_space.low == -1) or not np.all(old.action_space.high == 1)
            or old.observation_space.shape != (296,) or old.action_space.shape != (16,)):
        raise ValueError("Expected a raw296 unregularized Gaussian MLP source")

    class SpacesOnlyEnv(gym.Env):
        observation_space, action_space = old.observation_space, old.action_space

        def reset(self, *, seed=None, options=None):
            raise AssertionError("Migration must never train or step this space-only environment")

        def step(self, action):
            raise AssertionError("Migration must never train or step this space-only environment")

    env = DummyVecEnv([SpacesOnlyEnv for _ in range(old.n_envs)])
    try:
        kwargs = dict(old.policy_kwargs, full_std=True, use_expln=False, squash_output=False)
        kwargs.pop("use_sde", None)
        new = PPO("MlpPolicy", env, device="cpu", seed=old.seed, use_sde=True,
                  sde_sample_freq=sample_frequency, n_steps=old.n_steps, batch_size=old.batch_size,
                  n_epochs=old.n_epochs, learning_rate=float(old.lr_schedule(1.)),
                  gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=float(old.clip_range(1.)),
                  clip_range_vf=None if old.clip_range_vf is None else float(old.clip_range_vf(1.)),
                  normalize_advantage=old.normalize_advantage, ent_coef=old.ent_coef,
                  vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm, target_kl=old.target_kl,
                  policy_kwargs=kwargs)
        weights = new.policy.state_dict()
        if weights.keys() != old.policy.state_dict().keys():
            raise ValueError("Unsupported policy state layout")
        for name, value in old.policy.state_dict().items():
            if name == "log_std":
                continue
            if weights[name].shape != value.shape:
                raise ValueError(f"Unexpected actor/critic shape change: {name}")
            weights[name] = value.clone()
        tensor = torch.from_numpy(observations)
        with torch.no_grad():
            latent = old.policy.mlp_extractor.forward_actor(old.policy.extract_features(tensor))
            norm = latent.square().sum(dim=1).mean()
            target = old.policy.log_std.exp().square()
            epsilon = new.policy.action_dist.epsilon
            if not torch.isfinite(norm) or norm <= 1e-8 or (target <= epsilon).any():
                raise ValueError("Degenerate features/variance in gSDE calibration")
            calibrated = .5*torch.log((target-epsilon)/norm)
            weights["log_std"] = calibrated.expand_as(weights["log_std"]).clone()
        new.policy.load_state_dict(weights, strict=True)
        # Only the log_std parameter changes shape. Preserve Adam history for
        # every unchanged actor/critic parameter, reset this new noise matrix.
        old_names = {p:n for n,p in old.policy.named_parameters()}
        new_names = {p:n for n,p in new.policy.named_parameters()}
        new_parameters = dict(new.policy.named_parameters())
        if len(old.policy.optimizer.param_groups) != len(new.policy.optimizer.param_groups):
            raise ValueError("Optimizer parameter groups differ")
        for before, after in zip(old.policy.optimizer.param_groups, new.policy.optimizer.param_groups):
            if [old_names[p] for p in before["params"]] != [new_names[p] for p in after["params"]]:
                raise ValueError("Optimizer parameter ordering differs")
            after.update({k:deepcopy(v) for k,v in before.items() if k != "params"})
        for parameter, state in old.policy.optimizer.state.items():
            name = old_names[parameter]
            if name != "log_std":
                new.policy.optimizer.state[new_parameters[name]] = deepcopy(state)
        new.num_timesteps, new._n_updates = old.num_timesteps, old._n_updates
        new.policy.reset_noise(new.n_envs)
        old.policy.set_training_mode(False)
        new.policy.set_training_mode(False)
        random = np.random.default_rng(771).normal(size=(64, 296)).astype(np.float32)
        inputs = torch.from_numpy(np.concatenate([observations, random]))
        with torch.no_grad():
            a = old.policy.get_distribution(inputs).distribution
            b = new.policy.get_distribution(inputs).distribution
            errors = dict(action_mean=float((a.mean-b.mean).abs().max()), value=float(
                (old.policy.predict_values(inputs)-new.policy.predict_values(inputs)).abs().max()))
            standard = new.policy.get_distribution(tensor).distribution.stddev
            relative_variance_error = float(((standard.square().mean(dim=0)-target)/target).abs().max())
        if (not all(np.isfinite(v) and v <= 1e-6 for v in errors.values())
                or not np.isfinite(relative_variance_error) or relative_variance_error > 1e-5):
            raise ValueError(f"gSDE migration verification failed: {errors}, {relative_variance_error}")
        proof = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     timestamp=datetime.now(timezone.utc).isoformat(), additional_training_steps=0,
                     calibration_observations=len(observations), verification_observations=len(inputs),
                     max_absolute_errors=errors, mean_variance_relative_error=relative_variance_error,
                     calibration_std_range=[float(standard.min()), float(standard.max())],
                     source_std=old.policy.log_std.detach().exp().tolist(),
                     noise_matrix_shape=list(new.policy.log_std.shape), sample_frequency=sample_frequency,
                     optimizer="unchanged actor/critic moments preserved; log_std moments reset",
                     preserved_optimizer_parameters=len(new.policy.optimizer.state),
                     distribution_change="State-dependent and temporally correlated; not distribution parity")
        metadata.update(policy_class="ActorCriticPolicy", policy_features="FlattenExtractor",
                        policy_distribution="StateDependentNoiseDistribution", use_sde=True,
                        sde_sample_freq=sample_frequency, sde_migration=proof)
        destination.mkdir(exist_ok=False, parents=True)
        new.save(destination/"policy.zip")
        (destination/"config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        np.savez_compressed(destination/"calibration.npz", observations=observations)
        return proof
    finally:
        env.close()
