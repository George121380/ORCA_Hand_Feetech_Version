"""Explicit v6 to v7 observation expansion; no implicit legacy deployment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import load_v6_config_for_migration
from .manipulation import ManipulationEnv, TASK_ID, FRAME_SIZE, HISTORY

LEGACY_FRAME_SIZE = 58


def legacy_columns():
    return np.concatenate([i * FRAME_SIZE + np.arange(LEGACY_FRAME_SIZE) for i in range(HISTORY)])


def migrate_v6_checkpoint(source, destination):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import FlattenExtractor

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    config, metadata = load_v6_config_for_migration(source)
    torch.set_num_threads(1)
    old = PPO.load(source, device="cpu")
    if (old.observation_space.shape != (LEGACY_FRAME_SIZE * HISTORY,)
            or old.action_space.shape != (16,) or old.use_sde
            or not isinstance(old.policy.features_extractor, FlattenExtractor)
            or not old.policy.share_features_extractor):
        raise ValueError("Migration supports the v6 flattened MLP Gaussian PPO contract only")
    env = ManipulationEnv(config)
    try:
        new = PPO("MlpPolicy", env, device="cpu", n_steps=old.n_steps, batch_size=old.batch_size,
                  n_epochs=old.n_epochs, learning_rate=float(old.lr_schedule(1.)),
                  gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=float(old.clip_range(1.)),
                  clip_range_vf=None if old.clip_range_vf is None else float(old.clip_range_vf(1.)),
                  normalize_advantage=old.normalize_advantage, ent_coef=old.ent_coef,
                  vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm, target_kl=old.target_kl,
                  policy_kwargs=old.policy_kwargs, seed=old.seed)
        columns = legacy_columns()
        weights = new.policy.state_dict()
        expanded = []
        for name, value in old.policy.state_dict().items():
            if name not in weights:
                raise ValueError(f"Unexpected legacy parameter: {name}")
            if value.shape != weights[name].shape:
                if (name not in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
                        or value.shape[1:] != (LEGACY_FRAME_SIZE * HISTORY,)
                        or weights[name].shape != (value.shape[0], FRAME_SIZE * HISTORY)):
                    raise ValueError(f"Unsupported observation-dependent parameter: {name}")
                weights[name].zero_()
                weights[name][:, columns] = value
                expanded.append(name)
            else:
                weights[name] = value.clone()
        if len(expanded) != 2:
            raise ValueError("Expected separate actor and critic input projections")
        new.policy.load_state_dict(weights, strict=True)
        new.num_timesteps = old.num_timesteps
        # Fresh Adam moments are intentional: its old input tensors have a
        # different shape. All actor, critic and log-std weights are preserved.
        if new.policy.optimizer.state:
            raise RuntimeError("Migration must start with a fresh optimizer")
        old.policy.set_training_mode(False)
        new.policy.set_training_mode(False)
        rng = np.random.default_rng(771)
        before = rng.normal(size=(64, LEGACY_FRAME_SIZE * HISTORY)).astype(np.float32)
        after = rng.normal(size=(64, FRAME_SIZE * HISTORY)).astype(np.float32)
        after[:, columns] = before
        with torch.no_grad():
            old_tensor, new_tensor = torch.from_numpy(before), torch.from_numpy(after)
            a = old.policy.get_distribution(old_tensor).distribution
            b = new.policy.get_distribution(new_tensor).distribution
            errors = dict(action_mean=float((a.mean-b.mean).abs().max()),
                          action_std=float((a.stddev-b.stddev).abs().max()),
                          value=float((old.policy.predict_values(old_tensor)
                                       - new.policy.predict_values(new_tensor)).abs().max()))
        if max(errors.values()) > 1e-5:
            raise ValueError(f"Expanded policy changes initial behavior: {errors}")
        new_metadata = dict(metadata)
        new_metadata.update(task_id=TASK_ID, fingerprint=config.fingerprint(), frame_size=FRAME_SIZE,
                            migration=dict(source=str(source),
                                           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                           source_task_id=metadata["task_id"],
                                           timestamp=datetime.now(timezone.utc).isoformat(),
                                           optimizer="reset", new_input_weights="zero",
                                           verification_samples=64, max_absolute_errors=errors,
                                           v7_training_steps=0))
        destination.mkdir(parents=True, exist_ok=False)
        new.save(destination / "policy.zip")
        (destination / "config.json").write_text(json.dumps(new_metadata, indent=2) + "\n")
        return new_metadata["migration"]
    finally:
        env.close()
