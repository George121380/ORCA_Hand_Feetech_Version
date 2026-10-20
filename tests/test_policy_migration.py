from dataclasses import asdict
import json

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv, TaskConfig, TASK_ID
from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.policy_migration import (legacy_columns, migrate_v6_checkpoint,
                                         augment_goal_error_checkpoint)


class LegacyInputEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (232,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (16,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(232, np.float32), {}

    def step(self, action):
        raise AssertionError("Migration test must never train or step this stub")


def legacy_checkpoint(path):
    path.mkdir()
    cfg = TaskConfig(object="pen", angle=30)
    model = PPO("MlpPolicy", LegacyInputEnv(), device="cpu", seed=3, n_steps=16, batch_size=16,
                policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}})
    model.num_timesteps = 1234
    model.save(path / "policy.zip")
    meta = dict(task_id="orca-v1-printable-yaw-v6", task=asdict(cfg),
                fingerprint=cfg.fingerprint(task_id="orca-v1-printable-yaw-v6"),
                joint_names=JOINT_NAMES, history=4, frame_size=58, upstream=provenance(),
                trained=True, actual_steps=1234)
    (path / "config.json").write_text(json.dumps(meta))
    return model, meta


def test_migration_keeps_actions_values_and_timesteps_with_new_state(tmp_path):
    import torch
    old, _ = legacy_checkpoint(tmp_path / "v6")
    with pytest.raises(ValueError, match="mismatch"):
        load_config(tmp_path / "v6/policy.zip")
    proof = migrate_v6_checkpoint(tmp_path / "v6/policy.zip", tmp_path / "v7")
    assert max(proof["max_absolute_errors"].values()) < 1e-5
    assert proof["optimizer"] == "reset"
    cfg, meta = load_config(tmp_path / "v7/policy.zip")
    assert meta["task_id"] == TASK_ID and meta["frame_size"] == 74
    new = PPO.load(tmp_path / "v7/policy.zip", device="cpu")
    assert new.num_timesteps == 1234 and not new.policy.optimizer.state
    env = ManipulationEnv(cfg)
    try:
        obs, _ = env.reset(seed=33)
        # Controller state has no influence until fine tuning learns its new
        # zero-initialized input weights. Old columns must map per frame.
        obs.reshape(4, 74)[:, 58:] = np.linspace(-1, 1, 16)
        legacy = obs[legacy_columns()]
        assert np.allclose(old.predict(legacy, deterministic=True)[0],
                           new.predict(obs, deterministic=True)[0], atol=1e-6)
        with torch.no_grad():
            a = old.policy.predict_values(torch.from_numpy(legacy[None]))
            b = new.policy.predict_values(torch.from_numpy(obs[None]))
            assert torch.allclose(a, b, atol=1e-6)
    finally:
        env.close()
    with pytest.raises(FileExistsError):
        migrate_v6_checkpoint(tmp_path / "v6/policy.zip", tmp_path / "v7")


def test_migration_rejects_tampered_legacy_contract_before_writing(tmp_path):
    _, meta = legacy_checkpoint(tmp_path / "v6")
    meta["task"]["friction"] = 1.5
    (tmp_path / "v6/config.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="mismatch"):
        migrate_v6_checkpoint(tmp_path / "v6/policy.zip", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_migration_rejects_nonfinite_source_weights(tmp_path):
    import torch
    model, _ = legacy_checkpoint(tmp_path / "v6")
    with torch.no_grad():
        next(model.policy.parameters()).fill_(float("nan"))
    model.save(tmp_path / "v6/policy.zip")
    with pytest.raises(ValueError, match="Non-finite"):
        migrate_v6_checkpoint(tmp_path / "v6/policy.zip", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_internal_goal_features_preserve_external_contract_and_initial_outputs(tmp_path):
    import torch
    legacy_checkpoint(tmp_path / "v6")
    migrate_v6_checkpoint(tmp_path / "v6/policy.zip", tmp_path / "raw")
    proof = augment_goal_error_checkpoint(tmp_path / "raw/policy.zip", tmp_path / "derived")
    assert max(proof["max_absolute_errors"].values()) < 1e-5
    cfg, meta = load_config(tmp_path / "derived/policy.zip")
    assert meta["frame_size"] == 74 and meta["policy_features"] == "GoalErrorFeatures"
    raw = PPO.load(tmp_path / "raw/policy.zip", device="cpu")
    new = PPO.load(tmp_path / "derived/policy.zip", device="cpu")
    assert raw.observation_space.shape == new.observation_space.shape == (296,)
    assert new.policy.features_extractor.features_dim == 332
    assert new.num_timesteps == raw.num_timesteps and not new.policy.optimizer.state
    env = ManipulationEnv(cfg)
    try:
        obs, _ = env.reset(seed=45)
        assert np.allclose(raw.predict(obs, deterministic=True)[0],
                           new.predict(obs, deterministic=True)[0], atol=1e-6)
        tensor = torch.from_numpy(obs[None])
        assert torch.allclose(raw.policy.predict_values(tensor), new.policy.predict_values(tensor), atol=1e-6)
        new.policy.get_distribution(tensor).distribution.mean.sum().backward()
        columns = np.concatenate([i*83+np.arange(74, 83) for i in range(4)])
        grad = new.policy.mlp_extractor.policy_net[0].weight.grad[:, columns]
        assert grad.abs().sum() > 0
    finally:
        env.close()
    with pytest.raises(ValueError, match="raw flattened"):
        augment_goal_error_checkpoint(tmp_path / "derived/policy.zip", tmp_path / "twice")
    assert not (tmp_path / "twice").exists()
