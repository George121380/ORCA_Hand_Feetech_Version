from dataclasses import asdict
import json
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import TaskConfig, TASK_ID, FRAME_SIZE, HISTORY
from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.policy_exploration import migrate_sde_checkpoint


class ObservationEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (296,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (16,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full(296, .1, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.full(296, .1+.01*self.steps, np.float32), float(self.steps % 3), False, self.steps >= 8, {}


def source_checkpoint(path):
    path.mkdir()
    model = PPO("MlpPolicy", ObservationEnv(), device="cpu", seed=3, n_steps=16, batch_size=16,
                n_epochs=2, policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    model.num_timesteps = 1234
    weight = dict(model.policy.named_parameters())["mlp_extractor.policy_net.0.weight"]
    for parameter in (weight, model.policy.log_std):
        model.policy.optimizer.state[parameter] = dict(step=torch.tensor(7.),
            exp_avg=torch.full_like(parameter, .001), exp_avg_sq=torch.full_like(parameter, .002))
    model.save(path/"policy.zip")
    cfg = TaskConfig(object="pen", angle=35)
    metadata = dict(task_id=TASK_ID, task=asdict(cfg), fingerprint=cfg.fingerprint(),
                    joint_names=JOINT_NAMES, history=HISTORY, frame_size=FRAME_SIZE,
                    upstream=provenance(), trained=True, actual_steps=1234)
    (path/"config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return model


def test_sde_migration_preserves_deterministic_behavior_moments_and_calibrates_variance(tmp_path):
    old = source_checkpoint(tmp_path/"old")
    obs = np.random.default_rng(71).normal(size=(64, 296)).astype(np.float32)
    proof = migrate_sde_checkpoint(tmp_path/"old/policy.zip", tmp_path/"new", obs)
    cfg, meta = load_config(tmp_path/"new/policy.zip")
    new = PPO.load(tmp_path/"new/policy.zip", device="cpu")
    assert new.use_sde and new.sde_sample_freq == 4 and new.num_timesteps == 1234
    assert new.policy.log_std.shape == (16, 16)
    assert cfg.fingerprint() == meta["fingerprint"]
    assert max(proof["max_absolute_errors"].values()) == 0
    assert proof["mean_variance_relative_error"] < 1e-5
    for name, value in old.policy.state_dict().items():
        if name != "log_std":
            assert torch.equal(value, new.policy.state_dict()[name])
    assert np.array_equal(old.predict(obs, deterministic=True)[0], new.predict(obs, deterministic=True)[0])
    old_weight = dict(old.policy.named_parameters())["mlp_extractor.policy_net.0.weight"]
    new_weight = dict(new.policy.named_parameters())["mlp_extractor.policy_net.0.weight"]
    for key, value in old.policy.optimizer.state[old_weight].items():
        assert torch.equal(value, new.policy.optimizer.state[new_weight][key])
    assert new.policy.log_std not in new.policy.optimizer.state
    new.policy.reset_noise(1)
    first = new.predict(obs[0], deterministic=False)[0]
    assert np.array_equal(first, new.predict(obs[0], deterministic=False)[0])
    new.policy.reset_noise(1)
    assert not np.array_equal(first, new.predict(obs[0], deterministic=False)[0])
    with pytest.raises(ValueError, match="unregularized Gaussian"):
        migrate_sde_checkpoint(tmp_path/"new/policy.zip", tmp_path/"again", obs)
    assert not (tmp_path/"again").exists()


def test_degenerate_calibration_is_rejected_before_writing(tmp_path):
    source_checkpoint(tmp_path/"old")
    with pytest.raises(ValueError, match="Degenerate"):
        migrate_sde_checkpoint(tmp_path/"old/policy.zip", tmp_path/"invalid", np.zeros((64, 296)))
    assert not (tmp_path/"invalid").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Actual policy optimization belongs on Linux")
def test_sde_checkpoint_runs_real_ppo_with_periodic_noise_resampling(tmp_path, monkeypatch):
    source_checkpoint(tmp_path/"old")
    obs = np.random.default_rng(71).normal(size=(64, 296)).astype(np.float32)
    migrate_sde_checkpoint(tmp_path/"old/policy.zip", tmp_path/"new", obs)
    model = PPO.load(tmp_path/"new/policy.zip", env=ObservationEnv(), device="cpu")
    calls = []
    original = model.policy.reset_noise
    def reset(n_envs=1):
        calls.append(n_envs)
        return original(n_envs)
    monkeypatch.setattr(model.policy, "reset_noise", reset)
    model.learn(total_timesteps=32, reset_num_timesteps=False)
    assert model.num_timesteps == 1266 and calls.count(1) >= 8
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
