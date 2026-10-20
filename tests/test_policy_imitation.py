import copy
import json
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from orca_feetech.policy_imitation import (censored_action_loss, digest, imitate_actor,
                                          load_demonstrations)


def test_censored_loss_preserves_valid_saturation_and_corrects_saturated_interior_error():
    mean = torch.tensor([[4., -4., 3., -3., 0., 0.]], requires_grad=True)
    target = torch.tensor([[1., -1., .5, -.5, 1., -1.]])
    loss = censored_action_loss(mean, target)
    loss.backward()
    np.testing.assert_allclose(mean.grad, [[0., 0., 5/6, -5/6, -2/6, 2/6]])
    assert loss.item() == pytest.approx((2*2.5**2+2)/6)


def demonstration_fixture(tmp_path):
    source = tmp_path/"policy.zip"
    source.write_bytes(b"hash-only source fixture, never loaded as a model")
    (tmp_path/"config.json").write_text("{}")
    folder = tmp_path/"data"
    folder.mkdir()
    example = dict(seed=0, success=True, dropped=False, end_reason="success", steps=10,
                   residual=[0.]*16, trajectory="trajectory.npz")
    np.savez(folder/"trajectory.npz", observations=np.ones((10,296),np.float32),
             actions=np.zeros((10,16),np.float32), seed=np.array(0), residual=np.zeros(16))
    report = dict(status="complete", is_formal_evaluation=False, checkpoint_sha256=digest(source),
                  config_sha256=digest(tmp_path/"config.json"), fingerprint="contract", seed=0, episodes=1,
                  trials=[dict(seed=0, baseline=example, candidates=[copy.deepcopy(example)])])
    path = folder/"probe.json"
    path.write_text(json.dumps(report))
    return source, path, report


def test_dataset_requires_source_hash_training_seeds_both_groups_and_bounded_observable_labels(tmp_path):
    source, report_path, report = demonstration_fixture(tmp_path)
    groups, proof = load_demonstrations([report_path], source, "contract")
    assert set(groups) == {"baseline", "rescue"} and groups["rescue"][0].shape == (10,296)
    assert len(proof[0]["trajectories"]) == 2
    with pytest.raises(ValueError, match="disjoint"):
        load_demonstrations([report_path, report_path], source, "contract")
    report["seed"] = report["trials"][0]["seed"] = 820000
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="training seed"):
        load_demonstrations([report_path], source, "contract")
    report["seed"] = report["trials"][0]["seed"] = 0
    report_path.write_text(json.dumps(report))
    path = report_path.parent/"trajectory.npz"
    np.savez(path, observations=np.ones((10,296)), actions=np.full((10,16),1.01),
             seed=np.array(0), residual=np.zeros(16))
    with pytest.raises(ValueError, match="Malformed"):
        load_demonstrations([report_path], source, "contract")
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source protocol"):
        load_demonstrations([report_path], source, "contract")


class SmallEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf,np.inf,(296,),np.float32)
    action_space = gym.spaces.Box(-1,1,(16,),np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full(296,.1,np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.full(296,.1,np.float32), float(action.mean()), False, self.steps >= 8, {}


@pytest.mark.skipif(sys.platform != "linux", reason="Actual policy optimization belongs on Linux")
def test_offline_actor_update_preserves_critic_variance_moments_and_counters_then_resumes_ppo(tmp_path):
    torch.set_num_threads(1)
    model = PPO("MlpPolicy",SmallEnv(),device="cpu",seed=3,n_steps=16,batch_size=16,n_epochs=2,
                policy_kwargs={"net_arch":{"pi":[16],"vf":[16]}})
    model.num_timesteps = 1234
    parameters = dict(model.policy.named_parameters())
    for p in parameters.values():
        model.policy.optimizer.state[p] = dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,.001),
                                               exp_avg_sq=torch.full_like(p,.002))
    before = {n:p.detach().clone() for n,p in parameters.items()}
    old_state = {n:copy.deepcopy(model.policy.optimizer.state[p]) for n,p in parameters.items()}
    obs = np.random.default_rng(4).normal(size=(64,296)).astype(np.float32)
    actions = model.predict(obs,deterministic=True)[0]
    groups = dict(baseline=(obs,actions),rescue=(obs,np.clip(actions+.2,-1,1)))
    proof = imitate_actor(model,groups,updates=64,batch_size=32)
    assert proof["loss_after"]["rescue"] < proof["loss_before"]["rescue"]
    assert model.num_timesteps == 1234 and model._n_updates == 0
    assert proof["additional_environment_steps"] == 0
    for name,p in parameters.items():
        if name.startswith(("action_net.","mlp_extractor.policy_net.")):
            assert p not in model.policy.optimizer.state
        else:
            assert torch.equal(p,before[name])
            for key,value in old_state[name].items():
                assert torch.equal(model.policy.optimizer.state[p][key],value)
    assert any(not torch.equal(p,before[n]) for n,p in parameters.items())
    model.save(tmp_path/"policy.zip")
    loaded = PPO.load(tmp_path/"policy.zip",env=SmallEnv(),device="cpu")
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],
                                  loaded.predict(obs,deterministic=True)[0])
    loaded.learn(total_timesteps=32,reset_num_timesteps=False)
    assert loaded.num_timesteps == 1266
    assert all(torch.isfinite(p).all() for p in loaded.policy.parameters())
