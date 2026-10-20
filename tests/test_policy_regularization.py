import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from orca_feetech.policy_regularization import ActionMeanPenaltyPolicy


def test_penalty_backward_matches_explicit_loss_without_changing_forward():
    space = gym.spaces.Box(-np.inf, np.inf, (296,), np.float32)
    action = gym.spaces.Box(-1, 1, (16,), np.float32)
    kwargs = dict(net_arch={"pi": [16], "vf": [16]})
    regularized = ActionMeanPenaltyPolicy(space, action, lambda _: .0003, **kwargs)
    reference = ActorCriticPolicy(space, action, lambda _: .0003, **kwargs)
    with torch.no_grad():
        regularized.action_net.bias[:8].fill_(3.)
        regularized.action_net.bias[8:].fill_(-2.)
    reference.load_state_dict(regularized.state_dict())
    obs = torch.randn(6, 296)
    actions = torch.randn(6, 16)
    a = regularized.evaluate_actions(obs, actions)
    b = reference.evaluate_actions(obs, actions)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    def base(x):
        return -x[1].mean() + .5*x[0].square().mean() - .001*x[2].mean()
    mean = reference.get_distribution(obs).distribution.mean
    explicit = base(b) + .01*torch.relu(mean.abs()-1).square().mean()
    base(a).backward()
    explicit.backward()
    for (_, x), (_, y) in zip(regularized.named_parameters(), reference.named_parameters()):
        assert torch.allclose(x.grad, y.grad, atol=1e-6, rtol=1e-5)
    assert regularized.last_action_mean_stats["mean_penalty"] > 0
    # Ordinary distribution/inference use must not acquire a training penalty.
    regularized.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    regularized.get_distribution(obs).distribution.mean.sum().backward()
    reference.get_distribution(obs).distribution.mean.sum().backward()
    for (_, x), (_, y) in zip(regularized.named_parameters(), reference.named_parameters()):
        if x.grad is not None:
            assert torch.equal(x.grad, y.grad)


class ZeroRewardEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (296,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (16,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(296, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(296, np.float32), 0., False, self.steps >= 8, {}


@pytest.mark.skipif(sys.platform != "linux", reason="Policy optimizer smoke belongs on Linux")
def test_real_ppo_update_applies_penalty_even_with_zero_advantage():
    torch.set_num_threads(1)
    model = PPO(ActionMeanPenaltyPolicy, ZeroRewardEnv(), device="cpu", seed=3,
                n_steps=16, batch_size=16, n_epochs=2, ent_coef=0, learning_rate=.001,
                policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    with torch.no_grad():
        model.policy.action_net.bias.fill_(3.)
    before = model.policy.log_std.detach().clone()
    model.learn(total_timesteps=32)
    assert model.policy.action_net.bias.mean() < 3
    assert torch.equal(before, model.policy.log_std)
    assert model.policy.last_action_mean_stats["mean_penalty"] > 0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
