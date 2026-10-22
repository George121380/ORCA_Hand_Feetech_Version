from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

import gymnasium as gym
import numpy as np
import pytest
import torch

from orca_feetech.manipulation import yaw_rotation
from orca_feetech.policy_features import GoalErrorFeatures,RelativeGoalExtractor


def test_goal_features_use_observed_pose_and_are_invariant_to_common_body_rotation():
    extractor = GoalErrorFeatures(gym.spaces.Box(-np.inf, np.inf, (296,), np.float32))
    rng = np.random.default_rng(29)
    inputs = rng.normal(size=(2, 4, 74)).astype(np.float32)
    for sample in range(2):
        for frame in range(4):
            R = yaw_rotation(.2*frame + sample)
            G = yaw_rotation(.6) @ R
            inputs[sample, frame, 38:47] = R.ravel()
            inputs[sample, frame, 47:56] = G.ravel()
    transformed = inputs.copy()
    S = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    for start in (38, 47):
        transformed[..., start:start+9] = (inputs[..., start:start+9].reshape(2, 4, 3, 3) @ S).reshape(2, 4, 9)
    a = extractor(torch.from_numpy(inputs.reshape(2, 296))).reshape(2, 4, 83)
    b = extractor(torch.from_numpy(transformed.reshape(2, 296))).reshape(2, 4, 83)
    assert torch.equal(a[..., :74], torch.from_numpy(inputs))
    assert torch.allclose(a[..., 74:], b[..., 74:], atol=1e-6)
    assert np.allclose(a[0, 0, 74:].numpy().reshape(3, 3), yaw_rotation(.6), atol=1e-6)
    changed = inputs.copy()
    changed[0, 0, 38:47] = yaw_rotation(.3).ravel()
    c = extractor(torch.from_numpy(changed.reshape(2, 296))).reshape(2, 4, 83)
    assert not torch.allclose(a[0, 0, 74:], c[0, 0, 74:])
    # Each history frame must use its own held/noisy camera pose, without any
    # simulator state or an extra source of fresh observations.
    assert torch.equal(a[0, 1:], c[0, 1:])
    with pytest.raises(ValueError, match="v7"):
        GoalErrorFeatures(gym.spaces.Box(-np.inf, np.inf, (232,), np.float32))


def test_relative_error_has_correct_world_frame_and_leaves_other_signals_unchanged():
    raw=np.random.default_rng(21).normal(size=(3,4,74)).astype(np.float32)
    r=Rotation.random(12,random_state=17).as_matrix().reshape(3,4,3,3)
    desired=Rotation.from_euler('z',15,degrees=True).as_matrix()
    raw[:,:,38:47]=r.reshape(3,4,9)
    raw[:,:,47:56]=(desired@r).reshape(3,4,9)
    original=torch.from_numpy(raw.reshape(3,296).copy())
    extractor=RelativeGoalExtractor(gym.spaces.Box(-np.inf,np.inf,(296,),np.float32))
    out=extractor(original).reshape(3,4,74).detach().numpy()
    np.testing.assert_allclose(out[:,:,47:56],np.broadcast_to(desired.reshape(9),(3,4,9)),atol=2e-7)
    np.testing.assert_array_equal(out[:,:,:47],raw[:,:,:47])
    np.testing.assert_array_equal(out[:,:,56:],raw[:,:,56:])
    np.testing.assert_array_equal(original.numpy(),raw.reshape(3,296))
    np.testing.assert_allclose((out[:,:,47:56].reshape(3,4,3,3)@r).reshape(3,4,9),raw[:,:,47:56],atol=2e-7)


def test_fresh_pure_ppo_has_zero_mean_no_optimizer_or_bc_counters_and_reloads(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from manipulation import fresh_ppo
    from stable_baselines3 import PPO
    class NoPhysics(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,(296,),np.float32)
        action_space=gym.spaces.Box(-1,1,(16,),np.float32)
        def reset(self,**kwargs):
            raise AssertionError('Initialization may not simulate')
        def step(self,action):
            raise AssertionError('Initialization may not simulate')
    args=SimpleNamespace(seed=0,device='cpu',num_envs=1,learning_rate=1e-5,entropy_coef=.001,
        target_kl=.02,initial_log_std=-3.,output=tmp_path,relative_goal_features=True,
        zero_initial_action_mean=True)
    torch.set_num_threads(1)
    model=fresh_ppo(NoPhysics(),args)
    obs=np.random.default_rng(9).normal(size=(64,296)).astype(np.float32)
    assert np.array_equal(model.predict(obs,deterministic=True)[0],np.zeros((64,16),np.float32))
    assert not model.policy.optimizer.state and model.num_timesteps==0 and model._n_updates==0
    assert type(model.policy.features_extractor) is RelativeGoalExtractor
    model.save(tmp_path/'initial.zip')
    loaded=PPO.load(tmp_path/'initial.zip',device='cpu')
    np.testing.assert_array_equal(loaded.predict(obs,deterministic=True)[0],model.predict(obs,deterministic=True)[0])
