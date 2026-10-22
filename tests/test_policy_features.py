from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from orca_feetech.policy_features import RelativeGoalExtractor


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
