from pathlib import Path
import json

import gymnasium as gym
import numpy as np
import pytest


def test_failed_dataset_gate_prevents_any_model_initialization(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from imitate_rotation_demonstrations import load_training_pairs
    report=tmp_path/'probe.json'
    audit=tmp_path/'audit.json'
    report.write_text(json.dumps(dict(status='failed')))
    audit.write_text(json.dumps(dict(status='complete')))
    with pytest.raises(ValueError,match='frozen admitted training corpus'):
        load_training_pairs(report,audit)


def test_actor_fit_preserves_critic_exploration_optimizer_and_rotation_counters(monkeypatch):
    import torch
    from stable_baselines3 import PPO
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from imitate_rotation_demonstrations import fit_actor
    torch.set_num_threads(1)
    class NoPhysics(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,(296,),np.float32)
        action_space=gym.spaces.Box(-1,1,(16,),np.float32)
        def reset(self,**kwargs):
            raise AssertionError('Offline fitting must never reset or simulate')
        def step(self,action):
            raise AssertionError('Offline fitting must never simulate')
    model=PPO('MlpPolicy',NoPhysics(),n_steps=2,batch_size=2,device='cpu',seed=0,
              policy_kwargs=dict(net_arch=dict(pi=[8,8],vf=[8,8])))
    before={n:p.detach().clone() for n,p in model.policy.named_parameters()}
    obs=np.zeros((2,296),np.float32)
    obs[:,0]=[-1,1]
    groups={(3111000,0,-1):(obs[:1],np.full((1,16),-.05,np.float32)),
            (3111007,0,1):(obs[1:],np.full((1,16),.05,np.float32))}
    proof=fit_actor(model,groups)
    assert sum(proof['loss_after'])<sum(proof['loss_before'])/10
    assert proof['additional_environment_steps']==0 and model.num_timesteps==0 and model._n_updates==0
    assert not model.policy.optimizer.state
    for name,value in model.policy.named_parameters():
        if not name.startswith(('mlp_extractor.policy_net.','action_net.')):
            assert torch.equal(value,before[name])
        assert value.requires_grad


def test_paired_development_gate_requires_both_directions_and_gain(monkeypatch):
    from copy import deepcopy
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from run_rotation_imitation_pilot import development_comparison
    source=dict(trials=[dict(seed=3211000+i,placement_index=i%7,direction=-1 if (i//7)%2==0 else 1,
        initial_position=[i,0,0],initial_rotation=[1,0,0],goal_rotation=[1,0,0],prefix_grasp_success=True,
        prefix_steps=140,prefix_end_reason='success',is_success=False,dropped=False) for i in range(28)])
    candidate=deepcopy(source)
    for i in (0,1,7,8):
        candidate['trials'][i]['is_success']=True
    assert development_comparison(source,candidate)['bounded_ppo_admitted']
    candidate['trials'][7]['direction']=-1
    with pytest.raises(ValueError,match='matched original physical prefixes'):
        development_comparison(source,candidate)
    candidate=deepcopy(source)
    for i in range(4):
        candidate['trials'][i]['is_success']=True
    assert not development_comparison(source,candidate)['bounded_ppo_admitted']
