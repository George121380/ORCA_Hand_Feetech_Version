from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from orca_feetech.pen_grasp import GraspConfig,PenGraspEnv,TASK_ID_GRASP,load_grasp_config,grasp_contract


class ActionRewardEnv(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(296,),np.float32)
    action_space=gym.spaces.Box(-1.,1.,(16,),np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return np.ones(296,np.float32),{}

    def step(self,action):
        return np.ones(296,np.float32),float(action[0]),False,False,{}


def test_zero_head_preserves_critic_and_exploration_but_stays_trainable_and_reloadable(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from manipulation import fresh_ppo
    torch.set_num_threads(1)
    args=SimpleNamespace(seed=0,device='cpu',num_envs=1,learning_rate=1e-4,entropy_coef=.001,
        target_kl=.02,initial_log_std=-4,output=tmp_path,zero_initial_action_mean=False)
    default=fresh_ppo(ActionRewardEnv(),args)
    args.zero_initial_action_mean=True
    zero=fresh_ppo(ActionRewardEnv(),args)
    for k,v in default.policy.state_dict().items():
        if not k.startswith('action_net.'):
            assert torch.equal(v,zero.policy.state_dict()[k])
    obs=np.random.default_rng(0).normal(size=(20,296)).astype(np.float32)
    assert np.any(default.predict(obs,deterministic=True)[0]!=0)
    assert np.array_equal(zero.predict(obs,deterministic=True)[0],np.zeros((20,16)))
    zero.learn(32)
    learned=zero.predict(obs,deterministic=True)[0]
    assert np.any(learned!=0)
    zero.save(tmp_path/'learned.zip')
    loaded=PPO.load(tmp_path/'learned.zip',device='cpu')
    assert np.array_equal(learned,loaded.predict(obs,deterministic=True)[0])
    default.get_env().close()
    zero.get_env().close()


def test_real_grasp_cli_snapshots_zero_initial_policy_and_resume_keeps_learned_head(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import manipulation as cli
    import orca_feetech.policy_initialization as init
    kwargs=dict(config_class=GraspConfig,env_class=PenGraspEnv,task_id=TASK_ID_GRASP,
                config_loader=load_grasp_config,task_contract=grasp_contract())
    base=['cli','train','--object','pen','--angle','10','--horizon','140','--randomization','.3',
        '--initial-log-std','-4','--zero-initial-action-mean','--num-envs','1','--device','cpu',
        '--steps','32','--eval-every','1000000','--checkpoint-every','1000000']
    out=tmp_path/'fresh'
    monkeypatch.setattr(sys,'argv',base+['--output',str(out)])
    cli.main(train_fn=partial(cli.train,**kwargs))
    initial=PPO.load(out/'initial_policy.zip',device='cpu')
    assert torch.count_nonzero(initial.policy.action_net.weight)==0
    trained=PPO.load(out/'policy.zip',device='cpu')
    expected={k:v.clone() for k,v in trained.policy.state_dict().items()}
    assert torch.count_nonzero(expected['action_net.weight'])>0
    def refuse_reset(policy):
        raise AssertionError('Resume must never reinitialize the actor')
    monkeypatch.setattr(init,'zero_actor_mean',refuse_reset)
    learn=PPO.learn
    def check_before_learning(self,*a,**kw):
        assert all(torch.equal(v,self.policy.state_dict()[k]) for k,v in expected.items())
        return learn(self,*a,**kw)
    monkeypatch.setattr(PPO,'learn',check_before_learning)
    resumed=tmp_path/'resumed'
    monkeypatch.setattr(sys,'argv',base+['--output',str(resumed),'--resume',str(out/'policy.zip')])
    cli.main(train_fn=partial(cli.train,**kwargs))
    assert not (resumed/'initial_policy.zip').exists()
    assert load_grasp_config(resumed/'policy.zip')[1]['zero_initial_action_mean']
