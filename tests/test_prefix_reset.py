from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv,TaskConfig,TASK_ID,HISTORY,FRAME_SIZE
from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.prefix_reset import PhysicalPrefixReset,PREFIX_INFO,make_training_environment


class Teacher:
    calls = 0
    def predict(self,obs,deterministic):
        assert deterministic and obs.shape == (296,)
        self.calls += 1
        return np.zeros(16,np.float32),None


class PrefixFixture(gym.Env):
    observation_space = gym.spaces.Box(-np.inf,np.inf,(296,),np.float32)
    action_space = gym.spaces.Box(-1,1,(16,),np.float32)
    def __init__(self,fail=False):
        self.fail,self.resets = fail,0
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.steps = self.completed = 0
        self.position,self.velocity,self.goal = .1,.02,1
        self.resets += 1
        return self.obs(),self.info()
    def obs(self):
        return np.full(296,self.position+self.goal,np.float32)
    def info(self):
        return dict(completed_turns=self.completed,is_success=self.completed==2,dropped=False,
                    angle_error_deg=30.,goal_angle_deg=30.)
    def step(self,action):
        self.steps += 1
        self.position += self.velocity*.05
        old = self.completed
        if not self.fail:
            self.completed = int(self.steps>=10)+int(self.steps>=20)
            self.goal = 1+self.completed
        info = self.info()
        if self.steps == 10 and not self.fail:
            info["angle_error_deg"] = 0.  # describes old goal, as canonical step does
        return self.obs(),10.*(self.completed-old),self.completed==2 or self.fail and self.steps==3,False,info


def test_physical_prefix_preserves_state_and_remaining_clock_and_only_counts_learner_rewards():
    from stable_baselines3.common.monitor import Monitor
    reference = PrefixFixture()
    reference.reset(seed=7)
    for _ in range(10):
        expected = reference.step(np.zeros(16))[0]
    inner = PrefixFixture()
    wrapped = PhysicalPrefixReset(inner,Teacher(),probability=1)
    monitor = Monitor(wrapped,info_keywords=PREFIX_INFO)
    obs,info = monitor.reset(seed=7)
    np.testing.assert_array_equal(obs,expected)
    assert (inner.position,inner.velocity,inner.steps,inner.goal)==(
        reference.position,reference.velocity,reference.steps,reference.goal)
    assert inner.resets == 1 and info["completed_turns"] == 1 and not info["is_success"]
    assert info["angle_error_deg"] == 30 and info["training_prefix_steps"] == 10
    assert info["training_prefix_attempts"] == 1 and info["training_prefix_rollout_steps"] == 10
    for _ in range(10):
        result = monitor.step(np.ones(16))
        expected = reference.step(np.ones(16))
        np.testing.assert_array_equal(result[0],expected[0])
        assert result[1:4] == expected[1:4]
    assert result[4]["is_success"] and result[4]["episode"]["r"] == 10
    assert result[4]["episode"]["l"] == 10 and inner.steps == 20
    assert result[4]["episode"]["training_prefix_used"] == 1


def test_failed_prefix_is_bounded_and_falls_back_without_fabricating_success():
    inner,teacher = PrefixFixture(fail=True),Teacher()
    env = PhysicalPrefixReset(inner,teacher,probability=1)
    _,info = env.reset(seed=5)
    assert inner.resets == 6 and teacher.calls == 15
    assert inner.steps == 0 and info["completed_turns"] == 0 and not info["is_success"]
    assert info["training_prefix_used"] == 0 and info["training_prefix_steps"] == 0
    assert info["training_prefix_attempts"] == 5 and info["training_prefix_rollout_steps"] == 15


def test_probability_zero_preserves_real_mujoco_observations_physics_and_rng():
    cfg = TaskConfig(object="pen",angle=30,turns=2,horizon=480)
    reference,inner = ManipulationEnv(cfg),ManipulationEnv(cfg)
    teacher = Teacher()
    wrapped = PhysicalPrefixReset(inner,teacher,probability=0)
    try:
        a,_ = reference.reset(seed=71)
        b,info = wrapped.reset(seed=71)
        np.testing.assert_array_equal(a,b)
        for _ in range(20):
            a,ra,da,ta,ia = reference.step(np.full(16,.1))
            b,rb,db,tb,ib = wrapped.step(np.full(16,.1))
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(reference.data.qpos,inner.data.qpos)
            np.testing.assert_array_equal(reference.data.qvel,inner.data.qvel)
            assert (ra,da,ta)==(rb,db,tb)
        assert teacher.calls == 0 and info["training_prefix_used"] == 0
    finally:
        reference.close()
        wrapped.close()


def test_evaluation_cannot_load_prefix_or_use_training_goal_distribution(monkeypatch):
    import orca_feetech.prefix_reset as module
    monkeypatch.setattr(module,"load_frozen_teacher",lambda *a:pytest.fail("Evaluation loaded a teacher"))
    env = make_training_environment(TaskConfig(object="pen",angle=30,turns=2,horizon=480),
        evaluation=True,negative_goal_probability=1,minimum_goal_angle=15,prefix_checkpoint=Path("absent.zip"))
    try:
        assert type(env) is ManipulationEnv and env.negative_goal_probability == .5
        assert env.minimum_goal_angle is None
        _,info = env.reset(seed=8)
        assert info["completed_turns"] == 0 and not any(k in info for k in PREFIX_INFO)
    finally:
        env.close()


@pytest.mark.skipif(sys.platform != "linux",reason="Actual policy optimization belongs on Linux")
def test_real_cli_prefix_training_saves_contract_and_runs_ppo(tmp_path):
    import torch
    from stable_baselines3 import PPO
    from test_policy_exploration import ObservationEnv
    from orca_feetech.prefix_reset import validate_prefix_checkpoint
    root = Path(__file__).parents[1]
    source = tmp_path/"source"
    source.mkdir()
    model = PPO("MlpPolicy",ObservationEnv(),device="cpu",seed=3,n_steps=16,batch_size=16,n_epochs=2,
                policy_kwargs={"net_arch":{"pi":[16],"vf":[16]}})
    model.num_timesteps = 1234
    model.save(source/"policy.zip")
    cfg = TaskConfig(object="pen",angle=30)
    meta = dict(task_id=TASK_ID,task=asdict(cfg),fingerprint=cfg.fingerprint(),joint_names=JOINT_NAMES,
                history=HISTORY,frame_size=FRAME_SIZE,upstream=provenance(),trained=True)
    (source/"config.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="matched"):
        validate_prefix_checkpoint(source/"policy.zip",TaskConfig(object="pen",angle=35,turns=2,horizon=480))
    output = tmp_path/"output"
    command = [sys.executable,str(root/"scripts/manipulation.py"),"train","--object","pen","--angle","30",
        "--turns","2","--horizon","480","--randomization","1","--num-envs","1","--device","cpu",
        "--steps","32","--checkpoint-every","32","--eval-every","1000000","--resume",str(source/"policy.zip"),
        "--prefix-reset-checkpoint",str(source/"policy.zip"),"--prefix-reset-probability","1","--output",str(output)]
    result = subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    info = json.loads((output/"config.json").read_text())
    assert info["training_prefix_reset"]["probability"] == 1
    assert info["training_prefix_reset"]["evaluation_assistance"] is False
    assert info["actual_steps"] == 1266 and info["task"]["turns"] == 2
    loaded = PPO.load(output/"policy.zip",device="cpu")
    assert loaded.num_timesteps == 1266 and loaded.observation_space.shape == (296,)
    assert all(torch.isfinite(p).all() for p in loaded.policy.parameters())
