from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import FRAME_SIZE,HISTORY,yaw_rotation
from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.pen_grasp import GraspConfig,PenGraspEnv,load_grasp_config
from orca_feetech.pen_rotate import (RotationConfig,PenRotateEnv,TASK_ID_ROTATE,
    rotation_contract,load_rotation_config,final_rotation_success,validate_grasp_prefix)


def test_rotation_contract_is_separate_and_source_is_pinned(tmp_path):
    cfg=RotationConfig()
    meta=dict(task_id=TASK_ID_ROTATE,task=asdict(cfg),fingerprint=cfg.fingerprint(),
        joint_names=JOINT_NAMES,history=HISTORY,frame_size=FRAME_SIZE,upstream=provenance(),
        task_contract=rotation_contract())
    checkpoint=tmp_path/'policy.zip'
    (tmp_path/'config.json').write_text(json.dumps(meta))
    assert isinstance(load_rotation_config(checkpoint)[0],RotationConfig)
    for loader in (load_config,load_grasp_config):
        with pytest.raises(ValueError):
            loader(checkpoint)
    meta['task_contract']['prefix_sha256']='different'
    (tmp_path/'config.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='contract mismatch'):
        load_rotation_config(checkpoint)
    checkpoint.write_bytes(b'unaccepted policy')
    with pytest.raises(ValueError,match='immutable accepted'):
        validate_grasp_prefix(checkpoint)
    for kwargs in (dict(angle=10),dict(turns=2),dict(horizon=139),dict(object='cube')):
        with pytest.raises(ValueError):
            RotationConfig(**kwargs)


def test_rotation_needs_final_full_window_and_never_credits_transient_grasp():
    window=deque([(np.zeros(3),np.eye(3),True)]*20,maxlen=20)
    info=dict(dropped=False,source_grasp_completed=True)
    assert not final_rotation_success(window,info,None,at_horizon=False)[0]
    assert final_rotation_success(window,info,None,at_horizon=True)[0]
    window.append((np.zeros(3),np.eye(3),False))
    assert not final_rotation_success(window,info,None,at_horizon=True)[0]
    window=deque([(np.zeros(3),np.eye(3),True)]*19+[(np.zeros(3),yaw_rotation(.1),True)],maxlen=20)
    assert not final_rotation_success(window,info,None,at_horizon=True)[0]
    window=deque([(np.zeros(3),np.eye(3),True)]*20,maxlen=20)
    assert not final_rotation_success(window,info,'penetration_limit',at_horizon=True)[0]


@pytest.fixture
def zero_prefix(monkeypatch):
    # Physical integration stays real. Only the expensive external frozen policy
    # file is replaced with deterministic zero actions for the continuity test.
    from stable_baselines3 import PPO
    class Zero:
        def predict(self,obs,deterministic=True):
            return np.zeros(16),None
    monkeypatch.setattr(PPO,'load',lambda *a,**k:Zero())
    monkeypatch.setattr('orca_feetech.pen_rotate.validate_grasp_prefix',
                        lambda path:(GraspConfig(randomization=0),{}))
    return PenRotateEnv(RotationConfig(randomization=0),prefix_checkpoint='test-only')


def test_real_prefix_handoff_keeps_physics_controller_sensor_history(zero_prefix,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_grasp_rotation_readiness import physical_control_signature
    env=zero_prefix
    source=PenGraspEnv(GraspConfig(randomization=0))
    try:
        obs,info=source.reset(seed=2020000,options={'placement_index':5})
        for _ in range(140):
            obs,_,done,_,info=source.step(np.zeros(16))
        assert done and info['is_success']
        signature=physical_control_signature(source)
        history=np.array(source.observer.frames)
        obs,initial=env.reset(seed=2020000,options={'placement_index':5,'direction':-1})
        assert initial['prefix_grasp_success'] and not initial['is_success']
        assert initial['completed_turns']==0 and env.steps==0 and env.carrier.steps==140
        assert env.carrier.data.time==pytest.approx(7.)
        assert physical_control_signature(env.carrier)==signature
        np.testing.assert_array_equal(np.array(env.carrier.observer.frames)[:3],history[1:])
        np.testing.assert_allclose(env.goal,yaw_rotation(np.deg2rad(-15))@source.measured_rot)
        assert env.observation_space.contains(obs)
        original_origin=env.carrier.origin.copy()
        # Source grasp completion cannot leak into rotation success or early done.
        for tick in range(140):
            obs,reward,done,truncated,info=env.step(np.zeros(16))
            assert np.isfinite(reward) and env.observation_space.contains(obs)
            assert info['completed_turns']==0
            if tick<139:
                assert not done and not truncated and not info['is_success']
        assert done and env.carrier.steps==280 and env.carrier.data.time==pytest.approx(14.)
        np.testing.assert_array_equal(env.carrier.origin,original_origin)
        with pytest.raises(RuntimeError,match='Reset after'):
            env.step(np.zeros(16))
    finally:
        source.close()
        env.close()


def test_failed_prefix_is_counted_without_retry_or_integration(zero_prefix,monkeypatch):
    env=zero_prefix
    calls=[]
    def failed(action):
        calls.append(1)
        info=dict(env.carrier.info(),is_success=False,termination_reason='forced_test_failure')
        return env.carrier._observation(),-20.,True,False,info
    monkeypatch.setattr(env.carrier,'step',failed)
    try:
        _,info=env.reset(seed=2020000,options={'placement_index':5})
        assert not info['prefix_grasp_success'] and len(calls)==1
        before=env.carrier.data.qpos.copy()
        time=env.carrier.data.time
        _,reward,done,_,info=env.step(np.zeros(16))
        assert done and reward==-20 and not info['is_success']
        assert info['termination_reason']=='grasp_prefix_failed'
        np.testing.assert_array_equal(before,env.carrier.data.qpos)
        assert env.carrier.data.time==time and len(calls)==1
    finally:
        env.close()


def test_rotation_evaluation_serializes_prefix_and_goal_results(zero_prefix,monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import manipulation as cli
    args=SimpleNamespace(checkpoint=tmp_path/'policy.zip',seed=2020000,episodes=1,
                         output=tmp_path/'evaluation',baseline='hold',video=False)
    cli.evaluate(args,env_class=lambda config:zero_prefix,
        config_loader=lambda checkpoint:(zero_prefix.config,dict(task_id=TASK_ID_ROTATE)),
        reset_options=lambda episode:dict(placement_index=5,direction=-1))
    metrics=json.loads((args.output/'metrics.json').read_text())
    trial=metrics['trials'][0]
    assert trial['direction']==-1 and trial['prefix_steps']==140 and trial['prefix_grasp_success']
    assert trial['steps']==140 and trial['rotation_steps']==140 and trial['completed_turns']==0
    assert not metrics['acceptance_met']
