from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import TaskConfig,ManipulationEnv,TASK_ID,HISTORY,FRAME_SIZE,DT
from orca_feetech.manipulation_v8 import (TaskConfigV8,ManipulationV8Env,TASK_ID_V8,task_contract,
    load_v8_config,RESET_HAND_SPEED,RESET_WRIST_TOLERANCE)
from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.reference_control import BrakingActionCodec


def test_new_checkpoint_contract_cannot_mix_legacy_or_tampered_controller(tmp_path):
    checkpoint=tmp_path/'policy.zip'
    for cls,task,loader,other in ((TaskConfig,TASK_ID,load_config,load_v8_config),
                                (TaskConfigV8,TASK_ID_V8,load_v8_config,load_config)):
        cfg=cls(object='pen')
        meta=dict(task_id=task,task=asdict(cfg),fingerprint=cfg.fingerprint(),joint_names=JOINT_NAMES,
                  history=HISTORY,frame_size=FRAME_SIZE,upstream=provenance())
        if cls is TaskConfigV8:
            meta['task_contract']=task_contract()
        (tmp_path/'config.json').write_text(json.dumps(meta))
        loaded,_=loader(checkpoint)
        assert type(loaded) is cls
        with pytest.raises(ValueError):
            other(checkpoint)
        if cls is TaskConfigV8:
            meta['task_contract']['invalid_observation']='Instantaneous hold'
            (tmp_path/'config.json').write_text(json.dumps(meta))
            with pytest.raises(ValueError,match='contract mismatch'):
                loader(checkpoint)
    assert TaskConfig(object='pen').fingerprint()!=TaskConfigV8(object='pen').fingerprint()
    with pytest.raises(ValueError,match='legacy'):
        TaskConfigV8().fingerprint(task_id=TASK_ID)
    with pytest.raises(TypeError,match='TaskConfigV8'):
        ManipulationV8Env(TaskConfig())


@pytest.mark.parametrize('strength',[0.,.3,1.])
def test_real_v8_reset_waits_for_hand_and_codec_matches_observed_command_differences(strength):
    env=ManipulationV8Env(TaskConfigV8(object='pen',randomization=strength,angle=10))
    try:
        obs,info=env.reset(seed=2000001)
        assert env.observation_space.contains(obs) and not info['is_success']
        assert env.model.names.startswith(TASK_ID_V8.encode()+b'\0')
        assert abs(env.data.qpos[env.qadr[-1]]-env.hand_spec.neutral[-1])<RESET_WRIST_TOLERANCE
        assert np.max(np.abs(env.data.qvel[env.hand_dofs]))<RESET_HAND_SPEED
        original=env.data.qvel[env.hand_dofs].copy()
        env.data.qvel[env.hand_dofs[0]]=2*RESET_HAND_SPEED
        assert not env._reset_ready()
        env.data.qvel[env.hand_dofs]=original
        shadow=BrakingActionCodec(env.config,env.codec.q)
        rng=np.random.default_rng(2000002)
        commands=[env.codec.q.copy()]
        for tick in range(60):
            if tick%5==0:
                env.valid=False
            action=rng.normal(size=16)
            expected=shadow.step(action,valid=env.valid)
            obs,_,done,trunc,_=env.step(action)
            np.testing.assert_allclose(env.codec.q,expected,atol=1e-12)
            np.testing.assert_allclose(obs[-16:]*env.config.max_velocity,env.codec.velocity,atol=1e-7)
            commands.append(env.codec.q.copy())
            if done or trunc:
                break
        velocity=np.diff(np.asarray(commands)[:,:16],axis=0)/DT
        acceleration=np.diff(np.vstack([np.zeros(16),velocity]),axis=0)/DT
        assert np.max(np.abs(velocity))<=.8+1e-10
        assert np.max(np.abs(acceleration))<=2+1e-10
    finally:
        env.close()


def test_legacy_wrist_and_controller_remain_unchanged():
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        env.reset(seed=2000000)
        assert env.model.eq_data[0,0]==pytest.approx(-.15)
        assert type(env.codec).__name__=='ActionCodec'
        assert abs(env.data.qpos[env.qadr[-1]]+.15)>.1
    finally:
        env.close()


def test_v8_resume_rejects_changed_actuator_envelope_before_creating_run(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import manipulation as cli
    previous=TaskConfigV8(object='pen',randomization=.3)
    args=SimpleNamespace(device='cpu',prefix_reset_checkpoint=None,object='pen',angle=15,turns=1,
        horizon=240,randomization=.3,mass_kg=None,friction=1.,delay_max_ms=150,
        command_bias_deg=2.,encoder_bias_deg=1.5,device_config=None,resume=tmp_path/'source.zip',
        output=tmp_path/'must-not-exist')
    with pytest.raises(ValueError,match='physical/action contract'):
        cli.train(args,config_class=TaskConfigV8,env_class=ManipulationV8Env,task_id=TASK_ID_V8,
                  config_loader=lambda _: (previous,{}),task_contract=task_contract())
    assert not args.output.exists()
