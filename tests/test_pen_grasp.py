from dataclasses import asdict
import json

import mujoco
import numpy as np
import pytest

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import HISTORY,FRAME_SIZE
from orca_feetech.manipulation_v8 import load_v8_config
from orca_feetech.model import JOINT_NAMES
from orca_feetech.paths import provenance
from orca_feetech.pen_grasp import (GraspConfig,PenGraspEnv,TASK_ID_GRASP,grasp_contract,
                                    load_grasp_config,grasp_asset)


def test_grasp_contract_is_distinct_and_manifest_is_pinned(tmp_path):
    cfg=GraspConfig()
    assert len(grasp_asset()['placement_offsets_m'])==7
    meta=dict(task_id=TASK_ID_GRASP,task=asdict(cfg),fingerprint=cfg.fingerprint(),joint_names=JOINT_NAMES,
              history=HISTORY,frame_size=FRAME_SIZE,upstream=provenance(),task_contract=grasp_contract())
    (tmp_path/'config.json').write_text(json.dumps(meta))
    checkpoint=tmp_path/'policy.zip'
    assert isinstance(load_grasp_config(checkpoint)[0],GraspConfig)
    for other in (load_config,load_v8_config):
        with pytest.raises(ValueError):
            other(checkpoint)
    meta['task_contract']['failures']['penetration_m']=.02
    (tmp_path/'config.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='contract mismatch'):
        load_grasp_config(checkpoint)
    for kwargs in (dict(object='cube'),dict(angle=15),dict(turns=2),dict(horizon=240)):
        with pytest.raises(ValueError):
            GraspConfig(**kwargs)


def test_real_grasp_holds_full_seven_seconds_and_never_gives_early_success():
    env=PenGraspEnv()
    try:
        obs,initial=env.reset(seed=2020000,options={'placement_index':5})
        assert env.observation_space.contains(obs) and initial['reset_failure'] is None
        assert not initial['is_success'] and initial['settle_seconds']==0
        assert np.max(np.abs(env.reached_hand_qvel))>1e-8  # Real preparation velocity was preserved.
        assert np.linalg.norm(env.data.qvel[env.ov:env.ov+6])==0  # Object released once at rest.
        initial_position=env.pose()[0]
        commands=[env.codec.q.copy()]
        for tick in range(140):
            obs,reward,done,truncated,info=env.step(np.zeros(16))
            commands.append(env.codec.q.copy())
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if tick<139:
                assert not done and not truncated and not info['is_success']
        assert done and not truncated and info['is_success'] and info['termination_reason']=='success'
        assert env.data.time==pytest.approx(7.)
        assert info['loaded_opposition'] and info['resting']
        assert info['final_500ms_orientation_span_deg']<2 and info['final_500ms_position_span_m']<.002
        assert info['peak_object_penetration_m']<=.002
        assert np.linalg.norm(env.pose()[0]-initial_position)>.001  # Free dynamics, no pose tracking injection.
        assert np.max(np.abs(np.diff(commands,axis=0)))<1e-10
        # This is a loaded hold, despite substantial settling rotation, not a rotation-goal success.
        assert info['angle_error_deg']>8 and info['goal_angle_deg']==0
    finally:
        env.close()


def test_manual_release_preserves_hand_state_and_invalid_start_is_counted_without_retry():
    env=PenGraspEnv()
    try:
        env.reset(seed=2020001,options={'placement_index':0})
        env.data.qvel[env.hand_dofs]=np.linspace(-.02,.02,17)
        q=env.data.qpos[env.qadr].copy()
        v=env.data.qvel[env.hand_dofs].copy()
        env._place_object([.6,0,.8],np.eye(3))
        np.testing.assert_array_equal(env.data.qpos[env.qadr],q)
        np.testing.assert_array_equal(env.data.qvel[env.hand_dofs],v)
        env.reset_failure='test_invalid_initial_geometry'
        before=env.data.qpos.copy()
        _,reward,done,_,info=env.step(np.zeros(16))
        assert done and reward==-20 and not info['is_success']
        assert info['termination_reason']=='test_invalid_initial_geometry' and env.steps==0
        np.testing.assert_array_equal(env.data.qpos,before)  # Do not integrate a rejected initial overlap.
        with pytest.raises(ValueError):
            env.reset(options={'placement_index':7})
    finally:
        env.close()


def test_floor_or_excess_overlap_cannot_count_as_grasp_success():
    env=PenGraspEnv()
    try:
        env.reset(seed=2020002,options={'placement_index':1})
        env.data.qpos[env.oq:env.oq+3]=[.6,0,.04]
        mujoco.mj_forward(env.model,env.data)
        _,_,done,_,info=env.step(np.zeros(16))
        assert done and not info['is_success']
        assert info['termination_reason'] in ('dropped','displacement_limit','excluded_support')
    finally:
        env.close()
