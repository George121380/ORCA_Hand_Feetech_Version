from pathlib import Path

import mujoco
import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv, TaskConfig


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_physical_trajectory
    return probe_physical_trajectory


def test_smooth_targets_join_at_rest_with_actual_position_difference_limits(probe):
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        initial=env.hand_spec.neutral.copy()
        offsets=np.vstack([np.full(16,.3),np.full(16,-.3)])
        samples=[dict(command=np.r_[probe.smooth_target(initial,offsets,t,env.lower,env.upper),initial[-1]])
                 for t in np.arange(1,81)*.05]
        result=probe.command_kinematics(initial,samples,env.config)
        assert result['peak_velocity_rad_s']<=.8
        assert result['peak_acceleration_rad_s2']<=2
        commands=np.array([s['command'] for s in samples])
        assert np.all(commands>=env.lower) and np.all(commands<=env.upper)
        np.testing.assert_allclose(commands[59:],commands[-1][None,:]+np.zeros((21,17)),atol=1e-12)
        for invalid in (np.zeros((1,16)),np.full((2,16),.31),np.full((2,16),np.nan)):
            with pytest.raises(ValueError,match='two finite 16-joint offsets'):
                probe.smooth_target(initial,invalid,.1,env.lower,env.upper)
    finally:
        env.close()


def test_real_physics_branches_preserve_settled_velocity_and_replay_deterministically(probe,monkeypatch):
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        env.reset(seed=1970000,options={'direction':1})
        state=probe.branch_state(env)
        original=state.copy()
        qpos,qvel=env.data.qpos.copy(),env.data.qvel.copy()
        command=env.codec.q.copy()
        assert np.linalg.norm(qvel[env.ov:env.ov+6])>1e-8
        step=mujoco.mj_step
        seen=[]
        def first_step(model,data,**kwargs):
            if not seen:
                np.testing.assert_array_equal(data.qpos,qpos)
                np.testing.assert_array_equal(data.qvel,qvel)
                seen.append(True)
            return step(model,data,**kwargs)
        monkeypatch.setattr(mujoco,'mj_step',first_step)
        offsets=np.zeros((2,16))
        run=probe.trajectory(env,state,command,offsets,1)
        assert seen and run['complete'] and run['simulated_seconds']==pytest.approx(4.)
        assert run['initial']['angular_speed_rps']==pytest.approx(np.linalg.norm(qvel[env.ov+3:env.ov+6]))
        env.data.warning[0].number=3  # Metadata from another candidate must not leak.
        again=probe.trajectory(env,state,command,offsets,1,fine_collision_audit=True)
        np.testing.assert_array_equal(state,original)
        np.testing.assert_allclose(again['end']['object_position_m'],run['end']['object_position_m'],atol=1e-12)
        assert again['numerical_failure'] is None and not again['warnings']
        assert run['command_limits_met'] and again['command_limits_met']
        assert not run['diagnostic_motion_preserved']  # No motion cannot pass a 10deg/2deg target.
        assert not run['acceptance_met']
        moving=probe.trajectory(env,state,command,np.vstack([np.full(16,.3),np.full(16,-.3)]),1)
        assert moving['command_limits_met']
        for sample in moving['samples']:
            assert sample['command'][-1]==env.hand_spec.neutral[-1]
    finally:
        env.close()


def test_independent_rollout_does_not_freeze_an_unsupported_object(probe):
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        command=env.hand_spec.neutral.copy()
        env.data.qpos[env.qadr]=command
        env.data.ctrl[env.actuators]=command
        env.data.qpos[env.oq:env.oq+7]=[.6,0,.8,1,0,0,0]
        mujoco.mj_forward(env.model,env.data)
        result=probe.trajectory(env,probe.branch_state(env),command,np.zeros((2,16)),1)
        assert result['end']['object_position_m'][2]<.7
        assert result['end']['linear_speed_mps']>1
        assert not result['complete'] and not result['feasible_diagnostic']
        assert not result['diagnostic_motion_preserved']
    finally:
        env.close()


def test_extra_hold_advances_existing_physics_without_zeroing_velocity(probe):
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        info,command,preparation=probe.prepare(env,1970000,1,5.)
        assert info['settle_seconds']>0
        assert len(preparation['samples'])==100
        assert env.data.time==pytest.approx(5.)
        assert preparation['end']['rotation_change_deg']>1.
        assert not preparation['velocities_rewritten']
        assert np.linalg.norm(env.data.qvel[env.ov:env.ov+6])>1e-10
        np.testing.assert_array_equal(command,env.hand_spec.neutral)
        with pytest.raises(ValueError,match='zero or five second'):
            probe.prepare(env,1970000,1,6.)
    finally:
        env.close()
