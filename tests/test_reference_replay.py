import copy
from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ActionCodec, ManipulationEnv, TaskConfig
from orca_feetech.model import ModelSpec


@pytest.fixture
def replay_probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_reference_replay
    return probe_reference_replay


def reference(q0,q1):
    return {'trials':[
        dict(time_s=0.,qpos=q0.tolist(),baseline={'qpos':q0.tolist()},
             object_position_m=[.6,0,.8],object_quaternion_wxyz=[1.,0,0,0]),
        dict(time_s=.5,qpos=q1.tolist(),baseline={'qpos':q1.tolist()},
             object_position_m=[.6,0,.8],object_quaternion_wxyz=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]),
    ]}


def test_trajectory_interpolates_joint_positions_and_proper_object_rotation(replay_probe):
    q0=np.zeros(16)
    q1=np.ones(16)*.2
    source=reference(q0,q1)
    trajectory=replay_probe.Trajectory(source,'refined')
    q,pos,rot=trajectory.sample(.25)
    np.testing.assert_allclose(q,.1)
    np.testing.assert_allclose(pos,[.6,0,.8])
    np.testing.assert_allclose(rot[:,1],[-np.sqrt(.5),np.sqrt(.5),0],atol=1e-12)
    assert np.linalg.det(rot)==pytest.approx(1)
    # Antipodal quaternion representations must produce identical interpolation.
    opposite=copy.deepcopy(source)
    opposite['trials'][1]['object_quaternion_wxyz']=(-np.array(source['trials'][1]['object_quaternion_wxyz'])).tolist()
    np.testing.assert_allclose(replay_probe.Trajectory(opposite,'refined').sample(.25)[2],rot)
    source['trials'][1]['time_s']=0
    with pytest.raises(ValueError,match='strictly ordered'):
        replay_probe.Trajectory(source,'refined')


def test_replay_obeys_shared_velocity_acceleration_limits_without_teleporting_object(replay_probe):
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        q0=env.hand_spec.neutral[:16].copy()
        q1=q0+.3*(env.upper[:16]-q0)
        trajectory=replay_probe.Trajectory(reference(q0,q1),'refined')
        result=replay_probe.replay(env,trajectory,1.)
        assert result['termination']=='dropped_or_left_initial_workspace'
        assert not result['command_schedule_completed'] and not result['acceptance_met']
        assert result['numerical_failure'] is None and not result['warnings']
        assert 0<result['control_ticks']<=10
        assert result['end']['object_position_m'][2]<.7
        assert result['end']['reference_position_m'][2]==.8
        assert result['end']['reference_position_error_m']>.12
        commands=np.array([np.r_[q0,env.hand_spec.neutral[-1]],*[s['command'] for s in result['samples']]])
        velocities=np.array([np.zeros(16),*[s['controller_velocity_rad_s'] for s in result['samples']]])
        assert np.max(np.abs(velocities))<=.8+1e-10
        assert np.max(np.abs(np.diff(velocities,axis=0)))/.05<=2+1e-10
        np.testing.assert_allclose(np.diff(commands[:,:16],axis=0)/.05,velocities[1:],atol=1e-12)
        np.testing.assert_allclose(commands[:,-1],env.hand_spec.neutral[-1])
        with pytest.raises(ValueError,match='between 1 and 4'):
            replay_probe.replay(env,trajectory,5.)
    finally:
        env.close()


def test_command_audit_detects_existing_codec_acceleration_jump_at_rom_limit(replay_probe):
    config=TaskConfig(object='pen',randomization=0)
    spec=ModelSpec.load()
    initial=spec.upper.copy()
    initial[:16]-=.03
    initial[-1]=spec.neutral[-1]
    codec=ActionCodec(config,initial)
    samples=[]
    velocities=[]
    for _ in range(5):
        samples.append({'command':codec.step(np.ones(16)).tolist()})
        velocities.append(codec.velocity.copy())
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(16),velocities]),axis=0)))/.05<=2+1e-10
    audit=replay_probe.command_kinematics(initial,samples,config)
    assert audit['peak_velocity_rad_s']<=.8+1e-10
    assert audit['peak_acceleration_rad_s2']>5
    assert audit['acceleration_exceeded_samples']>0
