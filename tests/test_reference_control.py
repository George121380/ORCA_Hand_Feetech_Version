import numpy as np
import pytest

from orca_feetech.manipulation import DT, TaskConfig
from orca_feetech.model import ModelSpec
from orca_feetech.reference_control import BrakingActionCodec, stoppable_speed


def test_discrete_stopping_bound_matches_full_future_deceleration_distance():
    distance=np.geomspace(1e-10,2.,100)
    velocity=stoppable_speed(distance,2.)
    k=np.arange(int(np.ceil(velocity.max()/(2*DT)))+2)
    total=DT*np.maximum(velocity[:,None]-k*2*DT,0).sum(axis=1)
    np.testing.assert_allclose(total,distance,rtol=1e-10,atol=1e-12)
    faster=DT*np.maximum(velocity[:,None]+1e-7-k*2*DT,0).sum(axis=1)
    assert np.all(faster>distance)


@pytest.mark.parametrize('start',['neutral','lower','upper'])
def test_effective_position_commands_obey_rom_velocity_and_acceleration_under_abrupt_actions(start):
    spec=ModelSpec.load()
    initial=getattr(spec,start).copy()
    initial[-1]=spec.neutral[-1]
    codec=BrakingActionCodec(TaskConfig(object='pen'),initial)
    rng=np.random.default_rng(108)
    commands=[initial]
    for tick in range(2500):
        # Long saturated pushes, frequent reversals, random actions, and
        # intermittent invalid observations stress both braking boundaries.
        action=(np.ones(16)*(1 if (tick//75)%2 else -1) if tick%3 else rng.normal(0,3,16))
        commands.append(codec.step(action,valid=tick%19>=3))
        np.testing.assert_allclose((commands[-1][:16]-commands[-2][:16])/DT,codec.velocity,atol=1e-12)
    commands=np.asarray(commands)
    velocity=np.diff(commands[:,:16],axis=0)/DT
    acceleration=np.diff(np.vstack([np.zeros(16),velocity]),axis=0)/DT
    assert np.max(np.abs(velocity))<=.8+1e-10
    assert np.max(np.abs(acceleration))<=2+1e-10
    assert np.all(commands>=spec.lower-1e-12) and np.all(commands<=spec.upper+1e-12)
    np.testing.assert_allclose(commands[:,-1],spec.neutral[-1])


def test_boundary_push_stops_without_clipping_then_reverses_and_invalid_observation_brakes():
    spec=ModelSpec.load()
    initial=spec.upper.copy()
    initial[:16]-=.03
    initial[-1]=spec.neutral[-1]
    codec=BrakingActionCodec(TaskConfig(object='pen'),initial)
    commands=[initial]
    for _ in range(20):
        commands.append(codec.step(np.ones(16)))
    np.testing.assert_allclose(codec.q[:16],spec.upper[:16],atol=1e-10)
    np.testing.assert_allclose(codec.velocity,0,atol=1e-10)
    for _ in range(10):
        commands.append(codec.step(-np.ones(16)))
    assert np.max(codec.velocity)<-.7
    before=codec.q.copy()
    for _ in range(10):
        commands.append(codec.step(np.ones(16),valid=False))
    assert np.max(np.abs(codec.velocity))<1e-10
    assert np.all(codec.q[:16]<before[:16])  # Decelerated rather than froze.
    velocity=np.diff(np.asarray(commands)[:,:16],axis=0)/DT
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(16),velocity]),axis=0)))/DT<=2+1e-10
    with pytest.raises(ValueError,match='finite actions'):
        codec.step(np.full(16,np.nan))
    invalid=spec.upper.copy()
    invalid[0]+=.001
    with pytest.raises(ValueError,match='within ROM'):
        BrakingActionCodec(TaskConfig(object='pen'),invalid)
