import numpy as np
import pytest

from orca_feetech.manipulation import DT
from orca_feetech.pen_rotate import RotationConfig
from orca_feetech.model import ModelSpec
from orca_feetech.reference_control import BrakingActionCodec
from orca_feetech.rotation_trajectory import trajectory_target,trajectory_cost


def test_two_knots_hold_to_horizon_and_inherit_existing_braking_velocity():
    config=RotationConfig()
    codec=BrakingActionCodec(config,ModelSpec.load().neutral)
    codec.step(np.full(16,.25))
    initial=codec.q.copy()
    previous_velocity=codec.velocity.copy()
    assert np.linalg.norm(previous_velocity)>0
    offsets=np.vstack([np.full(16,.18),np.full(16,-.18)])
    targets=[]
    for tick in range(140):
        target=trajectory_target(initial,offsets,(tick+1)*DT,codec.lower,codec.upper)
        targets.append(target)
        action=np.clip((target-codec.q[:16])/(config.max_velocity*DT),-1,1)
        old=codec.q.copy()
        command=codec.step(action)
        velocity=(command[:16]-old[:16])/DT
        assert max(abs(velocity))<=.8+1e-10
        assert max(abs(velocity-previous_velocity))/DT<=2.+1e-10
        assert np.all(command>=codec.lower) and np.all(command<=codec.upper)
        previous_velocity=velocity
    np.testing.assert_allclose(targets[99:],np.tile(targets[-1],(41,1)),atol=1e-12)
    np.testing.assert_allclose(trajectory_target(initial,offsets,0,codec.lower,codec.upper),initial[:16])
    for invalid in (np.full((2,16),.181),np.full((2,16),np.nan),np.zeros((1,16))):
        with pytest.raises(ValueError):
            trajectory_target(initial,invalid,1.,codec.lower,codec.upper)


def test_censored_goal_hit_and_prefix_success_do_not_beat_a_complete_candidate():
    def rows(error=15.,success=False):
        return [dict(angle_error_deg=error,linear_speed_mps=0.,angular_speed_rps=0.,
            handoff_displacement_m=0.,loaded_opposition=True,is_success=success,
            source_grasp_completed=True)]*20
    offsets=np.zeros((2,16))
    complete=trajectory_cost(rows(),offsets,complete=True,command_limits_met=True)
    for kwargs in (dict(complete=False,command_limits_met=True),dict(complete=True,command_limits_met=False)):
        assert trajectory_cost(rows(0,True),offsets,**kwargs)>complete
    assert complete>0  # A successful prefix alone never gets success credit.
    assert trajectory_cost(rows(1,True),offsets,complete=True,command_limits_met=True)<0
