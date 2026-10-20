from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ObservationCodec,TaskConfig,yaw_rotation


def test_goal_audit_keeps_real_preparation_history_and_all_non_goal_inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_rotation_goal_response import counterfactual_goal
    from orca_feetech.model import ModelSpec
    q=ModelSpec.load().neutral
    codec=ObservationCodec(TaskConfig())
    current,opposite=yaw_rotation(.26),yaw_rotation(-.26)
    args=(q,q,np.array([.1,.02,.2]),np.eye(3))
    for goal in (np.eye(3),current,current):
        obs=codec.push(*args,goal,np.zeros(3),True,0,controller_velocity=np.arange(16)*.01)
    before=obs.copy()
    changed,mask=counterfactual_goal(obs,current,opposite)
    assert mask.tolist()==[False,False,True,True]
    np.testing.assert_array_equal(obs,before)
    frames=changed.reshape(4,74)
    np.testing.assert_array_equal(frames[:2],obs.reshape(4,74)[:2])
    np.testing.assert_array_equal(frames[:,np.r_[0:47,56:74]],obs.reshape(4,74)[:,np.r_[0:47,56:74]])
    np.testing.assert_allclose(frames[-1,47:56],opposite.ravel(),atol=1e-7)
    with pytest.raises(ValueError,match='Latest observation'):
        counterfactual_goal(obs,yaw_rotation(.8),opposite)


def test_signed_progress_uses_world_axis_and_distinguishes_off_axis_rotation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_rotation_goal_response import rotation_components
    from scipy.spatial.transform import Rotation
    start=Rotation.from_euler('x',90,degrees=True).as_matrix()
    end=yaw_rotation(np.deg2rad(15))@start
    result=rotation_components(start,end,1)
    assert result['signed_goal_yaw_progress_deg']==pytest.approx(15.)
    assert result['off_axis_rotation_deg']==pytest.approx(0.,abs=1e-10)
    assert rotation_components(start,end,-1)['signed_goal_yaw_progress_deg']==pytest.approx(-15.)
    roll=Rotation.from_euler('x',10,degrees=True).as_matrix()@start
    result=rotation_components(start,roll,1)
    assert result['signed_goal_yaw_progress_deg']==pytest.approx(0.)
    assert result['off_axis_rotation_deg']==pytest.approx(10.)
