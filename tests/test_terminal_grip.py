from pathlib import Path

import numpy as np
import pytest

from orca_feetech.terminal_grip import terminal_knots,static_grip_score


def test_terminal_adjustment_preserves_early_trajectory_and_original_bounds():
    original=np.zeros((4,16))
    delta=np.zeros(16)
    delta[1]=.08
    actual=terminal_knots(original,delta,np.zeros(16),np.full(16,-1),np.ones(16))
    np.testing.assert_array_equal(actual[:2],original[:2])
    np.testing.assert_allclose(actual[:,1],[0,0,.04,.08])
    assert not np.any(actual[:,[j for j in range(16) if j!=1]])
    original[-1,1]=.29
    with pytest.raises(ValueError,match='0.3rad total'):
        terminal_knots(original,delta,np.zeros(16),np.full(16,-1),np.ones(16))
    with pytest.raises(ValueError,match='original ROM'):
        terminal_knots(np.zeros((4,16)),delta,np.zeros(16),np.full(16,-1),np.full(16,.07))


def test_static_contact_threshold_and_penetration_cannot_be_hidden_by_good_score():
    geometry=dict(excluded_support=False,self_penetration_m=0.,object_penetration_m=.001)
    assert static_grip_score(-.7,geometry,np.zeros(16)) is not None
    for dot in (None,-.38268,np.nan):
        assert static_grip_score(dot,geometry,np.zeros(16)) is None
    assert static_grip_score(-1,dict(geometry,object_penetration_m=.0021),np.zeros(16)) is None
    assert static_grip_score(-1,dict(geometry,excluded_support=True),np.zeros(16)) is None


def test_shared_rollout_respects_smaller_caller_budget_before_reset(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_rotation_contact_ik import run_continuation
    with pytest.raises(RuntimeError,match='budget exhausted'):
        run_continuation(None,None,None,None,None,tmp_path,dict(cases=[{}]*7),None,
                         maximum_cases=7,maximum_continuations=6)
