from copy import deepcopy
from pathlib import Path

import numpy as np


def test_audit_exposes_transient_endpoint_success_and_does_not_promote_it(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from audit_rotation_policy import failure_summary
    row=dict(position=[0,0,0],rotation=np.eye(3).tolist(),contact=True,
        info=dict(angle_error_deg=2.,loaded_opposition=True,resting=True,stable=True))
    samples=[deepcopy(row) for _ in range(140)]
    samples[-10]['info'].update(angle_error_deg=10.,stable=False)
    end=dict(prefix_grasp_success=True,termination_reason='horizon',is_success=False)
    result=failure_summary(samples,end)
    assert result['tail_counts']['pose']==19
    assert result['reasons']==['incomplete_final_1s_pose']
    assert not result['native_success']


def test_audit_reports_contact_and_motion_failures_even_with_good_pose(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from audit_rotation_policy import failure_summary
    samples=[dict(position=[0,0,0],rotation=np.eye(3).tolist(),contact=True,
        info=dict(angle_error_deg=4.,loaded_opposition=False,resting=False,stable=False)) for _ in range(140)]
    result=failure_summary(samples,dict(prefix_grasp_success=True,termination_reason='horizon',is_success=False))
    assert result['tail_counts']['pose']==20
    assert set(result['reasons'])=={'incomplete_final_1s_contact','incomplete_final_1s_rest'}


def test_failed_prefix_is_not_classified_as_successful_rotation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from audit_rotation_policy import failure_summary
    assert failure_summary([],dict(prefix_grasp_success=False))==dict(reasons=['prefix_failed'],tail_ticks=0)
