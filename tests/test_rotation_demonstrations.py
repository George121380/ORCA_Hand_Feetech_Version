from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from orca_feetech.rotation_demonstrations import training_states,demonstration_admission


def test_training_seeds_balanced_and_disjoint_from_development_and_formal():
    states=training_states()
    assert [s['seed'] for s in states]==list(range(3111000,3111028))
    for direction in (-1,1):
        assert [s['placement_index'] for s in states if s['direction']==direction]==list(range(7))*2
    assert not {s['seed'] for s in states}.intersection(range(3211000,3211028))
    assert not {s['seed'] for s in states}.intersection(range(3311000,3311100))


def test_admission_counts_distinct_states_and_requires_both_directions_and_native_gates():
    cases=[dict(s,native_task_success=True,full_uncensored_duration=True,command_limits_met=True,
                end=dict(is_success=True,dropped=False)) for s in training_states()[:8]]
    assert not demonstration_admission(cases*3)['imitation_admitted']  # Only one positive state.
    cases.append(dict(cases[-1],seed=3111008,placement_index=1))
    assert demonstration_admission(cases)['imitation_admitted']
    cases[-1]['command_limits_met']=False
    assert not demonstration_admission(cases)['imitation_admitted']
    assert demonstration_admission(cases)['successful_states']==8


def test_shared_rollout_keeps_negative_goal_and_failed_prefix_without_retry(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_rotation_contact_ik as probe
    options=[]
    initial=dict(prefix_grasp_success=False,angle_error_deg=15.,termination_reason='grasp_prefix_failed',
                 dropped=False,is_success=False)
    def reset(**kwargs):
        options.append(kwargs)
        return np.zeros(296),initial
    carrier=SimpleNamespace(codec=SimpleNamespace(q=np.zeros(17),velocity=np.zeros(17)),
                            origin=np.zeros(3),data=SimpleNamespace(time=0.))
    env=SimpleNamespace(reset=reset,carrier=carrier,pose=lambda:(np.zeros(3),np.eye(3)),goal=np.eye(3))
    monkeypatch.setattr(probe,'paired_signature',lambda _: 'fixed')
    monkeypatch.setattr(probe,'contacts',lambda _: [])
    report=dict(cases=[])
    (tmp_path/'cases').mkdir()
    result=probe.run_continuation(env,dict(seed=3111000,placement_index=0,direction=-1),
        np.zeros((4,16)),'fixed','negative',tmp_path,report,lambda:None,
        maximum_cases=1,maximum_continuations=1)
    assert options==[dict(seed=3111000,options=dict(placement_index=0,direction=-1))]
    assert not result['native_task_success'] and result['simulated_seconds']==0
    with pytest.raises(RuntimeError,match='budget exhausted'):
        probe.run_continuation(None,None,None,None,None,None,dict(cases=[dict(end={})]),None,
                               maximum_cases=10,maximum_continuations=1)
