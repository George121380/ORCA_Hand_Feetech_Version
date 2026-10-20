from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    import probe_second_subgoal
    return probe_second_subgoal


def test_residual_starts_only_after_real_handoff_and_preserves_history(probe, monkeypatch):
    class Policy:
        def __init__(self, value):
            self.value, self.observations = value, []

        def predict(self, obs, deterministic):
            assert deterministic
            self.observations.append(obs.copy())
            return np.full(16, self.value, np.float32), None

    class Env:
        config = SimpleNamespace(horizon=4)
        direction = 1
        goal = np.eye(3)

        def reset(self, seed):
            self.completed = self.steps = 0
            self.actions = []
            return np.zeros(296), dict(object_position=[0, 0, 1])

        def pose(self):
            return np.zeros(3), np.eye(3)

        def step(self, action):
            self.actions.append(action.copy())
            self.steps += 1
            if self.steps == 2:
                self.completed = 1
            if self.steps == 4:
                self.completed = 2
            return (np.full(296, self.steps), 0., self.steps == 4, False,
                    dict(is_success=self.steps == 4, dropped=False, vision_age=0.,
                         completed_turns=self.completed, angle_error_deg=4., shaft_error_deg=3., stable=True))

    monkeypatch.setattr(probe, "handoff_signature", lambda env, obs: str(env.steps)+str(obs[0]))
    env, first, candidate = Env(), Policy(.2), Policy(.8)
    baseline = probe.rollout(env, candidate, 3, first=first)
    prefix = np.array(env.actions[:2])
    changed = probe.rollout(env, candidate, 3, first=first, residual=np.full(16, .3))
    np.testing.assert_array_equal(prefix, env.actions[:2])
    np.testing.assert_array_equal(env.actions[2:], np.ones((2, 16)))
    assert changed["handoff"] == baseline["handoff"] == dict(step=2, signature="22")
    assert changed["second_steps"] == changed["second_longest_stable_steps"] == 2
    assert changed["second_minimum_error_deg"] == 4
    # The new actor receives the actual first actor's final observation.
    np.testing.assert_array_equal(candidate.observations[-2], np.full(296, 2))
    with pytest.raises(ValueError, match="bounded"):
        probe.rollout(env, candidate, 3, residual=np.ones(16))


def test_search_rejects_changed_physical_handoff_and_skips_camera_failure(probe, monkeypatch):
    for name in ("ENV", "FIRST", "CANDIDATE"):
        monkeypatch.setattr(probe, name, object(), raising=False)
    calls = []

    def camera(*args, **kwargs):
        calls.append(kwargs)
        return dict(initial="same", end_reason="vision_timeout", completed_subgoals=1)

    monkeypatch.setattr(probe, "rollout", camera)
    assert not probe.audit_seed(160)["rescued"] and len(calls) == 2
    calls.clear()

    def changed(*args, **kwargs):
        calls.append(kwargs)
        return dict(initial="same", end_reason="horizon", completed_subgoals=1,
                    handoff="changed" if "residual" in kwargs else "same")

    monkeypatch.setattr(probe, "rollout", changed)
    with pytest.raises(RuntimeError, match="physical first-skill handoff"):
        probe.audit_seed(160)
    assert len(calls) == 3
