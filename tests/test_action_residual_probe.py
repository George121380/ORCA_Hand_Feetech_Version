from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv, TaskConfig


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    import probe_action_residuals
    return probe_action_residuals


def test_residual_search_is_bounded_reproducible_antithetic_and_rng_independent(probe):
    np.random.seed(71)
    before = np.random.get_state()
    a = np.asarray(probe.residual_candidates(12, 31))
    after = np.random.get_state()
    assert len(a) == 16 and a.shape == (16, 16)
    np.testing.assert_array_equal(a, probe.residual_candidates(12, 31))
    np.testing.assert_array_equal(a[::2], -a[1::2])
    assert np.max(np.abs(a[:8])) <= .15 and np.max(np.abs(a[8:])) <= .35
    assert not np.array_equal(a, probe.residual_candidates(13, 31))
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_zero_residual_exactly_replays_canonical_actions_observations_and_outcome(probe):
    class ObservablePolicy:
        def predict(self, obs, deterministic):
            assert obs.shape == (296,) and deterministic
            return np.sin(obs[:16]).astype(np.float32), None

    env = ManipulationEnv(TaskConfig(object="pen", angle=35, randomization=1, horizon=40))
    policy = ObservablePolicy()
    try:
        result, observations, actions = probe.rollout(env, policy, 7, np.zeros(16))
        obs, info = env.reset(seed=7)
        for i in range(result["steps"]):
            np.testing.assert_array_equal(obs, observations[i])
            action = policy.predict(obs, deterministic=True)[0]
            np.testing.assert_array_equal(action, actions[i])
            obs, _, done, truncated, info = env.step(action)
            assert (done or truncated) == (i+1 == result["steps"])
        assert result["success"] == info["is_success"]
        assert result["dropped"] == info["dropped"]
        assert result["final_error_deg"] == info["angle_error_deg"]
        assert observations.shape[1] == 296 and actions.shape[1] == 16
        with pytest.raises(ValueError, match="bounded"):
            probe.rollout(env, policy, 7, np.ones(16))
    finally:
        env.close()


def test_search_skips_non_horizon_and_preserves_canonical_initial_pairing(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ENV", object(), raising=False)
    monkeypatch.setattr(probe, "POLICY", object(), raising=False)
    monkeypatch.setattr(probe, "OUTPUT", tmp_path, raising=False)
    monkeypatch.setattr(probe, "SEARCH_SEED", 0, raising=False)
    monkeypatch.setattr(probe, "save_success", lambda *a: None)
    calls = []
    def rollout(*args):
        calls.append(args)
        return dict(end_reason="vision_timeout", success=False), None, None
    monkeypatch.setattr(probe, "rollout", rollout)
    assert not probe.search_seed(5)["rescued"] and len(calls) == 1
    calls.clear()
    def changed_initial(*args):
        calls.append(args)
        return dict(end_reason="horizon", success=False, initial=len(calls)), None, None
    monkeypatch.setattr(probe, "rollout", changed_initial)
    with pytest.raises(RuntimeError, match="initial state"):
        probe.search_seed(5)
