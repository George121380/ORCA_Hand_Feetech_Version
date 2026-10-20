import numpy as np
import pytest

from orca_feetech.manipulation import ActionCodec, TaskConfig, DT
from orca_feetech.model import ModelSpec


@pytest.mark.parametrize("side", [-1, 1])
def test_counterfactual_removes_limit_windup_and_reverses_on_next_tick(monkeypatch, side):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from audit_controller_state import LimitProjectionProbeCodec
    cfg = TaskConfig(object="pen")
    spec = ModelSpec.load()
    initial = spec.upper if side == 1 else spec.lower
    canonical = ActionCodec(cfg, initial)
    probe = LimitProjectionProbeCodec(cfg, initial)
    outward = np.full(16, side)
    for _ in range(12):
        canonical.step(outward)
        probe.step(outward)
    assert np.array_equal(canonical.q, probe.q)
    assert np.allclose(canonical.velocity, side*cfg.max_velocity)
    assert np.array_equal(probe.velocity, np.zeros(16))
    before = probe.q.copy()
    canonical.step(-outward)
    probe.step(-outward)
    assert np.array_equal(canonical.q, before)
    assert np.allclose(probe.q[:16]-before[:16], -side*cfg.max_acceleration*DT**2)
    assert np.all(probe.q >= spec.lower) and np.all(probe.q <= spec.upper)


def test_probe_preserves_commands_and_filter_away_from_limits_and_on_vision_hold(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from audit_controller_state import LimitProjectionProbeCodec
    cfg = TaskConfig(object="pen")
    spec = ModelSpec.load()
    initial = (spec.lower+spec.upper)/2
    canonical = ActionCodec(cfg, initial)
    probe = LimitProjectionProbeCodec(cfg, initial)
    for action in [np.full(16, .1), np.full(16, -.1)]*3:
        assert np.array_equal(canonical.step(action), probe.step(action))
        assert np.array_equal(canonical.velocity, probe.velocity)
    assert np.array_equal(canonical.step(np.ones(16), valid=False),
                          probe.step(np.ones(16), valid=False))
    assert np.array_equal(probe.velocity, np.zeros(16))
