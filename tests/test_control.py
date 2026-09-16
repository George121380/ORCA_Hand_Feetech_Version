import json

import numpy as np
import pytest

from orca_feetech.control import CommandGuard, JointTarget, minimum_jerk, move_duration
from orca_feetech.model import JOINT_NAMES, ModelSpec, from_orca, poses, to_orca
from orca_feetech.recording import load_trajectory


@pytest.mark.parametrize("bad", [np.zeros(16), np.full(17, np.nan), np.full(17, np.inf)])
def test_reject_bad_commands(bad):
    with pytest.raises(ValueError):
        JointTarget(bad)


def test_named_order_and_unit_roundtrip():
    q = poses()["pinch"]
    np.testing.assert_allclose(from_orca(to_orca(q)), q, atol=1e-14)
    with pytest.raises(ValueError):
        JointTarget(q, joint_names=JOINT_NAMES[::-1])


def test_guard_rejects_expired_and_future_and_out_of_order_frames():
    spec = ModelSpec.load()
    guard = CommandGuard(spec.neutral)
    for stamp in [0, 10.1]:
        q, fresh = guard.apply(JointTarget(poses()["fist"], timestamp=stamp), 0.05, now=10)
        assert not fresh
        np.testing.assert_array_equal(q, spec.neutral)
    guard.apply(JointTarget(poses()["fist"], timestamp=10), 0.05, now=10)
    with pytest.raises(ValueError, match="Out-of-order"):
        guard.apply(JointTarget(poses()["fist"], timestamp=9.99), 0.05, now=10)


def test_guard_limits_velocity_and_starts_with_bounded_acceleration():
    spec = ModelSpec.load()
    guard = CommandGuard(spec.neutral)
    previous = spec.neutral.copy()
    q, _ = guard.apply(JointTarget(spec.upper * 100, timestamp=1), 0.05, now=1)
    assert np.max(np.abs(q - previous)) <= 2 * 0.05**2 + 1e-12
    for i in range(1, 100):
        previous = q
        q, _ = guard.apply(JointTarget(spec.upper * 100, timestamp=i + 1), 0.05, now=i + 1)
        assert np.max(np.abs(q - previous)) <= 0.8 * 0.05 + 1e-12
        assert np.all(q >= spec.lower) and np.all(q <= spec.upper)


def test_minimum_jerk_endpoints_and_derivatives():
    a, b = poses()["open"], poses()["fist"]
    duration = move_duration(a, b)
    np.testing.assert_allclose(minimum_jerk(a, b, 0, duration), a)
    np.testing.assert_allclose(minimum_jerk(a, b, duration, duration), b)
    times = np.linspace(0, duration, 3001)
    samples = np.stack([minimum_jerk(a, b, t, duration) for t in times])
    velocity = np.gradient(samples, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    assert np.max(np.abs(velocity)) <= 0.8 + 0.001
    assert np.max(np.abs(acceleration)) <= 2.0 + 0.001
    assert np.max(np.abs(velocity[[0, -1]])) < 0.001


def test_recording_units_are_validated_before_replay(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    path.write_text(json.dumps({"time_s": 0, "command": [0] * 17}) + "\n")
    metadata = {"schema_version": 1, "model_id": "orca-v1-right", "joint_names": JOINT_NAMES,
                "units": "degrees", "status": "complete"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="units"):
        load_trajectory(path)


def test_official_base_hand_api_preserves_uncommanded_joints():
    from orca_core import BaseHand
    from orca_feetech.api import OrcaHandController
    with OrcaHandController("mock") as hand:
        assert isinstance(hand, BaseHand)
        before = from_orca(hand.get_joint_position())
        hand.set_joint_positions({"index_mcp": 30.0})
        after = from_orca(hand.get_joint_position())
        changed = np.flatnonzero(abs(after - before) > 1e-9)
        assert changed.tolist() == [JOINT_NAMES.index("index_mcp")]
