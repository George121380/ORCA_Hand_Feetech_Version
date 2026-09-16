import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("pinocchio")

from orca_feetech.official import make_retargeter
from orca_feetech.paths import workspace
from orca_feetech.vision import HandPoseExtractor

pytestmark = pytest.mark.teleop


def test_official_v1_aliases_resolve_all_finger_joints():
    retargeter = make_retargeter()
    assert retargeter.model_version == "v1"
    assert len(retargeter._model_q_indices) == 16
    assert retargeter._joint_aliases == {}


def test_retarget_landmarks_match_v1_geometry():
    """Check against independent URDF FK, including the missing human DIP joint."""
    import pinocchio as pin
    from orca_feetech.model import JOINT_NAMES, ModelSpec, poses
    from orca_feetech.paths import upstream
    r = make_retargeter()
    model = pin.buildModelFromUrdf(str(upstream("orcahand_description") / "v1/models/urdf/orcahand_right.urdf"))
    data = model.createData()
    spec = ModelSpec.load()
    for q in poses().values():
        # Official retargeting optimizes fingers with URDF wrist at zero.
        q = q.copy()
        q[-1] = spec.neutral[-1]
        full = pin.neutral(model)
        for i, name in enumerate(JOINT_NAMES):
            full[model.joints[model.getJointId("right_" + name)].idx_q] = q[i] - spec.neutral[i]
        pin.framesForwardKinematics(model, data, full)
        positions, _ = r._robot.compute_positions_and_jacobians(
            r._full_model_qpos(q[:16]), r._computed_frame_indices, r._frame_offsets)
        for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            tip = data.oMf[model.getFrameId(f"right_{finger}_fingertip")].translation
            np.testing.assert_allclose(positions[r._tip_indices[i]], tip, atol=1e-7)
            if finger != "thumb":
                pip = data.oMf[model.getFrameId(f"right_{finger}_ip")].translation
                np.testing.assert_allclose(positions[r._pip_indices[i]], pip, atol=1e-7)
                np.testing.assert_allclose(positions[r._dip_indices[i]], (pip + tip) / 2, atol=1e-7)
                assert np.linalg.norm(positions[r._pip_indices[i]] - positions[r._dip_indices[i]]) > 0.01


def test_real_rgb_detector_to_official_retargeter():
    cv2 = pytest.importorskip("cv2")
    from orca_teleop.retargeting.retargeter import TargetPose
    from orca_feetech.model import ModelSpec, from_orca
    image_path = workspace() / "local/fixtures/right_hands.jpg"
    if not image_path.is_file():
        pytest.skip("Run scripts/make_sample_video.py for Google's RGB integration fixture")
    frame = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    detector = HandPoseExtractor()
    retargeter = make_retargeter()
    try:
        target = None
        for index in range(35):
            result = detector.extract(frame, index * 50)
            assert result is not None
            world, _, _ = result
            assert world.shape == (21, 3)
            target = retargeter.retarget(TargetPose(world))
        assert target is not None
        q = from_orca(target)
        spec = ModelSpec.load()
        assert np.all(q[:16] >= spec.lower[:16] - 1e-7)
        assert np.all(q[:16] <= spec.upper[:16] + 1e-7)
        assert detector.extract(np.zeros_like(frame), 2000) is None
        with pytest.raises(ValueError, match="timestamps"):
            detector.extract(frame, 1999)
        pointing = cv2.cvtColor(cv2.imread(str(image_path.with_name("pointing_up.jpg"))), cv2.COLOR_BGR2RGB)
        for index in range(35):
            world, _, _ = detector.extract(pointing, 2100 + index * 50)
            target = retargeter.retarget(TargetPose(world))
        q = from_orca(target) - spec.neutral
        # The pointing fixture must extend the index and curl the middle finger.
        # This checks the RGB-to-joint semantics, not just finite solver output.
        from orca_feetech.model import JOINT_NAMES
        index_flex = sum(q[JOINT_NAMES.index("index_" + j)] for j in ("mcp", "pip"))
        middle_flex = sum(q[JOINT_NAMES.index("middle_" + j)] for j in ("mcp", "pip"))
        assert abs(index_flex) < np.deg2rad(30)
        assert middle_flex > index_flex + np.deg2rad(60)
    finally:
        detector.close()


def test_fixture_video_drops_stale_commands():
    import json
    path = workspace() / "runs/teleop-fixture/trajectory.jsonl"
    if not path.is_file():
        pytest.skip("Run the full fixture teleop command to check persisted dropout behavior")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    gaps = [(a, b) for a, b in zip(rows, rows[1:]) if not b["input_valid"]]
    assert len(gaps) >= 10
    for previous, current in gaps:
        np.testing.assert_array_equal(current["command"], previous["command"])
