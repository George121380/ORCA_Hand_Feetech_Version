import cv2
import mujoco
import numpy as np
import pytest

from orca_feetech.fiducials import solve_pose, transform, TagTracker
from orca_feetech.manipulation import (ManipulationEnv, TaskConfig, ActionCodec, ObservationCodec,
                                     DT, rotation_error)
from orca_feetech.model import ModelSpec
from orca_feetech.objects import geometry, marker_layout, DICTIONARY, collision_parts, mesh_volume


@pytest.mark.parametrize("kind", ["cube", "pen"])
def test_print_mesh_is_closed_and_labels_fit_inside_faces(kind):
    vertices, faces = geometry(kind)
    edges = {}
    volume = 0
    for face in faces:
        a, b, c = vertices[face]
        volume += np.dot(a, np.cross(b, c)) / 6
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i+1) % 3]))))
            edges[edge] = edges.get(edge, 0) + 1
    assert volume > 0
    assert set(edges.values()) == {2}
    for tag in marker_layout(kind):
        center, u, v = np.array(tag["center_m"]), np.array(tag["u"]), np.array(tag["v"])
        paper = np.array([center + tag["paper_size_m"]/2 * (a*u+b*v)
                          for a, b in ((-1, 1), (1, 1), (1, -1), (-1, -1))])
        fits = []
        for part_vertices, part_faces in collision_parts(kind):
            errors = []
            for face in part_faces:
                a, b, c = part_vertices[face]
                n = np.cross(b-a, c-a)
                n /= np.linalg.norm(n)
                errors.append(np.max((paper-a)@n))
            fits.append(max(errors))
        assert min(fits) < 1e-8
    assert sum(mesh_volume(v, t) for v, t in collision_parts(kind)) == pytest.approx(volume)


def test_pen_has_distinct_end_caps_and_148mm_side_baseline():
    tags = {t["id"]: t for t in marker_layout("pen")}
    assert len(tags) == 10
    assert np.linalg.norm(np.array(tags[10]["center_m"])-tags[11]["center_m"]) == pytest.approx(.148)
    assert tags[18]["center_m"][1] == -.085 and tags[19]["center_m"][1] == .085


@pytest.mark.parametrize("kind", ["cube", "pen"])
def test_supported_resets_no_free_success_and_deployable_observation(kind):
    env = ManipulationEnv(TaskConfig(object=kind))
    try:
        for seed in range(20):
            obs, info = env.reset(seed=seed)
            assert env.observation_space.contains(obs)
            assert not info["dropped"] and not info["is_success"]
            assert info["angle_error_deg"] == pytest.approx(15)
            for _ in range(12):
                obs, reward, done, trunc, info = env.step(np.zeros(16))
                assert np.isfinite(obs).all() and np.isfinite(reward)
                assert not info["is_success"]
                if done or trunc:
                    break
        env.reset(seed=100)
        env.data.qpos[env.oq:env.oq+3] += [0, 0, .09]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, env.goal.ravel())
        env.data.qpos[env.oq+3:env.oq+7] = quat
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        assert not env.contact()
        assert not env.info()["stable"]
    finally:
        env.close()


@pytest.mark.parametrize("kind,dimensions", [("cube", [42, 42, 42]), ("pen", [24, 24, 170])])
def test_a1mini_metric_export_matches_simulation_and_fits_bed(tmp_path, kind, dimensions):
    import zipfile
    import xml.etree.ElementTree as ET
    from orca_feetech.objects import export_objects, print_coordinates
    export_objects(tmp_path)
    with zipfile.ZipFile(tmp_path / f"{kind}-a1mini.3mf") as z:
        root = ET.fromstring(z.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    assert root.get("unit") == "millimeter"
    points = np.array([[float(v.get(c)) for c in ("x", "y", "z")]
                       for v in root.findall(".//m:vertex", ns)])
    expected = print_coordinates(kind, geometry(kind)[0])
    assert np.allclose(points, expected)
    assert np.all(points >= -1e-8) and np.all(points <= 180+1e-8)
    assert np.allclose(points.max(0)-points.min(0), dimensions)
    assert points[:, 2].min() == pytest.approx(0)



def test_action_acceleration_and_pose_loss_hold_match_deployment():
    cfg = TaskConfig()
    codec = ActionCodec(cfg, ModelSpec.load().neutral)
    initial = codec.q.copy()
    q = codec.step(np.ones(16))
    assert np.max(np.abs(q-initial)) <= cfg.max_acceleration*DT**2 + 1e-8
    held = codec.step(-np.ones(16), valid=False)
    assert np.array_equal(held, q)
    observer = ObservationCodec(cfg)
    a = observer.push(q, q, np.zeros(3), np.eye(3), np.eye(3), np.zeros(3), True, 0)
    b = observer.push(q, q, np.zeros(3), np.eye(3), np.eye(3), np.zeros(3), False, .15)
    assert a.shape == b.shape == (220,)
    assert b[-2] == 0 and b[-1] == pytest.approx(.15)
    with pytest.raises(ValueError):
        codec.step(np.full(16, np.nan))


K = np.array([[1400., 0, 960], [0, 1400, 540], [0, 0, 1]])
R = np.array([2.8, .1, .1])
T = np.array([.015, .015, .28])


def projections(tags, noise=0, rng=None):
    result = []
    for tag in tags:
        points = cv2.projectPoints(np.array(tag["corners_m"]), R, T, K, None)[0]
        if noise:
            points += rng.normal(0, noise, points.shape)
        result.append(points)
    return result


def test_joint_two_end_pose_and_single_end_degradation():
    tags = [t for t in marker_layout("pen") if t["id"] in (12, 13)]
    corners = projections(tags)
    joint = solve_pose(tags, corners, np.array([12, 13]), K, None)
    single = solve_pose(tags, corners[:1], np.array([12]), K, None)
    assert joint is not None and single is not None
    truth = transform(R, T)
    assert np.linalg.norm(joint.T_CO-truth) < 1e-5
    assert solve_pose(tags, corners, np.array([12, 13]), K, None, min_pixels=1000) is None
    assert solve_pose(tags, [corners[0]+[[[0, 20]], [[20, 0]], [[0, -20]], [[-20, 0]]], corners[1]],
                      np.array([12, 13]), K, None) is None
    # Longer baseline improves the directed shaft axis under identical pixel noise.
    rng = np.random.default_rng(19)
    joint_errors, single_errors = [], []
    axis = truth[:3, 1]
    for _ in range(100):
        pixels = projections(tags, .25, rng)
        j = solve_pose(tags, pixels, np.array([12, 13]), K, None)
        s = solve_pose(tags, pixels[:1], np.array([12]), K, None)
        if j:
            joint_errors.append(np.arccos(np.clip(j.T_CO[:3, 1]@axis, -1, 1)))
        if s:
            single_errors.append(np.arccos(np.clip(s.T_CO[:3, 1]@axis, -1, 1)))
    assert len(joint_errors) >= 95 and 0 < len(single_errors) < len(joint_errors)
    assert np.median(joint_errors) < np.median(single_errors)


def test_real_dictionary_detection_of_projected_printed_tag():
    tag = next(t for t in marker_layout("cube") if t["id"] == 4)
    pixels = projections([tag])[0].reshape(4, 2).astype(np.float32)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    marker = cv2.aruco.generateImageMarker(dictionary, 4, 800)
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [799, 0], [799, 799], [0, 799]], np.float32), pixels)
    image = cv2.warpPerspective(marker, H, (1920, 1080), borderValue=255)
    tracker = TagTracker("cube", dict(K=K.tolist(), distortion=[0]*5, image_size=[1920, 1080]))
    pose, reference = tracker.detect(image)
    assert pose is not None and reference is None and pose.ids == [4]
    assert np.linalg.norm(pose.T_CO[:3, 3]-T) < .003
    assert rotation_error(pose.T_CO[:3, :3], transform(R, T)[:3, :3]) < .03


def test_checkpoint_roundtrip_and_reject_changed_geometry(tmp_path):
    """Validate save/load and inference without training a policy on this Mac."""
    import json
    from dataclasses import asdict
    from stable_baselines3 import PPO
    from orca_feetech.checkpoint import load_config
    from orca_feetech.manipulation import TASK_ID, HISTORY, FRAME_SIZE
    from orca_feetech.model import JOINT_NAMES
    from orca_feetech.paths import provenance
    config = TaskConfig(object="pen")
    env = ManipulationEnv(config)
    try:
        obs, _ = env.reset(seed=33)
        model = PPO("MlpPolicy", env, device="cpu", n_steps=16, batch_size=16,
                    policy_kwargs={"net_arch": [16]}, seed=1)
        action = model.predict(obs, deterministic=True)[0]
        model.save(tmp_path / "policy.zip")
        meta = dict(task_id=TASK_ID, task=asdict(config), fingerprint=config.fingerprint(),
                    joint_names=JOINT_NAMES, history=HISTORY, frame_size=FRAME_SIZE, upstream=provenance())
        (tmp_path / "config.json").write_text(json.dumps(meta))
        loaded_config, _ = load_config(tmp_path / "policy.zip")
        assert loaded_config.object == "pen"
        loaded = PPO.load(tmp_path / "policy.zip", device="cpu")
        assert np.array_equal(action, loaded.predict(obs, deterministic=True)[0])
        meta["task"]["object"] = "cube"
        (tmp_path / "config.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="mismatch"):
            load_config(tmp_path / "policy.zip")
    finally:
        env.close()
