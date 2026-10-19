"""Reproducible geometry review. No learning or hardware commands are run.

Snapshots intentionally include intersecting solids to probe interference;
counts are not collision probabilities, force-closure tests or policy scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from orca_feetech.manipulation import build_model, TaskConfig
from orca_feetech.model import ModelSpec, JOINT_NAMES, poses
from orca_feetech.objects import specification, geometry, pen_dimensions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"docs/sim2real"
LENGTHS = (170, 190, 200, 210, 220, 230)
CAMERA = np.array([.25, -.10, .48])
CAMERA_TARGET = np.array([.185, .05, .19])


def visible_ends(model, data, spec, body):
    """Ideal geometric visibility only: front face, pixels, frustum and ray tests."""
    position, rotation = data.xpos[body], data.xmat[body].reshape(3, 3)
    forward = CAMERA_TARGET-CAMERA
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.stack([right, -up, forward])
    ends = set()
    for tag in spec["tags"]:
        normal = rotation@np.cross(tag["u"], tag["v"])
        points = np.array(tag["corners_m"])@rotation.T+position+normal*.0002
        to_camera = CAMERA-points.mean(0)
        if np.dot(normal, to_camera)/np.linalg.norm(to_camera) < .2:
            continue
        local = (points-CAMERA)@view.T
        if np.any(local[:, 2] <= 0):
            continue
        pixels = local[:, :2]/local[:, 2:]*1400+[960, 540]
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= 1920) or np.any(pixels[:, 1] >= 1080)
                or np.linalg.norm(np.roll(pixels, -1, axis=0)-pixels, axis=1).min() < 40):
            continue
        clear = True
        for point in points:
            vector = point-CAMERA
            distance = np.linalg.norm(vector)
            hit = mujoco.mj_ray(model, data, CAMERA, vector/distance, None, True, -1, None)
            if 0 <= hit < distance-.0001:
                clear = False
                break
        if clear:
            ends.add(tag["id"] % 2)
    return len(ends)


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_bodies(model, data, geoms):
    hits = {"heads": False, "grip_and_shoulders": False, "tower": False}
    for contact in data.contact:
        if contact.dist >= -.0001 or not any(g in (contact.geom1, contact.geom2) for g in geoms):
            continue
        obj = contact.geom1 if contact.geom1 in geoms else contact.geom2
        other = contact.geom2 if contact.geom1 in geoms else contact.geom1
        name = model.body(model.geom_bodyid[other]).name or ""
        if name == "right_tower":
            hits["tower"] = True
        elif name.startswith("right_") and name != "right_palm":
            hits["heads" if obj in (geoms[0], geoms[-1]) else "grip_and_shoulders"] = True
    return hits


def review(length):
    spec = specification("pen", pen_length_m=length/1000)
    model = build_model(TaskConfig(object="pen", randomization=0), object_spec=spec)
    data = mujoco.MjData(model)
    qadr = np.array([model.joint("right_"+n).qposadr[0] for n in JOINT_NAMES])
    actuators = [model.actuator("right_"+n+"_actuator").id for n in JOINT_NAMES]
    oq = model.joint("object_free").qposadr[0]
    geoms = [model.geom(f"object_collision_{i}").id for i in range(5)]
    neutral = ModelSpec.load().neutral
    named = poses()
    hand_poses = dict(open=neutral, half_close=neutral+.5*(named["fist"]-neutral),
                      pinch=named["pinch"], fist=named["fist"])
    body = model.body("task_object").id
    inertia_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(inertia_rotation, model.body_iquat[body])
    rotation = inertia_rotation.reshape(3, 3)
    yaw_inertia = float((rotation@np.diag(model.body_inertia[body])@rotation.T)[2, 2])
    centers = []
    for x in (.135, .170, .190, .210):
        counts = []
        for name, q in hand_poses.items():
            count = dict(pose=name, snapshots=0, heads=0, grip_and_shoulders=0, tower=0,
                         any_end_geometrically_visible=0, both_ends_geometrically_visible=0)
            data.qpos[qadr] = q
            for z in (.180, .200, .220):
                for angle in np.linspace(0, np.pi, 37):
                    data.qpos[oq:oq+3] = [x, .05, z]
                    data.qpos[oq+3:oq+7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
                    mujoco.mj_forward(model, data)
                    for key, value in contact_bodies(model, data, geoms).items():
                        count[key] += int(value)
                    if x == .190:
                        ends = visible_ends(model, data, spec, body)
                        count["any_end_geometrically_visible"] += int(ends >= 1)
                        count["both_ends_geometrically_visible"] += int(ends == 2)
                    count["snapshots"] += 1
            counts.append(count)
        centers.append(dict(center_x_m=x, center_y_m=.05, poses=counts,
                            **{key: sum(c[key] for c in counts) for key in
                               ("snapshots", "heads", "grip_and_shoulders", "tower",
                                "any_end_geometrically_visible", "both_ends_geometrically_visible")}))
    # Controlled pictures: exact same hand pose, object pose and orthographic camera.
    # The pen is positioned for inspection, not physically held by an invisible weld.
    data.qpos[qadr] = neutral
    data.qpos[oq:oq+3] = [.190, .049, .190]
    data.qpos[oq+3:oq+7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    model.vis.global_.orthographic, model.vis.global_.fovy = 1, .28
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:], camera.distance = [.185, .05, .18], .4
    camera.azimuth, camera.elevation = 90, -90
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 64), fill="#20342f")
    draw.text((22, 17), f"{length} mm / STATIC SIZE REVIEW / SAME SCALE", font=font(24), fill="white")
    draw.rectangle((20, 645, 260, 710), fill="#20342f")
    ruler = 50*720/280
    draw.line([(40, 683), (40+ruler, 683)], fill="white", width=4)
    draw.text((43, 650), "50 mm", font=font(20), fill="white")
    image.save(OUT/f"pen-size-{length}.png")
    # Independent gravity-settle check from the forward hand placement.
    supported = 0
    positions = []
    tower_contacts = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(model, data)
        data.qpos[qadr], data.ctrl[actuators] = neutral, neutral
        data.qpos[oq:oq+3] = [.190+rng.uniform(-.003, .003), .049+rng.uniform(-.003, .003), .192]
        angle = rng.uniform(-.08, .08)
        data.qpos[oq+3:oq+7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data, nstep=200)
        mujoco.mj_forward(model, data)
        pos = data.xpos[body].copy()
        positions.append(pos)
        bodies = set()
        for contact in data.contact:
            if contact.dist <= 0 and any(g in (contact.geom1, contact.geom2) for g in geoms):
                other = contact.geom2 if contact.geom1 in geoms else contact.geom1
                bodies.add(model.body(model.geom_bodyid[other]).name or "")
        good = any(name.startswith("right_") and name != "right_tower" for name in bodies)
        supported += int(good and pos[2] > .12 and np.linalg.norm(pos[:2]-[.190, .049]) < .07)
        tower_contacts += int("right_tower" in bodies)
    v, _ = geometry("pen", pen_length_m=length/1000)
    c = np.sqrt(.5)
    diagonal = v*1000 @ np.array([[c, c, 0], [-c, c, 0], [0, 0, 1]])
    span = np.ptp(diagonal, axis=0)
    return dict(length_mm=length, shaft_mm=length-64, tag_baseline_mm=length-22,
                nominal_mass_g=spec["nominal_mass_kg"]*1000,
                uniform_solid_yaw_inertia_kg_m2=yaw_inertia,
                half_plate_xy_mm=span[:2].tolist(),
                edge_margin_with_3mm_brim_and_01gap_mm=float((180-max(span[:2]))/2-3.1),
                centers=centers, settle=dict(trials=40, supported=supported,
                                           tower_contacts=tower_contacts,
                                           mean_position_m=np.mean(positions, axis=0).tolist()),
                dimensions=pen_dimensions(length/1000))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(method="static intersection sweep + independent gravity-settle checks",
                  warning="Not policy success, force closure, camera occlusion probability or hardware validation. "
                          "Controlled images position the pen without time evolution. "
                          "Inertia is a uniform-solid approximation, not measured print inertia.",
                  fixed_parameters=dict(shaft_diameter_mm=16, head_width_mm=24, head_length_mm=22,
                                        shoulder_length_mm=10, penetration_threshold_mm=.1,
                                        yaw_degrees=[0, 180, 5], heights_mm=[180, 200, 220],
                                        hand_poses=["open", "half_close", "pinch", "fist"]),
                  selected_length_mm=210, selection="Engineering starting candidate, not an identified optimum.",
                  visibility_camera=dict(position_m=CAMERA.tolist(), target_m=CAMERA_TARGET.tolist(),
                                         image_size=[1920, 1080], focal_px=1400, min_edge_px=40,
                                         evaluated_center_x_m=.190,
                                         note="Illustrative calibrated pinhole camera, not a measured real camera. "
                                              "All four corners must be in frame and ray-unoccluded. "
                                              "No AprilTag detector, blur, noise or illumination is simulated."),
                  candidates=[review(length) for length in LENGTHS])
    baseline = report["candidates"][0]["uniform_solid_yaw_inertia_kg_m2"]
    for candidate in report["candidates"]:
        candidate["yaw_inertia_relative_to_170"] = candidate["uniform_solid_yaw_inertia_kg_m2"]/baseline
    (OUT/"pen-size-review.json").write_text(json.dumps(report, indent=2)+"\n")
    for r in report["candidates"]:
        counts = next(x for x in r["centers"] if x["center_x_m"] == .19)
        print(r["length_mm"], "head intersections", counts["heads"], "/", counts["snapshots"],
              "inertia ratio", round(r["yaw_inertia_relative_to_170"], 2), "settle", r["settle"], flush=True)
