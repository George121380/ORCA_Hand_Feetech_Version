"""Render comparable orthographic views without changing task geometry or state.

Run with the sim extras installed. Images share a 250 mm vertical field of view;
the ruler is therefore metric, independent of object height. Contact counts are
initial support checks, never evidence of a trained grasp.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from orca_feetech.manipulation import ManipulationEnv, TaskConfig
from orca_feetech.objects import specification

OUTPUT = Path(__file__).resolve().parents[1] / "docs/sim2real"
WIDTH, HEIGHT = 960, 720
FOV_M = .25
CENTER = np.array([.18, .05])
PX_PER_M = HEIGHT / FOV_M


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def project(points):
    points = np.asarray(points)
    return np.column_stack((WIDTH / 2 + (points[:, 0] - CENTER[0]) * PX_PER_M,
                            HEIGHT / 2 - (points[:, 1] - CENTER[1]) * PX_PER_M))


def label(draw, point, text, size=22):
    point = np.asarray(point, dtype=float)
    bounds = draw.textbbox(tuple(point), text, font=font(size), anchor="mm")
    draw.rounded_rectangle((bounds[0]-9, bounds[1]-6, bounds[2]+9, bounds[3]+6),
                           radius=4, fill="#20342f")
    draw.text(tuple(point), text, font=font(size), fill="#ffffff", anchor="mm")


def dimension(draw, endpoints, text, offset):
    p, q = project(endpoints)
    direction = q - p
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    a, b = p + normal * offset, q + normal * offset
    color = "#f7dfaa"
    for start, end in ((p, a+normal*7), (q, b+normal*7), (a, b)):
        draw.line([tuple(start), tuple(end)], fill=color, width=3)
    for tip, inward in ((a, direction), (b, -direction)):
        draw.polygon([tuple(tip), tuple(tip+inward*12+normal*5),
                      tuple(tip+inward*12-normal*5)], fill=color)
    label(draw, (a+b)/2 + normal*23, text)


def world_mesh(env, geom):
    model, data = env.model, env.data
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start+count]
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def render(kind):
    env = ManipulationEnv(TaskConfig(object=kind, randomization=0))
    try:
        env.reset(seed=0)
        env.model.vis.global_.offwidth = WIDTH
        env.model.vis.global_.offheight = HEIGHT
        env.model.vis.global_.orthographic = 1
        env.model.vis.global_.fovy = FOV_M
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, camera)
        camera.lookat[:] = [*CENTER, .17]
        camera.distance, camera.azimuth, camera.elevation = .4, 90, -90
        with mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH) as renderer:
            renderer.update_scene(env.data, camera=camera)
            assert renderer.scene.camera[0].orthographic
            image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 63), fill="#20342f")
        draw.text((24, 18), f"{kind.upper()} / ORTHOGRAPHIC / SAME SCALE",
                  fill="white", font=font(23))
        position, rotation = env.pose()
        if kind == "cube":
            half = np.array(specification(kind)["dimensions_m"])[0] / 2
            ends = np.array([[-half, -half, half], [half, -half, half]])
            dimension(draw, ends @ rotation.T + position, f"{2*half*1000:g} mm", 30)
        else:
            ends = np.array([[0, -.085, 0], [0, .085, 0]])
            dimension(draw, ends @ rotation.T + position, "170 mm (3D)", 65)
            label(draw, [720, 632], "Shaft diameter 16 mm", size=21)
        # At 720 px / 250 mm, this ruler is exactly 144 px = 50 mm.
        ruler = 50 * PX_PER_M / 1000
        x, y = 55, 652
        draw.rectangle((x-20, y-40, x+ruler+20, y+30), fill="#20342f")
        draw.line([(x, y), (x+ruler, y)], fill="white", width=4)
        for xx in (x, x+ruler):
            draw.line([(xx, y-8), (xx, y+8)], fill="white", width=3)
        draw.text((x+ruler/2, y-21), "50 mm", fill="white", font=font(20), anchor="mm")
        image.save(OUTPUT / f"scene-{kind}-scale.png")
        palm = np.concatenate([world_mesh(env, g) for g in range(env.model.ngeom)
                               if env.model.body(env.model.geom_bodyid[g]).name == "right_palm"
                               and env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
        contacts = set()
        for c in env.data.contact:
            if c.dist <= 0 and any(g in (c.geom1, c.geom2) for g in env.geoms):
                other = c.geom2 if c.geom1 in env.geoms else c.geom1
                contacts.add(env.model.body(env.model.geom_bodyid[other]).name)
        return dict(dimensions_mm=(np.array(specification(kind)["dimensions_m"])*1000).tolist(),
                    palm_world_axis_bounds_mm=(np.ptp(palm, axis=0)*1000).tolist(),
                    initial_contact_bodies=sorted(contacts),
                    projected_pen_axis_mm=(170*np.linalg.norm(rotation[:2, 1])
                                           if kind == "pen" else None))
    finally:
        env.close()


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = dict(seed=0, randomization=0, camera="orthographic overhead",
                  vertical_field_mm=250, pixels_per_mm=PX_PER_M/1000,
                  note="Mesh bounds are simulated structure dimensions, not physical measurements. "
                       "Only initial palm support is checked; no learned grasp is shown.",
                  objects={kind: render(kind) for kind in ("cube", "pen")})
    (OUTPUT / "scale-check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
