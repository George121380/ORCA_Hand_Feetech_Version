"""Manufacturing meshes; the assembled outer surface stays in objects.py.

Optional manifold3d is used only to generate the split printing variant.
All coordinates in this module are millimeters.
"""
from __future__ import annotations

import numpy as np

from .objects import geometry

SOCKET_DIAMETER_MM = 4.3
SOCKET_DEPTH_MM = 3.15
PIN_DIAMETER_MM = 4.
PIN_LENGTH_MM = 6.
SOCKET_Y_MM = (-74., 0., 74.)


def split_pen():
    """Two identical Z>=0 halves with three blind sockets opening at the seam."""
    import manifold3d as m
    vertices, faces = geometry("pen")
    solid = m.Manifold(m.Mesh(np.asarray(vertices*1000, dtype=np.float32),
                             np.asarray(faces, dtype=np.uint32)))
    half, _ = solid.split_by_plane([0, 0, 1], 0)
    for y in SOCKET_Y_MM:
        hole = m.Manifold.cylinder(SOCKET_DEPTH_MM+.1, SOCKET_DIAMETER_MM/2,
                                   circular_segments=48).translate([0, y, -.1])
        half = half - hole
    pin = m.Manifold.cylinder(PIN_LENGTH_MM, PIN_DIAMETER_MM/2, circular_segments=48)
    if half.status() != m.Error.NoError or pin.status() != m.Error.NoError:
        raise ValueError("Split pen boolean failed")
    def mesh(manifold):
        result = manifold.to_mesh()
        return np.asarray(result.vert_properties[:, :3], dtype=float), np.asarray(result.tri_verts)
    return mesh(half), mesh(pin)


def split_pen_plate():
    """Flat seam on bed: two 170x24x12 halves + three pins on one A1 mini plate.

    Parallel placement uses less space than two diagonal parts. The 170 mm
    length leaves room for a 3 mm brim: Y extent 2..178 on the 180 mm bed.
    """
    (half_v, half_f), (pin_v, pin_f) = split_pen()
    parts = [(half_v + [62, 90, 0], half_f), (half_v + [118, 90, 0], half_f)]
    parts += [(pin_v + [90, y, 0], pin_f) for y in (55, 70, 85)]
    vertices, triangles, count = [], [], 0
    for v, f in parts:
        vertices.append(v)
        triangles.append(f+count)
        count += len(v)
    return np.concatenate(vertices), np.concatenate(triangles)


def export_split_pen(output):
    from pathlib import Path
    from .objects import export_3mf
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vertices, faces = split_pen_plate()
    export_3mf(output / "pen-split-a1mini.3mf", vertices, faces,
               "ORCA pen / two flat halves and three alignment pins / mm")
    for name, (v, f) in zip(("pen-half", "pen-alignment-pin"), split_pen(), strict=True):
        lines = [f"solid {name}_mm"]
        for face in f:
            p = v[face]
            normal = np.cross(p[1]-p[0], p[2]-p[0])
            normal /= np.linalg.norm(normal)
            lines += ["facet normal " + " ".join(map(str, normal)), "outer loop"]
            lines += ["vertex " + " ".join(map(str, point)) for point in p]
            lines += ["endloop", "endfacet"]
        lines.append(f"endsolid {name}_mm")
        (output/f"{name}.stl").write_text("\n".join(lines)+"\n")
