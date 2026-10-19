"""One metric definition for printable solids, MuJoCo collision and fiducials."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBJECT_VERSION = "printables-v3"
DICTIONARY = "DICT_APRILTAG_36h11"


def _box(half, center=(0, 0, 0)):
    vertices = np.array([[x, y, z] for x in (-half[0], half[0])
                         for y in (-half[1], half[1]) for z in (-half[2], half[2])]) + center
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return vertices, np.array([[f[0], f[i], f[i+1]] for f in faces for i in (1, 2)])


def _rings_mesh(rings):
    """Outward surface for coaxial convex rings, including smooth tapered shoulders."""
    count = len(rings[0])
    vertices = np.concatenate(rings)
    faces = []
    for layer in range(len(rings)-1):
        for i in range(count):
            j = (i+1) % count
            # Ring order is clockwise looking from +Y; reversing the axial strip
            # makes its normals point away from the axis.
            face = [layer*count+i, (layer+1)*count+i, (layer+1)*count+j, layer*count+j]
            faces += [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
    vertices = np.vstack([vertices, rings[0].mean(0), rings[-1].mean(0)])
    for i in range(count):
        j = (i+1) % count
        faces += [[len(vertices)-2, i, j],
                  [len(vertices)-1, (len(rings)-1)*count+j, (len(rings)-1)*count+i]]
    return vertices, np.array(faces)


def _pen_rings():
    theta = np.arange(64)*2*np.pi/64
    rays = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    square = rays * (.012 / np.abs(rays).max(axis=1))[:, None]
    circle = rays * .008
    return [np.array([[x, y, z] for x, z in cross]) for y, cross in
            ((-.085, square), (-.063, square), (-.053, circle),
             (.053, circle), (.063, square), (.085, square))]


def geometry(kind):
    if kind == "cube":
        return _box(np.array([.0275, .0275, .0275]))
    if kind == "pen":
        vertices, faces, lookup = [], [], {}
        for points, triangles in collision_parts("pen"):
            for tri in triangles:
                p = points[tri]
                if any(np.all(np.abs(p[:, 1]-y) < 1e-10) for y in (-.063, -.053, .053, .063)):
                    continue  # Remove only the shared internal end caps.
                face = []
                for point in p:
                    key = tuple(np.round(point, 12))
                    if key not in lookup:
                        lookup[key] = len(vertices)
                        vertices.append(point)
                    face.append(lookup[key])
                faces.append(face)
        return np.asarray(vertices), np.asarray(faces)
    raise ValueError("Object must be cube or pen")


def collision_parts(kind):
    """Convex decomposition, never the convex hull of the entire nonconvex pen."""
    if kind == "cube":
        return [geometry("cube")]
    from scipy.spatial import ConvexHull
    rings = _pen_rings()

    def shoulder(a, b):
        points = np.concatenate([a, b])
        hull = ConvexHull(points)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            p = points[face]
            if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), hull.equations[i, :3]) < 0:
                faces[i] = face[::-1]
        return points, faces

    # Exact convex pieces meet without volume overlap. The complete STL is their
    # exterior union, so MuJoCo's convexification cannot fill the thin shaft gap.
    # A square has only four true corners; remove collinear edge vertices before
    # joining its shoulder, so the exported union has no STL T-junctions.
    square_indices = [8, 24, 40, 56]
    return [_box(np.array([.012, .011, .012]), np.array([0, -.074, 0])),
            shoulder(rings[1][square_indices], rings[2]), _rings_mesh(rings[2:4]),
            shoulder(rings[3], rings[4][square_indices]),
            _box(np.array([.012, .011, .012]), np.array([0, .074, 0]))]


def mesh_volume(vertices, triangles):
    p = vertices[triangles]
    return float(np.sum(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2]))) / 6)



def marker_layout(kind):
    """Corners are OpenCV decoded TL, TR, BR, BL; all coordinates in meters.

    u points to printed right, v to printed top, u cross v is outward.
    Arrow above each label on the SVG points along v on the physical object.
    """
    tags = []
    if kind == "cube":
        frames = [([1, 0, 0], [0, 1, 0]), ([-1, 0, 0], [0, -1, 0]),
                  ([0, 1, 0], [-1, 0, 0]), ([0, -1, 0], [1, 0, 0]),
                  ([0, 0, 1], [1, 0, 0]), ([0, 0, -1], [-1, 0, 0])]
        for ident, (normal, u) in enumerate(frames):
            n, u = np.array(normal), np.array(u)
            tags.append((ident, n * .0275, u, np.cross(n, u), .032, .040))
    elif kind == "pen":
        # Four side faces on each square end, plus the two outward end caps.
        frames = [([1, 0, 0], [0, 0, -1]), ([0, 0, 1], [1, 0, 0]),
                  ([-1, 0, 0], [0, 0, 1]), ([0, 0, -1], [-1, 0, 0])]
        for face, (normal, u) in enumerate(frames):
            n, u = np.array(normal), np.array(u)
            for end, y in enumerate((-.074, .074)):
                tags.append((10+face*2+end, n*.012+[0, y, 0], u,
                             np.array([0, 1, 0]), .016, .020))
        tags += [(18, np.array([0, -.085, 0]), np.array([1, 0, 0]),
                  np.array([0, 0, 1]), .016, .020),
                 (19, np.array([0, .085, 0]), np.array([-1, 0, 0]),
                  np.array([0, 0, 1]), .016, .020)]

    else:
        raise ValueError("Object must be cube or pen")
    result = []
    for ident, center, u, v, size, paper in tags:
        corners = [center + size / 2 * (a * u + b * v)
                   for a, b in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        result.append(dict(id=ident, center_m=center.tolist(), u=u.tolist(), v=v.tolist(),
                           black_size_m=size, paper_size_m=paper,
                           corners_m=np.asarray(corners).tolist()))
    return result


def specification(kind):
    vertices, triangles = geometry(kind)
    return dict(version=OBJECT_VERSION, kind=kind, units="m", dictionary=DICTIONARY,
                nominal_mass_kg=.055 if kind == "cube" else .035,
                collision_parts=[dict(vertices_m=v.tolist(), triangles=t.tolist(), volume_m3=mesh_volume(v, t))
                                 for v, t in collision_parts(kind)],
                dimensions_m=(vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
                vertices_m=vertices.tolist(), triangles=triangles.tolist(), tags=marker_layout(kind),
                notes="Solid watertight shell; slicer sets infill. Weigh the finished tagged print. "
                      "No tag recess: white flat adhesive zones; sticker thickness is not in CAD.")


def geometry_signature(kind):
    """Portable hash input: ignore coplanar hull diagonals and floating point dust.

    Convex parts are defined uniquely by their vertex sets. QHull's triangulation
    order and summation rounding can differ between the Linux trainer and Mac.
    """
    def rounded(value):
        a = np.round(np.asarray(value, dtype=float), 10)
        a[np.abs(a) < 1e-10] = 0
        return a.tolist()
    parts = [sorted(map(tuple, rounded(v))) for v, _ in collision_parts(kind)]
    tags = [{key: rounded(value) if isinstance(value, list) else value
             for key, value in tag.items()} for tag in marker_layout(kind)]
    return dict(version=OBJECT_VERSION, dictionary=DICTIONARY, kind=kind,
                convex_parts=parts, tags=tags,
                nominal_mass_kg=.055 if kind == "cube" else .035)


def print_coordinates(kind, vertices):
    """A1 mini plates use millimeters, bed center (90,90), first layer z=0."""
    points = np.asarray(vertices)*1000
    if kind == "pen":
        points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    points[:, :2] += [90, 90]
    points[:, 2] -= points[:, 2].min()
    if np.any(points < -1e-8) or np.any(points > 180+1e-8):
        raise ValueError("Model exceeds A1 mini's 180 mm build volume")
    return points


def export_3mf(path, vertices, triangles, title):
    """Portable, metric 3MF model; printer/process/filament chosen in the slicer."""
    import zipfile
    import xml.etree.ElementTree as ET
    namespace = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ET.register_namespace("", namespace)
    def element(tag_name, **attrs):
        return ET.Element("{"+namespace+"}"+tag_name, attrs)
    root = element("model", unit="millimeter")
    metadata = element("metadata", name="Title")
    metadata.text = title
    root.append(metadata)
    resources = element("resources")
    root.append(resources)
    obj = element("object", id="1", type="model", name=title)
    resources.append(obj)
    mesh = element("mesh")
    obj.append(mesh)
    vs, fs = element("vertices"), element("triangles")
    mesh.extend([vs, fs])
    for p in vertices:
        vs.append(element("vertex", **dict(zip(("x", "y", "z"), (f"{v:.9f}" for v in p), strict=True))))
    for t in triangles:
        fs.append(element("triangle", **dict(zip(("v1", "v2", "v3"), map(str, t), strict=True))))
    build = element("build")
    build.append(element("item", objectid="1"))
    root.append(build)
    content_types = '<?xml version="1.0" encoding="UTF-8"?>' \
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' \
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' \
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>'
    relationships = '<?xml version="1.0" encoding="UTF-8"?>' \
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' \
        '<Relationship Id="rel0" Target="/3D/3dmodel.model" ' \
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", relationships)
        z.writestr("3D/3dmodel.model", ET.tostring(root, encoding="utf-8", xml_declaration=True))



def export_objects(output):
    import cv2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    all_tags = []
    for kind in ("cube", "pen"):
        spec = specification(kind)
        (output / f"{kind}.json").write_text(json.dumps(spec, indent=2) + "\n")
        vertices, triangles = geometry(kind)
        print_vertices = print_coordinates(kind, vertices)
        export_3mf(output / f"{kind}-a1mini.3mf", print_vertices, triangles, f"ORCA {kind} A1 mini 0.4mm")
        for name, points in ((kind, vertices*1000), (f"{kind}-a1mini", print_vertices)):
            lines = [f"solid {name}_millimeters"]
            for tri in triangles:
                p = points[tri]
                normal = np.cross(p[1]-p[0], p[2]-p[0])
                normal /= np.linalg.norm(normal)
                lines += ["facet normal " + " ".join(map(str, normal)), "outer loop"]
                lines += ["vertex " + " ".join(map(str, point)) for point in p]
                lines += ["endloop", "endfacet"]
            lines += [f"endsolid {name}_millimeters"]
            (output / f"{name}.stl").write_text("\n".join(lines)+"\n")
        # OpenSCAD source is provided as a second editable CAD representation.
        (output / f"{kind}.scad").write_text(
            "// Millimeters. Same vertices as STL and simulation.\npolyhedron(points="
            + json.dumps((vertices * 1000).tolist()) + ", faces="
            + json.dumps(triangles[:, ::-1].tolist()) + ", convexity=10);\n")
        all_tags += [(kind, t) for t in spec["tags"]]
    # A fixed world reference: center is the origin of W; +x right, +y toward
    # printed arrow, +z out of the board. Place it beside the palm, never on it.
    all_tags += [("world", dict(id=40, black_size_m=.050, paper_size_m=.066))]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
           '<rect width="210" height="297" fill="white"/>',
           '<text x="10" y="10" font-size="4">AprilTag 36h11 / print 100% / no fit-to-page</text>']
    for index, (kind, tag) in enumerate(all_tags):
        # Last reference tag has its own larger region.
        if kind == "cube":
            x, y = 12 + (index % 3) * 60, 24 + (index // 3) * 50
        elif kind == "pen":
            pen_index = index - 6
            x, y = 12 + (pen_index % 5) * 37, 138 + (pen_index // 5) * 35
        else:
            x, y = 125, 210
        paper, size = tag["paper_size_m"] * 1000, tag["black_size_m"] * 1000
        image = cv2.aruco.generateImageMarker(dictionary, tag["id"], 8)
        svg.append(f'<text x="{x}" y="{y-3}" font-size="2.8">{kind} ID {tag["id"]} / top +v ↑</text>')
        svg.append(f'<rect x="{x}" y="{y}" width="{paper}" height="{paper}" fill="white" stroke="#ccc" stroke-width=".1"/>')
        offset = (paper - size) / 2
        for iy, ix in zip(*np.where(image == 0)):
            svg.append(f'<rect x="{x+offset+ix*size/8}" y="{y+offset+iy*size/8}" '
                       f'width="{size/8}" height="{size/8}" fill="black"/>')
    svg += ['<path d="M 12 284 h 100 m -100 -2 v 4 m 100 -4 v 4" stroke="black" stroke-width=".2"/>',
            '<text x="12" y="280" font-size="3">Ruler must measure 100 mm after printing</text>', '</svg>']
    (output / "tags-a4.svg").write_text("\n".join(svg))
