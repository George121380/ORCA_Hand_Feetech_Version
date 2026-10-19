"""Manufacturing geometry and the actual printable PDF, not just source data."""
import importlib.util
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import cv2
import manifold3d as m
import numpy as np
import pypdfium2
import pytest
from pypdf import PdfReader

from orca_feetech.objects import DICTIONARY, geometry, marker_layout
from orca_feetech.printing import split_pen, split_pen_plate


def manifold(v, f):
    return m.Manifold(m.Mesh(np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.uint32)))


def test_split_pen_matches_assembled_outer_shape_and_plate_with_brim():
    (v, f), (pv, pf) = split_pen()
    half = manifold(v, f)
    assert half.status() == m.Error.NoError
    assert np.allclose(np.ptp(v, axis=0), [24, 210, 12])
    assert np.allclose(np.ptp(pv, axis=0), [4, 4, 6])
    assembled = half + half.rotate([0, 180, 0])
    for y in (-94, 0, 94):
        assembled = assembled + manifold(pv, pf).translate([0, y, -3])
    outer_v, outer_f = geometry("pen")
    outer = manifold(outer_v*1000, outer_f)
    assert (assembled-outer).volume() < .001
    # Only the enclosed socket clearance is missing; external CAD is unchanged.
    assert 40 < (outer-assembled).volume() < 60
    plate, faces = split_pen_plate()
    assert manifold(plate, faces).status() == m.Error.NoError
    assert np.all(plate[:, :2].min(0)-3.1 > 0)
    assert np.all(plate[:, :2].max(0)+3.1 < 180)
    assert plate[:, 2].min() == pytest.approx(0)
    assert plate[:, 2].max() == pytest.approx(12)


def test_size_review_changes_only_shaft_length_and_tag_positions():
    from orca_feetech.objects import mesh_volume, specification
    short = specification("pen", pen_length_m=.170)
    long = specification("pen", pen_length_m=.220)
    # A length change extrudes only the cylindrical shaft; heads/shoulders stay fixed.
    area = 64/2*.008**2*np.sin(2*np.pi/64)
    difference = (mesh_volume(np.array(long["vertices_m"]), np.array(long["triangles"]))
                  - mesh_volume(np.array(short["vertices_m"]), np.array(short["triangles"])))
    assert difference == pytest.approx(.050*area)
    for a, b in zip(short["tags"], long["tags"], strict=True):
        assert a["id"] == b["id"] and a["black_size_m"] == b["black_size_m"]
        assert abs(b["center_m"][1]-a["center_m"][1]) == pytest.approx(.025)


def test_pdf_tags_decode_at_exact_scale_and_calibration_target_matches_cli(tmp_path):
    path = Path(__file__).parents[1]/"scripts/prepare_print_kit.py"
    spec = importlib.util.spec_from_file_location("print_kit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.generate(tmp_path)
    file = tmp_path/"print-kit-a4.pdf"
    reader = PdfReader(file)
    assert len(reader.pages) == 6
    for i, page in enumerate(reader.pages):
        dims = [float(page.mediabox.width)*25.4/72, float(page.mediabox.height)*25.4/72]
        assert np.allclose(dims, [297, 210] if i == 3 else [210, 297], atol=.001)
    dpi = 300
    document = pypdfium2.PdfDocument(file)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    detector = cv2.aruco.ArucoDetector(dictionary)
    layout = {t["id"]: t for kind in ("cube", "pen") for t in marker_layout(kind)}
    for page in (1, 2, 3):
        image = document[page-1].render(scale=dpi/72).to_numpy()
        corners, ids, _ = detector.detectMarkers(image)
        expected = {t["id"]: t for t in manifest["tags"] if t["page"] == page}
        assert set(ids.ravel()) == set(expected)
        for ident, corner in zip(ids.ravel(), corners, strict=True):
            tag = expected[int(ident)]
            measured_mm = np.linalg.norm(corner[0, 1]-corner[0, 0])*25.4/dpi
            assert abs(measured_mm-tag["black_size_mm"]) < .15
            # Decoded TL is printed at the top left; rotations/mirroring fail.
            assert corner[0, 0, 0] < corner[0, 1, 0]
            assert corner[0, 0, 1] < corner[0, 3, 1]
            if int(ident) in layout:
                assert tag["black_size_mm"] == pytest.approx(layout[int(ident)]["black_size_m"]*1000)
                assert tag["paper_size_mm"] == pytest.approx(layout[int(ident)]["paper_size_m"]*1000)
    board = document[3].render(scale=150/72).to_numpy()
    found, _ = cv2.findChessboardCornersSB(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), (9, 6))
    assert found


@pytest.mark.parametrize("name", ["cube", "pen-split"])
def test_delivered_native_projects_contain_correct_plate_geometry_and_process(name):
    path = Path(__file__).parents[1]/"assets/objects/v4"/f"{name}-a1mini-pla.3mf"
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    with zipfile.ZipFile(path) as z:
        model = ET.fromstring(z.read("3D/Objects/object_1.model"))
        build = ET.fromstring(z.read("3D/3dmodel.model"))
        settings = json.loads(z.read("Metadata/project_settings.config"))
    points = np.array([[float(v.get(key)) for key in ("x", "y", "z")]
                       for v in model.findall(".//m:vertex", ns)])
    triangles = np.array([[int(v.get(key)) for key in ("v1", "v2", "v3")]
                          for v in model.findall(".//m:triangle", ns)])
    transform = np.array(list(map(float, build.find("m:build/m:item", ns).get("transform").split())))
    points = points@transform[:9].reshape(3, 3)+transform[9:]
    if name == "pen-split":
        expected, expected_faces = split_pen_plate()
        # QHull/boolean triangulation and vertex order can vary across OSes.
        # Compare the solids, including the socket cavities, not array order.
        delivered, canonical = manifold(points, triangles), manifold(expected, expected_faces)
        assert delivered.status() == canonical.status() == m.Error.NoError
        assert (delivered-canonical).volume() < .01
        assert (canonical-delivered).volume() < .01
    else:
        from orca_feetech.objects import print_coordinates
        expected = print_coordinates("cube", geometry("cube")[0])
        assert np.allclose(points, expected)
    assert np.allclose(points.min(0), expected.min(0))
    assert np.allclose(points.max(0), expected.max(0))
    def value(key):
        v = settings[key]
        return v[0] if isinstance(v, list) else v
    assert value("wall_loops") == ("4" if name == "pen-split" else "3")
    assert value("sparse_infill_density") == "20%"
    assert value("sparse_infill_pattern") == "gyroid"
    assert value("enable_support") == "0"
    assert value("filament_type") == "PLA"
