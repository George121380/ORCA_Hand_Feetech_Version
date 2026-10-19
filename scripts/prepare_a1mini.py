"""Create A1 mini projects from a native seed and installed Bambu vendor profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten(path, folder):
    data = json.loads(path.read_text())
    result = {}
    if data.get("inherits"):
        result.update(flatten(folder / (data["inherits"]+".json"), folder))
    for include in data.get("include", []):
        result.update(flatten(folder / (include+".json"), folder))
    result.update(data)
    result.pop("inherits", None)
    result.pop("include", None)
    return result


def build_project(seed, output, kind, material, profiles, split=False):
    import xml.etree.ElementTree as ET
    import zipfile
    import numpy as np
    from orca_feetech.objects import geometry
    namespace = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ns = {"m": namespace}
    ET.register_namespace("", namespace)
    ET.register_namespace("p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06")
    with zipfile.ZipFile(seed) as z:
        files = {name: z.read(name) for name in z.namelist()}
    settings = json.loads(files["Metadata/project_settings.config"])
    process = flatten(profiles/"process/0.20mm Standard @BBL A1M.json", profiles/"process")
    filament = flatten(profiles/f"filament/Generic {material} @BBL A1M.json", profiles/"filament")
    for profile in (process, filament):
        for key, value in profile.items():
            if key in settings:
                settings[key] = value
    changes = dict(wall_loops="3", top_shell_layers="5", bottom_shell_layers="5",
                   sparse_infill_density="20%", sparse_infill_pattern="gyroid",
                   brim_type="outer_only", brim_width="8", brim_object_gap="0.1",
                   enable_support="0", outer_wall_speed="40", inner_wall_speed="60",
                   default_acceleration="1000", travel_acceleration="1500")
    if split:
        changes.update(wall_loops="4", brim_width="3", elefant_foot_compensation="0.15")
    for key, value in changes.items():
        settings[key] = [value] if isinstance(settings.get(key), list) else value
    settings.update(filament_colour=["#EEEEEE"], filament_type=[material],
                    filament_settings_id=[f"Generic {material} @BBL A1M"])
    files["Metadata/project_settings.config"] = (json.dumps(settings, indent=2)+"\n").encode()
    if split:
        from orca_feetech.printing import split_pen_plate
        points, triangles = split_pen_plate()
        points -= [90, 90, 6]
    else:
        points, triangles = geometry(kind)
        points *= 1000
    if kind == "pen" and not split:
        raise ValueError("Long pen prints as two half-shells; use split=True")
    height = float(np.ptp(points[:, 2]))
    transform = f"1 0 0 0 1 0 0 0 1 90 90 {height/2:g}"
    root = ET.fromstring(files["3D/Objects/object_1.model"])
    vs, fs = root.find(".//m:vertices", ns), root.find(".//m:triangles", ns)
    vs.clear()
    fs.clear()
    for point in points:
        ET.SubElement(vs, "{"+namespace+"}vertex", dict(zip(("x", "y", "z"), map(str, point), strict=True)))
    for tri in triangles:
        ET.SubElement(fs, "{"+namespace+"}triangle", dict(zip(("v1", "v2", "v3"), map(str, tri), strict=True)))
    files["3D/Objects/object_1.model"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    root = ET.fromstring(files["3D/3dmodel.model"])
    root.find("m:build/m:item", ns).set("transform", transform)
    for metadata in root.findall("m:metadata", ns):
        if metadata.get("name") in ("DesignerUserId", "Designer", "DesignerCover", "ProfileCover"):
            metadata.text = ""
        if metadata.get("name") == "Title":
            metadata.text = ("ORCA 210mm pen HALF / PRINT TWICE / " if split else f"ORCA {kind} / A1 mini / ")+material
    files["3D/3dmodel.model"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    root = ET.fromstring(files["Metadata/model_settings.config"])
    for meta in root.findall(".//metadata"):
        key = meta.get("key")
        if key == "name":
            meta.set("value", ("ORCA 210mm pen HALF PRINT TWICE " if split else f"ORCA {kind} A1 mini ")+material)
        if key == "source_file":
            meta.set("value", "pen-split-a1mini.3mf" if split else f"{kind}-a1mini.3mf")
        if key == "source_offset_z":
            meta.set("value", str(height/2))
        if meta.get("face_count"):
            meta.set("face_count", str(len(triangles)))
    for item in root.findall(".//assemble_item"):
        item.set("transform", transform)
    for stat in root.findall(".//mesh_stat"):
        stat.set("face_count", str(len(triangles)))
    for node in root.findall(".//plate/metadata"):
        if node.get("key") in ("thumbnail_file", "thumbnail_no_light_file", "top_file", "pick_file"):
            node.set("value", "")
    files["Metadata/model_settings.config"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if not name.endswith(".png") and "gcode" not in name.lower():
                z.writestr(name, data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--studio", type=Path, default=Path("/Applications/BambuStudio.app"))
    p.add_argument("--seed", type=Path, default=Path("assets/printing/a1mini-seed.3mf"))
    p.add_argument("--output", type=Path, default=Path("assets/objects/v4"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = args.studio/"Contents/Resources/profiles/BBL"
    for kind in ("cube",):
        for material in ("PLA", "PETG"):
            build_project(args.seed, args.output/f"{kind}-a1mini-{material.lower()}.3mf",
                          kind, material, profiles)
    from orca_feetech.printing import export_split_pen
    export_split_pen(args.output)
    for material in ("PLA", "PETG"):
        build_project(args.seed, args.output/f"pen-split-a1mini-{material.lower()}.3mf",
                      "pen", material, profiles, split=True)


if __name__ == "__main__":
    main()
