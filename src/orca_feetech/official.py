"""Version-specific configuration adapters; pinned upstream sources stay untouched."""
from __future__ import annotations

import json
import contextlib
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from .model import JOINT_NAMES, ModelSpec
from .paths import upstream, workspace


def retarget_paths() -> tuple[Path, Path, Path]:
    spec = ModelSpec.load()
    core = upstream("orca_core")
    teleop = upstream("orca_teleop")
    description = upstream("orcahand_description")
    folder = workspace() / "local" / "v1" / "retarget"
    folder.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((core / "orca_core/models/v1/orcahand-right/config.yaml").read_text())
    # These are MJCF qpos coordinates, matching the official retargeter's ref-offset
    # output, not the original Dynamixel's physical joint calibration coordinates.
    config["joint_ids"] = list(JOINT_NAMES)
    config["joint_roms"] = {n: [float(np.rad2deg(spec.lower[i])), float(np.rad2deg(spec.upper[i]))]
                            for i, n in enumerate(JOINT_NAMES)}
    config["neutral_position"] = dict(zip(JOINT_NAMES, np.rad2deg(spec.neutral).tolist()))
    config["motor_type"] = "feetech"
    config["control_mode"] = "position"
    config["baudrate"] = 1_000_000
    config_path = folder / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    settings = yaml.safe_load((teleop / "src/orca_teleop/retargeting/configs/adaptive_analytical_orca.yaml").read_text())
    settings["joint_name_aliases"] = {"right": {}, "left": {}}
    settings["retarget"]["lp_alpha"] = 0.35
    settings_path = folder / "retarget.yaml"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    urdf = description / "v1/models/urdf/orcahand_right.urdf"
    (folder / "adaptation.json").write_text(json.dumps({
        "reason": "v1 thumb_pip is a real URDF joint; no thumb_cmc alias. ROM/neutral use MJCF qpos.",
        "fingertips": "Use the v1 URDF fingertip origins without a second distal offset.",
        "virtual_dip": "Four two-link fingers use the midpoint of the last phalanx as a DIP landmark.",
        "source_model_sha256": spec.source_sha256,
    }, indent=2) + "\n")
    return config_path, urdf, settings_path


def make_retargeter():
    from orca_teleop.retargeting.retargeter import Retargeter
    config, urdf, settings = retarget_paths()
    # Official construction prints 17 motor-calibration warnings even though this
    # instance only reads kinematic configuration and never connects to a bus.
    # Capture stdout only during that construction; exceptions remain visible.
    with contextlib.redirect_stdout(io.StringIO()):
        retargeter = Retargeter.from_paths(str(config), str(urdf), backend="adaptive_analytical", config_path=str(settings))
    _adapt_v1_landmarks(retargeter, urdf)
    return retargeter


def _adapt_v1_landmarks(retargeter, urdf: Path):
    """Adapt landmark geometry on the pinned solver instance, before calibration.

    v1 already places *_fingertip at the tip. The upstream additional 30–45 mm
    offsets double-count distal length. Its non-thumb PIP joint and IP body also
    coincide; use a virtual DIP halfway along that rigid distal link. This is a
    geometric approximation for a human joint absent from the two-link robot.
    """
    if retargeter.model_version != "v1":
        raise ValueError("Landmark adaptation is only validated against the pinned v1 model")
    root = ET.parse(urdf).getroot()
    for index in retargeter._tip_indices:
        retargeter._frame_offsets[index] = np.zeros(3)
    for finger in ("index", "middle", "ring", "pinky"):
        parent_name, tip_name = f"right_{finger}_ip", f"right_{finger}_fingertip"
        joint = next(j for j in root.findall("joint") if j.find("child").get("link") == tip_name)
        if joint.get("type") != "fixed" or joint.find("parent").get("link") != parent_name:
            raise ValueError(f"Unexpected v1 distal geometry for {finger}")
        offset = np.fromstring(joint.find("origin").get("xyz"), sep=" ")
        if offset.shape != (3,) or not 0.02 < np.linalg.norm(offset) < 0.06:
            raise ValueError(f"Unexpected v1 fingertip offset for {finger}")
        index = retargeter._computed_frame_names.index(parent_name)
        retargeter._frame_offsets[index] = 0.5 * offset
    # Scale estimation must use the corrected robot lengths too.
    retargeter._build_target_alignment_params()
