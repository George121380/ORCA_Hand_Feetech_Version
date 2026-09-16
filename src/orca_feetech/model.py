from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

JOINT_NAMES = tuple(
    [f"thumb_{part}" for part in ("mcp", "abd", "pip", "dip")]
    + [f"{finger}_{part}" for finger in ("index", "middle", "ring", "pinky")
       for part in ("abd", "mcp", "pip")]
    + ["wrist"]
)
MODEL_ID = "orca-v1-right"


def vector(value, size: int = 17) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite joint values, got shape {result.shape}")
    return result.copy()


@dataclass(frozen=True)
class ModelSpec:
    lower: np.ndarray
    upper: np.ndarray
    neutral: np.ndarray
    source_sha256: str

    @classmethod
    def load(cls) -> "ModelSpec":
        data = json.loads(files("orca_feetech").joinpath("data/v1_right.json").read_text())
        if tuple(data["joint_names"]) != JOINT_NAMES:
            raise ValueError("Model metadata joint order mismatch")
        return cls(*(vector(data[key]) for key in ("lower", "upper", "neutral")), data["sha256"])

    def clip(self, q) -> np.ndarray:
        return np.clip(vector(q), self.lower, self.upper)


def poses() -> dict[str, np.ndarray]:
    """Named model-space poses; real hardware uses the intersection with measured ROM."""
    spec = ModelSpec.load()
    opened = spec.neutral.copy()
    fist = opened.copy()
    for finger in ("index", "middle", "ring", "pinky"):
        for part, degrees in (("mcp", 55), ("pip", 65)):
            fist[JOINT_NAMES.index(f"{finger}_{part}")] += np.deg2rad(degrees)
    for joint, degrees in (("thumb_mcp", 15), ("thumb_abd", 20), ("thumb_pip", 35), ("thumb_dip", 40)):
        fist[JOINT_NAMES.index(joint)] += np.deg2rad(degrees)
    pinch = opened.copy()
    for joint, degrees in (("thumb_mcp", 15), ("thumb_abd", 25), ("thumb_pip", 30),
                           ("thumb_dip", 30), ("index_mcp", 35), ("index_pip", 45)):
        pinch[JOINT_NAMES.index(joint)] += np.deg2rad(degrees)
    result = {"open": opened, "fist": spec.clip(fist), "pinch": spec.clip(pinch)}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        q = opened.copy()
        for i, name in enumerate(JOINT_NAMES):
            if name.startswith(finger + "_"):
                q[i] = result["fist"][i]
        result[finger] = q
    return result


def to_orca(q):
    from orca_core import OrcaJointPositions
    return OrcaJointPositions.from_ndarray(np.rad2deg(vector(q)), joint_ids=JOINT_NAMES)


def from_orca(positions) -> np.ndarray:
    return vector(np.deg2rad(positions.as_array(JOINT_NAMES)))
