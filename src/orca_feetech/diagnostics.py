from __future__ import annotations

import importlib.metadata
import platform

from .model import MODEL_ID, ModelSpec
from .paths import provenance, upstream


def doctor():
    from serial.tools import list_ports
    packages = {}
    for name in ("orca-core", "orca-sim", "orca-teleop", "mujoco", "stable-baselines3",
                 "torch", "mediapipe-numpy2", "pin", "ftservo-python-sdk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for name in provenance()["repositories"]:
        try:
            sources[name] = str(upstream(name))
        except RuntimeError as exc:
            sources[name] = str(exc)
    return {"platform": platform.platform(), "python": platform.python_version(),
            "model": MODEL_ID, "model_sha256": ModelSpec.load().source_sha256,
            "packages": packages, "sources": sources,
            "serial_ports": [{"device": p.device, "description": p.description} for p in list_ports.comports()],
            "hardware_tested": False,
            "mac_viewer": "Use .venv/bin/mjpython -m orca_feetech.cli demo for interactive MuJoCo"}
