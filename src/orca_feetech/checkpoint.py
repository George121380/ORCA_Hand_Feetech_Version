from __future__ import annotations

import json
from pathlib import Path
from .manipulation import TaskConfig, TASK_ID, HISTORY, FRAME_SIZE
from .model import JOINT_NAMES
from .paths import provenance

def load_config(checkpoint):
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint.parent / "config.json").read_text())
    cfg = TaskConfig(**meta["task"])
    if (meta["task_id"] != TASK_ID or tuple(meta["joint_names"]) != JOINT_NAMES
            or meta["fingerprint"] != cfg.fingerprint() or meta["upstream"] != provenance()
            or meta["history"] != HISTORY or meta["frame_size"] != FRAME_SIZE):
        raise ValueError("Checkpoint geometry, observation, model or upstream lock mismatch")
    return cfg, meta

