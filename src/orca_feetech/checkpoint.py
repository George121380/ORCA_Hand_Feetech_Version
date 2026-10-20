from __future__ import annotations

import json
from pathlib import Path
from .manipulation import TaskConfig, TASK_ID, HISTORY, FRAME_SIZE
from .model import JOINT_NAMES
from .paths import provenance

def load_config(checkpoint):
    return _load_config(checkpoint, TASK_ID, FRAME_SIZE)


def load_v6_config_for_migration(checkpoint):
    """Explicit read-only legacy entry point; deployment still rejects v6."""
    return _load_config(checkpoint, "orca-v1-printable-yaw-v6", 58)


def _load_config(checkpoint, task_id, frame_size, config_class=TaskConfig):
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint.parent / "config.json").read_text())
    cfg = config_class(**meta["task"])
    if (meta["task_id"] != task_id or tuple(meta["joint_names"]) != JOINT_NAMES
            or meta["fingerprint"] != cfg.fingerprint(task_id=task_id) or meta["upstream"] != provenance()
            or meta["history"] != HISTORY or meta["frame_size"] != frame_size):
        raise ValueError("Checkpoint geometry, observation, model or upstream lock mismatch")
    return cfg, meta
