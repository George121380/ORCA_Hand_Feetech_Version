from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .model import JOINT_NAMES, MODEL_ID, vector
from .paths import provenance


class SessionRecorder:
    def __init__(self, output, backend, rate=20, video=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "trajectory.jsonl").exists():
            raise ValueError(f"{self.output} already contains a trajectory; choose a new --output")
        self.file = (self.output / "trajectory.jsonl").open("w")
        self.start = time.monotonic()
        self.rows = []
        self.writer = None
        self.metadata = {"schema_version": 1, "model_id": MODEL_ID, "joint_names": JOINT_NAMES,
                         "units": "radians", "time_unit": "seconds", "backend": backend,
                         "rate_hz": rate, "upstream": provenance(), "status": "running"}
        (self.output / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        if video:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(self.output / "simulation.mp4", fps=rate, macro_block_size=1)

    def append(self, target, command, state, *, time_s=None, frame=None, valid=True):
        row = {"time_s": time.monotonic() - self.start if time_s is None else float(time_s),
               "target": vector(target).tolist(), "command": vector(command).tolist(),
               "actual": vector(state.positions).tolist(), "estimated_feedback": state.estimated,
               "input_valid": bool(valid), "telemetry": state.telemetry}
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()
        self.rows.append(row)
        if self.writer is not None and frame is not None:
            self.writer.append_data(frame)

    def close(self, error=None):
        self.file.close()
        if self.writer:
            self.writer.close()
        self.metadata["status"] = "failed" if error else "complete"
        self.metadata["error"] = str(error) if error else None
        self.metadata["samples"] = len(self.rows)
        (self.output / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        if self.rows:
            try:
                self.plot()
            except ImportError:
                pass

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        times = [r["time_s"] for r in self.rows]
        for ax, joint in zip(axes, ("thumb_pip", "index_mcp", "middle_mcp"), strict=True):
            index = JOINT_NAMES.index(joint)
            for field, label in (("target", "Target"), ("command", "Limited command"), ("actual", "Feedback")):
                ax.plot(times, [np.rad2deg(r[field][index]) for r in self.rows], label=label, linewidth=1)
            ax.set(title=joint, ylabel="Angle (degrees)")
            ax.legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("Session time (s)")
        fig.suptitle(f"{self.metadata['backend']} joint tracking — "
                     + ("estimated joint feedback" if self.rows[0]["estimated_feedback"] else "simulation / mock feedback"))
        fig.savefig(self.output / "tracking.png", dpi=150)
        plt.close(fig)


def load_trajectory(path):
    path = Path(path)
    metadata = json.loads((path.parent / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("model_id") != MODEL_ID:
        raise ValueError("Trajectory schema/model mismatch")
    if tuple(metadata["joint_names"]) != JOINT_NAMES or metadata.get("units") != "radians":
        raise ValueError("Trajectory ordering/units mismatch")
    if metadata.get("status") != "complete":
        raise ValueError("Only completed sessions may be replayed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    previous = -np.inf
    for row in rows:
        if not np.isfinite(row["time_s"]) or row["time_s"] < 0 or row["time_s"] <= previous:
            raise ValueError("Trajectory timestamps must be finite, positive and increasing")
        row["command"] = vector(row["command"])
        previous = row["time_s"]
    if not rows:
        raise ValueError("Trajectory is empty")
    return rows
