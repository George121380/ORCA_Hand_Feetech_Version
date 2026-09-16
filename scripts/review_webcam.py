"""Recompute recorded-camera diagnostics and refresh its local HTML report.

Run: .venv/bin/python scripts/review_webcam.py runs/webcam-YYYYMMDD-HHMMSS
Uses saved trajectories; does not open the camera or rerun hand detection.
"""
from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path

import numpy as np
import pinocchio as pin

from orca_feetech.model import JOINT_NAMES, ModelSpec
from orca_feetech.paths import upstream


def review(folder: Path, metrics: dict) -> dict:
    rows = [json.loads(line) for line in (folder / "analysis/trajectory.jsonl").read_text().splitlines()]
    landmarks = [json.loads(line) for line in (folder / "analysis/landmarks.jsonl").read_text().splitlines()]
    first = next((i for i, row in enumerate(rows) if row["input_valid"]), None)
    periods = []
    if first is not None:
        for valid, group in groupby(range(first, len(rows)), key=lambda i: rows[i]["input_valid"]):
            indices = list(group)
            if valid:
                continue
            start, end = indices[0], indices[-1]
            periods.append({
                "start_s": round(rows[start]["time_s"], 6),
                "last_invalid_s": round(rows[end]["time_s"], 6),
                "resumed_s": round(rows[end + 1]["time_s"], 6) if end + 1 < len(rows) else None,
                "frames": len(indices),
                "commands_held": all(rows[i]["command"] == rows[start - 1]["command"] for i in indices),
            })

    spec = ModelSpec.load()
    model = pin.buildModelFromUrdf(str(upstream("orcahand_description") / "v1/models/urdf/orcahand_right.urdf"))
    data = model.createData()
    indices = [model.joints[model.getJointId("right_" + name)].idx_q for name in JOINT_NAMES]
    tips = [model.getFrameId(f"right_{finger}_fingertip") for finger in ("thumb", "index")]

    def tip_gap(position):
        q = pin.neutral(model)
        q[indices] = np.asarray(position) - spec.neutral
        pin.framesForwardKinematics(model, data, q)
        return float(1000 * np.linalg.norm(data.oMf[tips[0]].translation - data.oMf[tips[1]].translation))

    phases = []
    for phase in metrics["phases"]:
        start, end = phase["end_s"] - 2, phase["end_s"]
        tail = [r for r in rows if start <= r["time_s"] < end and r["input_valid"]]
        points = [np.asarray(r["keypoints_m"]) for r in landmarks if start <= r["time_s"] < end]
        if not tail:
            continue
        error = np.asarray([np.asarray(r["target"])[:16] - np.asarray(r["actual"])[:16] for r in tail])
        phases.append({
            "name": phase["name"], "tail_start_s": start, "tail_end_s": end,
            "valid_samples": len(tail),
            "sim_target_to_feedback_mae_deg": float(np.mean(np.abs(np.rad2deg(error)))),
            "sim_thumb_index_tip_gap_median_mm": float(np.median([tip_gap(r["actual"]) for r in tail])),
            "target_thumb_index_tip_gap_median_mm": float(np.median([tip_gap(r["target"]) for r in tail])),
            "mediapipe_estimated_thumb_index_gap_median_mm":
                float(np.median([1000 * np.linalg.norm(p[4] - p[8]) for p in points])) if points else None,
        })
    main = [r for r in rows if 8 <= r["time_s"] < 32]
    return {
        "first_detected_s": round(landmarks[0]["time_s"], 6) if landmarks else None,
        "first_commanded_s": round(rows[first]["time_s"], 6) if first is not None else None,
        "post_calibration_dropouts": periods,
        "main_guided_interval": {"start_s": 8, "end_s": 32, "frames": len(main),
                                 "detected_frames": sum(8 <= r["time_s"] < 32 for r in landmarks),
                                 "commanded_frames": sum(r["input_valid"] for r in main)},
        "phase_kinematics": phases,
        "method": "Pinned v1 URDF forward kinematics; q_URDF = q_MJCF - joint.ref; phase final 2 s, valid frames only",
        "limits": "Tip-frame origin distances are not mesh contact gaps. Monocular landmarks are estimates, not ground truth. Joint MAE compares robot target to simulated feedback over 16 finger joints.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    metrics = json.loads((folder / "validation.json").read_text())
    metrics["review"] = review(folder, metrics)
    (folder / "validation.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    from webcam_validation import report
    report(folder, metrics)
    print(json.dumps(metrics["review"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
