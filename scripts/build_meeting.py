"""Build the offline English meeting deck from checked-in media and results."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    html = (ROOT / "scripts/meeting.template.html").read_text()
    media = {
        "GESTURE_VIDEO": ("docs/assets/software/gesture-demo.mp4", "video/mp4"),
        "GESTURE_POSTER": ("docs/assets/software/preview.png", "image/png"),
        "TELEOP_VIDEO": ("docs/assets/software/teleop-fixture.mp4", "video/mp4"),
        "TELEOP_POSTER": ("docs/assets/software/pipeline-preview.png", "image/png"),
        "PEN_VIDEO": ("docs/assets/meeting/pen-success.mp4", "video/mp4"),
        "PEN_POSTER": ("docs/assets/meeting/pen-success.jpg", "image/jpeg"),
    }
    for key, (path, mime) in media.items():
        encoded = base64.b64encode((ROOT / path).read_bytes()).decode()
        html = html.replace(f"__{key}__", f"data:{mime};base64,{encoded}")
    pen = json.loads((ROOT / "artifacts/validation/pen/summary.json").read_text())
    summary = json.loads((ROOT / "artifacts/validation/summary.json").read_text())
    for key in ("hold", "pilot", "ppo"):
        error = pen["evaluations"][key]["mean_final_angle_error_deg"]
        html = html.replace(f"__{key.upper()}_ERROR__", f"{error:.1f}")
        html = html.replace(f"__{key.upper()}_WIDTH__", f"{error / 110 * 100:.2f}%")
    evaluation = pen["evaluations"]["ppo"]
    values = {
        "TRIALS": evaluation["episodes"],
        "SUCCESSES": sum(r["success"] for r in evaluation["results"]),
        "DROPS": sum(r["dropped"] for r in evaluation["results"]),
        "TRAINING_STEPS": f'{summary["pen_reorientation"]["training_steps"]:,}',
    }
    for key, value in values.items():
        html = html.replace(f"__{key}__", str(value))
    output = ROOT / "docs/meeting.html"
    output.write_text(html)
    print(f"Built {output.relative_to(ROOT)} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
