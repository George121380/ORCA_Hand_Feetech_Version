"""Build a self-contained HTML view of the recorded pen experiments."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {"hold": "pen-hold", "random": "pen-random", "pilot": "pen-ppo-eval-pilot", "ppo": "pen-ppo-eval"}
    experiments = {}
    for key, name in paths.items():
        folder = ROOT / "runs" / name
        if not (folder / "metrics.json").exists():
            continue
        experiments[key] = {
            "metrics": json.loads((folder / "metrics.json").read_text()),
            "traces": json.loads((folder / "traces.json").read_text()),
            "video": "data:video/mp4;base64," + base64.b64encode((folder / "evaluation.mp4").read_bytes()).decode(),
        }
    training = {}
    for key, name in (("pilot", "pen-ppo-pilot2"), ("ppo", "pen-ppo-final")):
        path = ROOT / "runs" / name / "config.json"
        if key in experiments and path.exists():
            training[key] = json.loads(path.read_text())
    data = {"experiments": experiments, "training": training}
    success_folder = ROOT / "runs/pen-success"
    if (success_folder / "metrics.json").exists():
        example = json.loads((success_folder / "metrics.json").read_text())
        if example["success_rate"] != 1 or example["episodes"] != 1:
            raise ValueError("Selected success replay did not reproduce")
        data["success_example"] = {
            "seed": example["results"][0]["seed"],
            "traces": json.loads((success_folder / "traces.json").read_text()),
            "video": "data:video/mp4;base64," + base64.b64encode((success_folder / "evaluation.mp4").read_bytes()).decode(),
        }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "scripts/pen_results.template.html").read_text()
    output = ROOT / "docs/pen.html"
    output.write_text(template.replace("__PEN_DATA__", payload))
    destination = ROOT / "artifacts/validation/pen"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps({
        "scope": "Simulation teacher policy, palm-supported free pen, target axis rotation +/-90 degrees; not continuous finger spinning or real hardware",
        "evaluations": {key: item["metrics"] for key, item in experiments.items()},
        "training": training,
    }, indent=2) + "\n")
    print(f"Built {output.relative_to(ROOT)}: {output.stat().st_size/1024/1024:.2f} MiB")


if __name__ == "__main__":
    main()
