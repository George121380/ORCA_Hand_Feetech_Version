"""Build the offline HTML report from checked-in validation data and media.

Run with Python 3.11+: python3 scripts/build_results.py
No packages, web server or internet connection are required to view the output.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    validation = root / "artifacts/validation"
    data = {
        "summary": json.loads((validation / "summary.json").read_text()),
        "series": json.loads((root / "artifacts/visualization/series.json").read_text()),
        "evaluations": {key: json.loads((validation / f"{key}.json").read_text())
                        for key in ("seed0", "seed1", "seed2", "baseline", "random", "moving-goal")},
        "upstream": json.loads((root / "upstream.lock.json").read_text()),
    }
    media_files = {
        "demo": "gesture-demo.mp4", "ppo": "ppo-inference.mp4", "vision": "teleop-fixture.mp4",
        "demoPoster": "preview.png", "visionPoster": "pipeline-preview.png",
    }
    data["media"] = {}
    for name, filename in media_files.items():
        path = root / "docs/assets/software" / filename
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data["media"][name] = f"data:{mimetypes.guess_type(filename)[0]};base64,{encoded}"
    template = (root / "scripts/results.template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    if template.count("__RESULTS_DATA__") != 1:
        raise ValueError("Template must contain exactly one data placeholder")
    result = template.replace("__RESULTS_DATA__", payload)
    output = root / "docs/results.html"
    output.write_text(result)
    print(f"Built {output.relative_to(root)} ({output.stat().st_size / 1024 / 1024:.2f} MiB, self-contained)")


if __name__ == "__main__":
    main()
