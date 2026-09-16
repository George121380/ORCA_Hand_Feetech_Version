"""Build a clearly labelled pipeline fixture from Google's MediaPipe test images."""
from pathlib import Path
import hashlib
import json
import urllib.request

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "right_hands.jpg": "4b5134daa4cb60465535239535f9f74c2842aba3aa5fd30bf04ef5678f93d87f",
    "thumb_up.jpg": "5d673c081ab13b8a1812269ff57047066f9c33c07db5f4178089e8cb3fdc0291",
    "pointing_up.jpg": "ecf8ca2611d08fa25948a4fc10710af9120e88243a54da6356bacea17ff3e36e",
}


def main():
    folder = ROOT / "local/fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "mediapipe_fixture.mp4"
    with imageio.get_writer(output, fps=30, macro_block_size=1) as writer:
        for name, digest in ASSETS.items():
            path = folder / name
            if not path.exists():
                with urllib.request.urlopen("https://storage.googleapis.com/mediapipe-assets/" + name, timeout=30) as r:
                    path.write_bytes(r.read())
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Fixture digest changed: {name}")
            frame = cv2.imread(str(path))
            scale = min(720 / frame.shape[1], 420 / frame.shape[0])
            resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
            canvas = np.full((480, 720, 3), 35, dtype=np.uint8)
            h, w = resized.shape[:2]
            canvas[40:40 + h, (720 - w) // 2:(720 - w) // 2 + w] = resized
            cv2.putText(canvas, "MediaPipe static-image pipeline fixture", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            for _ in range(75):
                writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            if name == "thumb_up.jpg":
                for _ in range(30):
                    writer.append_data(np.zeros((480, 720, 3), dtype=np.uint8))
    (folder / "sources.json").write_text(json.dumps({
        "base_url": "https://storage.googleapis.com/mediapipe-assets/", "sha256": ASSETS,
        "purpose": "Repeated static images + black frames; functional test, not a live teleoperation benchmark",
        "test_reference": "https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/hand_landmarker_test.py",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
