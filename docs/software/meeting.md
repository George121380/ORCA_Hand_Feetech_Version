# English meeting slides

[Open the five-slide deck](../meeting.html). On GitHub, download the raw HTML first and open it in a browser. All three videos, posters, styles, and scripts are embedded; presentation playback works offline.

Suggested length: **4–5 minutes**, plus an optional live camera demo.

1. The idea: a low-cost ORCA Hand for dexterous manipulation.
2. Working RGB teleoperation in simulation.
3. What the pen-reorientation policy observes, controls, and learns.
4. Early results: one success and six drops in 32 trials.
5. Three areas for team contributions and a proposed next milestone.

## Present

Open `docs/meeting.html` directly, or serve the docs:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Visit `http://127.0.0.1:8765/meeting.html`. Use left/right arrows to change slides, **F** for fullscreen, **P** to pause/play the current video, and **N** for speaker notes and sources. Notes appear on the same screen, so close them before screen sharing. The `#1`–`#5` URL fragments open individual slides.

The live-demo link on slide 2 requires a separate local server:

```bash
.venv/bin/python scripts/webcam_live.py
```

It does not start the camera automatically. Click Start in the live page and hold the right hand open for calibration. Stop the camera after the demonstration.

## Edit or rebuild

Edit `scripts/meeting.template.html`, then run:

```bash
python3 scripts/build_meeting.py
```

The builder uses only checked-in media and JSON results; no training run or camera recording is required. The pen replay in `docs/assets/meeting/pen-success.mp4` is the independently reproduced successful seed 20011, from the experimental checkpoint in `artifacts/pretrained/pen-v1-experimental`. It is explicitly labeled as a selected success. Its companion JPG shows the same simulation trial.

The teleoperation clip uses public MediaPipe test images; it is not live-camera footage. The deck contains no personal webcam recording. Hardware control and policy transfer remain unvalidated.
