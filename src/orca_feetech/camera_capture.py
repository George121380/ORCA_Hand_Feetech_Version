"""Local camera preview and timestamp-paced video capture for the validation UI."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path


class LocalCamera:
    def __init__(self, index=0, preview_jpeg=True):
        self.index = index
        self.preview_jpeg = preview_jpeg
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.stop_recording = threading.Event()
        self.frame = self.jpeg = None
        self.sequence = 0
        self.captured_at = 0.0
        self.error = None
        self.record_thread = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
        if not self.ready.wait(10) or self.frame is None:
            self.closed.set()
            raise RuntimeError(self.error or "Camera did not return a frame")

    def _capture(self):
        import cv2
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        try:
            if not cap.isOpened():
                raise RuntimeError("macOS has not allowed camera access")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while not self.closed.is_set():
                ok, frame = cap.read()
                captured_at = time.monotonic()
                if not ok:
                    raise RuntimeError("Camera frame read failed")
                jpeg = None
                if self.preview_jpeg:
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError("Camera preview encoding failed")
                    jpeg = encoded.tobytes()
                with self.lock:
                    self.frame, self.jpeg = frame, jpeg
                    self.sequence += 1
                    self.captured_at = captured_at
                self.ready.set()
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
        finally:
            cap.release()

    def snapshot(self):
        """One atomic, timestamped copy of the latest frame; never a frame queue."""
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            return self.frame.copy(), self.sequence, self.captured_at

    def close(self):
        self.stop_recording.set()
        self.closed.set()
        self.thread.join(timeout=2)

    def start(self, folder: Path, seconds, phases, callback):
        if self.record_thread is not None:
            raise ValueError("Recording already started")
        self.record_thread = threading.Thread(target=self._record, args=(folder, seconds, phases, callback), daemon=True)
        self.record_thread.start()

    def _record(self, folder, seconds, phases, callback):
        import cv2
        rate = 20
        with self.lock:
            height, width = self.frame.shape[:2]
        writer = cv2.VideoWriter(str(folder / "capture.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), rate, (width, height))
        if not writer.isOpened():
            self.error = "Could not create the camera recording"
            self.closed.set()
            callback(self.error)
            return
        started = time.monotonic()
        frames, sequences = 0, set()
        error = None
        try:
            while frames < round(seconds * rate) and not self.stop_recording.is_set():
                time.sleep(max(0, started + frames / rate - time.monotonic()))
                if self.error:
                    raise RuntimeError(self.error)
                with self.lock:
                    frame, sequence = self.frame.copy(), self.sequence
                writer.write(frame)
                sequences.add(sequence)
                frames += 1
        except Exception as exc:
            error = str(exc)
        finally:
            writer.release()
            self.closed.set()
            self.thread.join(timeout=2)
        metadata = {"source": "mac_camera_native", "recorded_at": datetime.now().astimezone().isoformat(),
                    "duration_s": frames / rate, "completed_guide": frames >= round(seconds * rate),
                    "frames": frames, "unique_camera_frames": len(sequences),
                    "repeated_frames": frames - len(sequences), "phases": phases,
                    "settings": {"width": width, "height": height, "recording_fps": rate},
                    "note": "Camera sampled on a 20 Hz wall-clock timeline; repeated source frames counted"}
        (folder / "capture.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        callback(error)
