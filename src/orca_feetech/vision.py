from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .backends import create_backend
from .control import CommandGuard, JointTarget
from .model import ModelSpec, from_orca
from .official import make_retargeter
from .paths import upstream
from .recording import SessionRecorder


class HandPoseExtractor:
    """Google Tasks API with ORCA's pinned model and detector settings.

    VIDEO mode is synchronous and preserves source timestamps. It works for a
    camera too and avoids the upstream publisher's late-callback timestamp reset.
    World landmarks are monocular estimates, in MediaPipe's documented metres.
    """
    def __init__(self, confidence=0.7):
        import mediapipe as mp
        if not 0 < confidence <= 1:
            raise ValueError("Confidence must be in (0, 1]")
        self.mp = mp
        self.confidence = confidence
        model = upstream("orca_teleop") / "src/orca_teleop/ingress/mediapipe/hand_landmarker.task"
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def extract(self, rgb, timestamp_ms):
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms <= self.last_timestamp_ms:
            raise ValueError("Video timestamps must strictly increase")
        self.last_timestamp_ms = timestamp_ms
        frame = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect_for_video(frame, timestamp_ms)
        for i, categories in enumerate(result.handedness):
            if categories[0].category_name.lower() != "right" or categories[0].score < self.confidence:
                continue
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
            image_points = np.array([[p.x, p.y] for p in result.hand_landmarks[i]])
            if world.shape == (21, 3) and np.isfinite(world).all():
                return world, image_points, float(categories[0].score)
        return None

    def close(self):
        self.detector.close()


def draw_overlay(frame, points, status):
    import cv2
    from orca_teleop.ingress.mediapipe.publisher import _HAND_CONNECTIONS
    if points is not None:
        pixels = np.rint(points * [frame.shape[1], frame.shape[0]]).astype(int)
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), (90, 205, 120), 2)
        for pt in pixels:
            cv2.circle(frame, tuple(pt), 3, (40, 80, 240), -1)
    if status is not None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (25, 25, 25), -1)
        cv2.putText(frame, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1)
    return frame


def teleop(source="camera:0", backend="sim", config=None, headless=False, rate=20,
           output="runs/teleop", video=False, confidence=0.7, mirror_input=False,
           max_frames=0, show_video=False):
    import cv2
    import imageio.v2 as imageio
    from orca_teleop.retargeting.retargeter import TargetPose

    if not 4 <= rate <= 60:
        raise ValueError("Control rate must be 4..60 Hz")
    camera = source.startswith("camera:")
    if not camera and not Path(source).is_file():
        raise ValueError(f"Video does not exist: {source}")
    if video and backend != "sim":
        raise ValueError("--video records simulated output; first run with --backend sim")
    capture = cv2.VideoCapture(int(source.split(":", 1)[1]) if camera else str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    render_mode = "rgb_array" if video else None if headless or backend != "sim" else "human"
    detector = retargeter = robot = recorder = overlay_writer = landmark_file = None
    error = None
    total = detected = commanded = 0
    latencies = []
    try:
        detector = HandPoseExtractor(confidence)
        retargeter = make_retargeter()
        recorder = SessionRecorder(output, backend, rate, video)
        overlay_writer = imageio.get_writer(Path(output) / "rgb_overlay.mp4", fps=rate, macro_block_size=1)
        landmark_file = (Path(output) / "landmarks.jsonl").open("w")
        # Hardware is only armed after detector/solver setup is complete.
        # Scale calibration runs before opening the real backend, avoiding startup timeout.
        if backend != "hardware":
            robot = create_backend(backend, render_mode=render_mode, rate=rate)
            guard = CommandGuard(robot.read().positions, robot.lower, robot.upper)
        spec = ModelSpec.load()
        previous = spec.neutral.copy()
        wall_start = time.monotonic()
        next_control_t = 0.0
        last_video_t = -1.0
        frame_index = 0
        last_dispatch = wall_start - 1 / rate
        while not max_frames or total < max_frames:
            ok, frame = capture.read()
            if not ok:
                if camera:
                    raise RuntimeError("Camera disconnected / failed frame read")
                break
            acquired = time.monotonic()
            timestamp = acquired - wall_start if camera else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not camera and (not np.isfinite(timestamp) or timestamp <= last_video_t):
                timestamp = frame_index / fps
            frame_index += 1
            last_video_t = timestamp
            if timestamp + 1e-8 < next_control_t:
                continue
            # Preserve timing: one processed sample per control period; no camera backlog.
            next_control_t = (np.floor(timestamp * rate + 1e-8) + 1) / rate
            if not camera:
                time.sleep(max(0, wall_start + timestamp - time.monotonic()))
                acquired = time.monotonic()
            total += 1
            if mirror_input:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = detector.extract(rgb, round(timestamp * 1000))
            target, points, status = None, None, "No right hand - holding"
            if pose is not None:
                world, points, score = pose
                detected += 1
                landmark_file.write(json.dumps({"time_s": timestamp, "keypoints_m": world.tolist(),
                                                "hand": "right", "handedness_score": score}) + "\n")
                try:
                    result = retargeter.retarget(TargetPose(joint_positions=world, source="mediapipe"))
                except (ValueError, RuntimeError) as exc:
                    status = f"Tracking rejected: {type(exc).__name__}"
                    result = None
                if result is None:
                    status = "Calibrating hand scale / no target"
                else:
                    q = from_orca(result)
                    q[-1] = spec.neutral[-1]  # No forearm reference in hand-only RGB tracking.
                    target = JointTarget(q, timestamp=acquired)
                    status = "Tracking right hand"
            if target is not None and robot is None:
                robot = create_backend(backend, hardware_config=config, rate=rate)
                guard = CommandGuard(robot.read().positions, robot.lower, robot.upper)
                # Backend setup can take longer than 250 ms; do not re-date this frame.
            frame = draw_overlay(frame, points, status)
            overlay_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if robot is not None:
                dispatched = time.monotonic()
                effective_dt = (min(0.25, max(1e-5, dispatched - last_dispatch))
                                if backend == "hardware" else 1 / rate)
                command, fresh = guard.apply(target, effective_dt, now=dispatched)
                last_dispatch = dispatched
                if fresh:
                    previous = target.positions
                    robot.write(command, target.timestamp)
                    commanded += 1
                elif backend != "hardware":
                    robot.write(command)
                state = robot.read()
                simulation_frame = robot.render()
                recorder.append(previous, command, state, time_s=timestamp, frame=simulation_frame, valid=fresh)
            latencies.append((time.monotonic() - acquired) * 1000)
            if show_video:
                cv2.imshow("ORCA RGB pose", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        summary = {"source": source, "backend": backend, "frames": total,
                   "detected_frames": detected, "commanded_frames": commanded,
                   "processing_p50_ms": float(np.median(latencies)) if latencies else None,
                   "processing_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                   "note": "Monocular estimated 3D; wrist held at model neutral; source timestamps retained"}
        (Path(output) / "vision_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if robot:
                robot.close()
        except BaseException as exc:
            error = exc
            raise
        finally:
            capture.release()
            if detector:
                detector.close()
            if overlay_writer:
                overlay_writer.close()
            if landmark_file:
                landmark_file.close()
            if recorder:
                recorder.close(error)
            if show_video:
                cv2.destroyAllWindows()
