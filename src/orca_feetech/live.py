"""Latest-frame camera → official retargeter → simulation for a local web UI."""
from __future__ import annotations

import base64
import threading
import time
from collections import deque

import numpy as np

from .backends import SimBackend
from .camera_capture import LocalCamera
from .control import CommandGuard, JointTarget
from .model import ModelSpec, from_orca
from .official import make_retargeter
from .vision import HandPoseExtractor, draw_overlay


class LiveSession:
    def __init__(self, rate=20, camera_index=0, camera_factory=None):
        if not 4 <= rate <= 60:
            raise ValueError("Rate must be 4..60 Hz")
        self.rate = rate
        self.camera_factory = camera_factory or (lambda: LocalCamera(camera_index, preview_jpeg=False))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.worker = None
        self.last_client = time.monotonic()
        self.state = {"status": "ready", "message": "点击开始，张开右手完成标定。", "frame_id": 0}
        self.pair = None
        self.captured_at = None
        self.generation = 0

    def start(self):
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return
            self.generation += 1
            self.stop_event.clear()
            self.reset_event.clear()
            self.pair = self.captured_at = None
            self.last_client = time.monotonic()
            self.state = {"status": "initializing", "message": "正在加载检测模型和官方仿真手…", "frame_id": 0}
            self.worker = threading.Thread(target=self._run, daemon=True, name="orca-live")
            self.worker.start()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                self.state.update(status="stopping", message="正在关闭摄像头…")

    def recalibrate(self):
        self.reset_event.set()

    def snapshot(self, after=None, heartbeat=True):
        with self.lock:
            if heartbeat:
                self.last_client = time.monotonic()
            state = dict(self.state)
            state["running"] = self.worker is not None and self.worker.is_alive()
            state["generation"] = self.generation
            state["frame_age_ms"] = (time.monotonic() - self.captured_at) * 1000 if self.captured_at else None
            if self.pair is not None and after != f'{self.generation}:{state["frame_id"]}':
                state["pair"] = self.pair
            return state

    def _run(self):
        import cv2
        from orca_teleop.retargeting.retargeter import TargetPose

        camera = detector = robot = None
        error = None
        count = detected = commanded = calibration = 0
        published_times = deque(maxlen=40)
        durations = deque(maxlen=100)
        try:
            detector = HandPoseExtractor()
            retargeter = make_retargeter()
            robot = SimBackend(render_mode="rgb_array", rate=self.rate)
            robot.render()  # Initialize the GL context before acquiring camera frames.
            spec = ModelSpec.load()
            guard = CommandGuard(robot.read().positions, robot.lower, robot.upper)
            if self.stop_event.is_set():
                return
            camera = self.camera_factory()
            last_sequence = -1
            last_dispatch = time.monotonic() - 1 / self.rate
            origin = None
            had_target = False
            while not self.stop_event.is_set():
                with self.lock:
                    idle = time.monotonic() - self.last_client
                if idle > 45:
                    break
                if self.reset_event.is_set():
                    retargeter = make_retargeter()
                    calibration, had_target = 0, False
                    self.reset_event.clear()
                started = time.monotonic()
                frame, sequence, captured_at = camera.snapshot()
                if started - captured_at > 1:
                    raise RuntimeError("Camera stopped delivering fresh frames")
                if sequence == last_sequence:
                    self.stop_event.wait(0.005)
                    continue
                last_sequence = sequence
                if origin is None:
                    origin = captured_at
                pose = detector.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round((captured_at - origin) * 1000))
                points = target = None
                status = "holding" if had_target else "awaiting_hand"
                message = "未检测到右手，保持当前命令。" if had_target else "请张开右手，让整只手清楚入镜。"
                if pose is not None:
                    world, points, _ = pose
                    detected += 1
                    try:
                        result = retargeter.retarget(TargetPose(joint_positions=world, source="mediapipe"))
                        if result is None:
                            calibration = min(30, calibration + 1)
                            status, message = "calibrating", f"尺度标定 {calibration}/30 · 请保持五指张开。"
                        else:
                            q = from_orca(result)
                            q[-1] = spec.neutral[-1]
                            target = JointTarget(q, timestamp=captured_at)
                            status, message = "tracking", "正在跟随右手 · 可以缓慢握拳、伸食指或捏合。"
                    except (ValueError, RuntimeError) as exc:
                        status, message = "rejected", f"本帧求解未通过，保持当前命令：{type(exc).__name__}"
                dispatched = time.monotonic()
                dt = min(0.25, max(1e-5, dispatched - last_dispatch))
                last_dispatch = dispatched
                command, fresh = guard.apply(target, dt, now=dispatched)
                if target is not None and not fresh:
                    status, message = "stale", "输入已超过 250 ms，保持当前命令。"
                if fresh:
                    had_target = True
                    commanded += 1
                # Match simulated time to the measured interval, including slow frames.
                robot.env.frame_skip = max(1, round(dt / robot.env.model.opt.timestep))
                robot.write(command)
                actual = robot.read().positions
                simulation = cv2.cvtColor(robot.render(), cv2.COLOR_RGB2BGR)
                rgb_overlay = draw_overlay(frame, points, None)
                pair = {}
                for name, pixels in (("camera", rgb_overlay), ("simulation", simulation)):
                    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if not ok:
                        raise RuntimeError("Preview encoding failed")
                    pair[name] = base64.b64encode(encoded).decode("ascii")
                published = time.monotonic()
                count += 1
                published_times.append(published)
                durations.append((published - started) * 1000)
                fps = ((len(published_times) - 1) / (published_times[-1] - published_times[0])
                       if len(published_times) > 1 else 0.0)
                with self.lock:
                    self.captured_at, self.pair = captured_at, pair
                    self.state = {
                        "status": status, "message": message, "frame_id": count,
                        "camera_sequence": sequence, "detected_frames": detected, "commanded_frames": commanded,
                        "calibration_frames": calibration, "calibrated": had_target, "input_valid": fresh,
                        "fps": fps, "processing_ms": durations[-1],
                        "processing_p50_ms": float(np.median(durations)),
                        "processing_p95_ms": float(np.percentile(durations, 95)),
                        "frame_ready_age_ms": (published - captured_at) * 1000,
                        "command_deg": np.rad2deg(command).round(3).tolist(),
                        "actual_deg": np.rad2deg(actual).round(3).tolist(),
                    }
                self.stop_event.wait(max(0, started + 1 / self.rate - time.monotonic()))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            for resource in (camera, detector, robot):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        error = error or f"Cleanup failed: {exc}"
            with self.lock:
                self.pair = self.captured_at = None
                self.state.update(status="error" if error else "stopped",
                                  message=error or "实时跟随已结束，摄像头已关闭。")
