"""Calibrated, multi-tag object pose estimation; no camera/control side effects."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np

from .objects import DICTIONARY, marker_layout
from .manipulation import rotation_error


def transform(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def check_transform(value):
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Expected a finite rigid 4x4 transform")
    return T


@dataclass
class PoseEstimate:
    T_CO: np.ndarray
    reprojection_px: float
    ids: list[int]
    timestamp: float


def solve_pose(layout, corners, ids, K, distortion, previous=None, min_pixels=40, max_error=2.5,
               timestamp=None):
    """Use all usable decoded corners in ONE object-frame PnP, not pose averaging.

    IPPE enumerates planar ambiguity; SQPnP initializes non-coplanar boards.
    A large, nearly tied planar ambiguity is rejected rather than guessed.
    """
    mapping = {t["id"]: t for t in layout}
    points, pixels, used = [], [], []
    for image, ident in zip(corners, np.asarray(ids).ravel(), strict=True):
        if int(ident) not in mapping:
            continue
        image = np.asarray(image, dtype=float).reshape(4, 2)
        if min(np.linalg.norm(np.roll(image, -1, axis=0) - image, axis=1)) < min_pixels:
            continue
        points.extend(mapping[int(ident)]["corners_m"])
        pixels.extend(image)
        used.append(int(ident))
    if not points:
        return None
    points, pixels = np.array(points, dtype=np.float64), np.array(pixels, dtype=np.float64)
    planar = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[-1] < 1e-7
    methods = [cv2.SOLVEPNP_SQPNP]
    if planar:
        methods.append(cv2.SOLVEPNP_IPPE)
    solutions = [cv2.solvePnPGeneric(points, pixels, K, distortion, flags=method) for method in methods]
    candidates = []
    pairs = [(rv, tv) for solution in solutions if solution[0]
             for rv, tv in zip(solution[1], solution[2], strict=True)]
    for rv, tv in pairs:
        # IPPE can produce NaN for degenerate views.
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            continue
        T = transform(rv, tv)
        camera_points = points @ T[:3, :3].T + T[:3, 3]
        if camera_points[:, 2].min() <= 0:
            continue
        visible = True
        for ident in used:
            tag = mapping[ident]
            normal = np.cross(tag["u"], tag["v"])
            center = T[:3, :3] @ tag["center_m"] + T[:3, 3]
            if np.dot(T[:3, :3] @ normal, center) >= 0:
                visible = False
        if not visible:
            continue
        projected = cv2.projectPoints(points, rv, tv, K, distortion)[0].reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - pixels)**2, axis=1))))
        if error <= max_error:
            candidates.append((error, T))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return None
    unique = []
    for candidate in candidates:
        if all(rotation_error(candidate[1][:3, :3], other[1][:3, :3]) > np.deg2rad(1)
               for other in unique):
            unique.append(candidate)
    candidates = unique
    if (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .15
            and rotation_error(candidates[0][1][:3, :3], candidates[1][1][:3, :3]) > np.deg2rad(10)):
        return None
    error, T = candidates[0]
    if previous is not None:
        # Only gate jumps over a short interval. Lost tracking does not make an
        # arbitrarily old pose into a valid new observation.
        dt = (timestamp or time.monotonic()) - previous.timestamp
        if 0 < dt < .25 and (np.linalg.norm(T[:3, 3] - previous.T_CO[:3, 3]) > .02 + 1.5 * dt
                            or rotation_error(T[:3, :3], previous.T_CO[:3, :3]) > .3 + 8 * dt):
            return None
    return PoseEstimate(T, error, used, timestamp if timestamp is not None else time.monotonic())


def world_layout():
    size = .05
    return [dict(id=40, u=[1, 0, 0], v=[0, 1, 0], center_m=[0, 0, 0],
                 corners_m=[[-size/2, size/2, 0], [size/2, size/2, 0],
                            [size/2, -size/2, 0], [-size/2, -size/2, 0]])]


class TagTracker:
    def __init__(self, kind, camera):
        self.K = np.asarray(camera["K"], dtype=float)
        self.distortion = np.asarray(camera["distortion"], dtype=float)
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or not np.isfinite(self.distortion).all() or min(self.K[0, 0], self.K[1, 1]) <= 0):
            raise ValueError("Invalid camera intrinsics")
        self.size = tuple(camera["image_size"])
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.layout = marker_layout(kind)
        self.previous = None

    def detect(self, frame, timestamp=None):
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError("Capture resolution differs from calibration; do not resize frames")
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None, None
        object_pose = solve_pose(self.layout, corners, ids, self.K, self.distortion,
                                 previous=self.previous, timestamp=timestamp)
        reference = solve_pose(world_layout(), corners, ids, self.K, self.distortion, timestamp=timestamp)
        if object_pose:
            self.previous = object_pose
        return object_pose, reference


class PoseWorker:
    """Camera read and inference cannot block the policy's control deadline."""
    def __init__(self, source, tracker):
        self.tracker = tracker
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, tracker.size[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, tracker.size[1])
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Cannot open the selected camera")
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.fault = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                # Conservative timestamp before read: slow capture counts as age.
                timestamp = time.monotonic()
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed")
                obj, ref = self.tracker.detect(frame, timestamp)
                with self.lock:
                    self.latest = (obj, ref, timestamp)
        except Exception as exc:
            with self.lock:
                self.fault = str(exc)

    def read(self):
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.5)
        # Daemon may still be in a driver read. The control loop has already stopped.
        if not self.thread.is_alive():
            self.capture.release()
