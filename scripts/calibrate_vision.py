"""Offline camera calibration and rigid reference-board registration."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from orca_feetech.fiducials import check_transform


def calibrate(args):
    pattern = (args.cols, args.rows)
    grid = np.zeros((args.cols * args.rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_m
    objects, images, size = [], [], None
    for path in sorted(glob.glob(args.images)):
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if size is not None and gray.shape[::-1] != size:
            raise ValueError("Calibration images must all have the same resolution")
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
        if found:
            objects.append(grid.copy())
            images.append(corners)
    if len(images) < 15:
        raise ValueError("Need >=15 varied board views, covering image corners and tilt")
    rms, K, distortion, rv, tv = cv2.calibrateCamera(objects, images, size, None, None)
    errors = []
    for obj, image, r, t in zip(objects, images, rv, tv, strict=True):
        projected = cv2.projectPoints(obj, r, t, K, distortion)[0]
        errors.append(float(np.sqrt(np.mean(np.sum((projected-image)**2, axis=2)))))
    if rms > .8:
        raise ValueError(f"Calibration RMS {rms:.3f}px > .8px; inspect board scale/focus/views")
    args.output.write_text(json.dumps(dict(K=K.tolist(), distortion=distortion.ravel().tolist(),
                                          image_size=size, rms_px=rms, per_view_error_px=errors,
                                          square_m=args.square_m, valid_views=len(images)), indent=2)+"\n")
    print(f"Saved calibration, RMS {rms:.3f}px, {len(images)} views")


def align(args):
    pairs = json.loads(args.points.read_text())
    w = np.asarray(pairs["world_points_m"], dtype=float)
    s = np.asarray(pairs["simulation_points_m"], dtype=float)
    if w.shape != s.shape or w.ndim != 2 or w.shape[1] != 3 or len(w) < 3 or not np.isfinite([w, s]).all():
        raise ValueError("Need >=3 finite corresponding 3D landmarks in both frames")
    wc, sc = w.mean(0), s.mean(0)
    if np.linalg.svd(w-wc, compute_uv=False)[1] < .01:
        raise ValueError("Landmarks must be non-collinear and span at least 1 cm")
    u, _, vt = np.linalg.svd((w-wc).T @ (s-sc))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, sc-rotation@wc
    residual = float(np.sqrt(np.mean(np.sum((w@rotation.T+T[:3, 3]-s)**2, axis=1))))
    check_transform(T)
    if residual > .003:
        raise ValueError(f"Landmark residual {residual*1000:.1f}mm >3mm; check mounting/units")
    args.output.write_text(json.dumps(dict(verified=True, T_SW=T.tolist(),
                                          landmark_rms_m=residual, correspondences=pairs), indent=2)+"\n")
    print(f"Saved board-to-simulation registration, RMS {residual*1000:.2f}mm")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("intrinsics")
    c.add_argument("--images", required=True, help="Quoted glob of unresized calibration photos")
    c.add_argument("--cols", type=int, default=9)
    c.add_argument("--rows", type=int, default=6)
    c.add_argument("--square-m", type=float, default=.02)
    a = sub.add_parser("align-world")
    a.add_argument("--points", type=Path, required=True)
    for cmd in (a, c):
        cmd.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "intrinsics":
        if min(args.cols, args.rows) < 3 or args.square_m <= 0:
            p.error("Invalid chessboard dimensions")
        calibrate(args)
    else:
        align(args)


if __name__ == "__main__":
    main()

