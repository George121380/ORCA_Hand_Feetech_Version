"""Camera policy loop. Default dry-run opens no servo bus; --execute enables hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from orca_feetech.fiducials import PoseWorker, TagTracker, check_transform
from orca_feetech.manipulation import DT, ActionCodec, ObservationCodec, rotation_error, yaw_rotation, limits
from orca_feetech.model import ModelSpec
from orca_feetech.checkpoint import load_config


def run(args):
    from stable_baselines3 import PPO
    config, _ = load_config(args.checkpoint)
    camera = json.loads(args.camera_config.read_text())
    alignment = json.loads(args.alignment.read_text())
    if alignment.get("verified") is not True:
        raise ValueError("Register the physical fixture with calibrate_vision.py align-world")
    T_SW = check_transform(alignment["T_SW"])
    tracker = TagTracker(config.object, camera)
    policy = PPO.load(args.checkpoint, device="cpu")
    backend = worker = None
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = int(args.source) if args.source.isdigit() else args.source
        worker = PoseWorker(source, tracker)
        print("Waiting for object and fixed reference tag 40 (10 second deadline).")
        deadline = time.monotonic() + 10
        initial = None
        while time.monotonic() < deadline:
            sample = worker.read()
            if sample and sample[0] is not None and sample[1] is not None and time.monotonic()-sample[2] < .2:
                initial = sample
                break
            time.sleep(.02)
        if initial is None:
            raise RuntimeError("No fresh joint object/reference detection; no servo bus opened")
        obj, ref, timestamp = initial
        T = T_SW @ np.linalg.inv(ref.T_CO) @ obj.T_CO
        origin, initial_rotation = T[:3, 3].copy(), T[:3, :3].copy()
        if np.linalg.norm(origin - [.135, .049, .185]) > .08:
            raise ValueError("Object is outside the trained palm workspace; check alignment/placement")
        lo, hi = limits(config)
        if args.execute:
            from orca_feetech.hardware import load_device, HardwareBackend
            _, _, measured_lo, measured_hi = load_device(args.device_config)
            if np.any(lo < measured_lo - 1e-7) or np.any(hi > measured_hi + 1e-7):
                raise ValueError("Policy ROM exceeds calibrated hardware ROM. Retrain with --device-config.")
            # Check initial motor state after backend arms with matching current goals.
            backend = HardwareBackend(args.device_config)
            q = backend.read().positions
            if np.max(np.abs(q-ModelSpec.load().neutral)) > .10:
                raise ValueError("Place hand in calibrated neutral before policy execution")
        else:
            q = ModelSpec.load().neutral.copy()
        codec, observer = ActionCodec(config, q), ObservationCodec(config)
        increment = yaw_rotation(np.deg2rad(config.angle) * args.direction)
        goal = increment @ initial_rotation
        position, rotation = origin.copy(), initial_rotation.copy()
        last_timestamp, consumed = timestamp, timestamp
        completed = stable = 0
        started = next_tick = time.monotonic()
        previous_pose = None
        print("EXECUTE: calibrated servos enabled." if args.execute else "DRY RUN: virtual joint feedback; no servo communication.")
        with args.log.open("x") as log:
            for step in range(config.horizon):
                now = time.monotonic()
                if now-started > args.seconds:
                    break
                if now-next_tick > .1:
                    raise RuntimeError("Control deadline missed by >100 ms; ending rollout")
                sample = worker.read()
                valid = False
                if sample and sample[2] > consumed:
                    obj, ref, stamp = sample
                    consumed = stamp
                    if obj is not None and ref is not None and 0 <= now-stamp <= .15:
                        T = T_SW @ np.linalg.inv(ref.T_CO) @ obj.T_CO
                        candidate_position, candidate_rotation = T[:3, 3], T[:3, :3]
                        gap = max(.001, stamp-last_timestamp)
                        # Gate transformed pose as well: reference-board flips must not move W.
                        if (np.linalg.norm(candidate_position-position) <= .02 + 1.5*gap and
                                rotation_error(candidate_rotation, rotation) <= .3 + 8*gap):
                            position, rotation = candidate_position.copy(), candidate_rotation.copy()
                            last_timestamp, valid = stamp, True
                age = max(0., now-last_timestamp)
                if age > .25:
                    raise RuntimeError("Object/reference pose stale >250ms; ending rollout")
                if position[2] < .10 or np.linalg.norm(position-origin) > .12:
                    raise RuntimeError("Object left trained retention region")
                if backend:
                    q = backend.read().positions
                obs = observer.push(q, codec.q, position, rotation, goal, origin, valid, age)
                action = policy.predict(obs, deterministic=True)[0]
                command = codec.step(action, valid=valid)
                if backend:
                    # The same timestamp that passed pose validity guards the existing watchdog.
                    backend.write(command, source_timestamp=last_timestamp)
                else:
                    q = command.copy()
                angle = rotation_error(rotation, goal)
                # Vision-only real metrics are deliberately separate from simulator
                # acceptance, which also requires physical contact.
                if valid and previous_pose is not None:
                    pp, pr, pt = previous_pose
                    sample_dt = max(.001, last_timestamp-pt)
                    slow = (np.linalg.norm(position-pp)/sample_dt < .08 and
                            rotation_error(rotation, pr)/sample_dt < 1.)
                else:
                    slow = False
                stable = stable + 1 if valid and slow and angle < np.deg2rad(8) and np.linalg.norm(position-origin) < .04 else 0
                if valid:
                    previous_pose = position.copy(), rotation.copy(), last_timestamp
                if stable >= 10:
                    completed += 1
                    stable = 0
                    if completed < config.turns:
                        goal = increment @ goal
                row = dict(time_s=now-started, execute=args.execute, q=q.tolist(), command=command.tolist(),
                           position=position.tolist(), rotation=rotation.tolist(), valid=valid, age_s=age,
                           angle_error_deg=float(np.rad2deg(angle)), completed_turns=completed)
                log.write(json.dumps(row)+"\n")
                log.flush()
                if completed >= config.turns:
                    print("Vision stability target reached; simulator contact acceptance not measured on hardware.")
                    break
                next_tick += DT
                time.sleep(max(0., next_tick-time.monotonic()))
    finally:
        # Serial shutdown precedes camera shutdown, including camera-driver failures.
        if backend:
            backend.close()
        if worker:
            worker.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--camera-config", type=Path, required=True)
    p.add_argument("--alignment", type=Path, required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--direction", type=int, choices=[-1, 1], default=1)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--device-config", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.execute and args.device_config is None:
        p.error("--execute requires a completed --device-config")
    if not 0 < args.seconds <= 60:
        p.error("--seconds must be in (0,60]")
    run(args)


if __name__ == "__main__":
    main()
