from __future__ import annotations

import time

import numpy as np

from .backends import create_backend
from .control import CommandGuard, JointTarget, minimum_jerk, move_duration
from .model import poses as gesture_poses
from .recording import SessionRecorder, load_trajectory


def run_targets(targets, *, backend="sim", config=None, headless=False, rate=20,
                output="runs/demo", video=False, fast=False):
    if not 4 <= rate <= 60:
        raise ValueError("Control rate must be 4..60 Hz")
    if backend == "hardware" and fast:
        raise ValueError("Hardware replay must run at real time")
    if video and backend != "sim":
        raise ValueError("--video records the simulated view; use --backend sim")
    render = "rgb_array" if video else None if headless or backend != "sim" else "human"
    recorder = SessionRecorder(output, backend, rate, video)
    robot, error = None, None
    try:
        robot = create_backend(backend, hardware_config=config, render_mode=render, rate=rate)
        initial = robot.read().positions
        guard = CommandGuard(initial, robot.lower, robot.upper)
        dt = 1 / rate
        wall_start = time.monotonic()
        last_dispatch = wall_start - dt
        count = 0
        for t, target_q in targets(initial):
            if not fast:
                time.sleep(max(0, wall_start + t - time.monotonic()))
            dispatched = time.monotonic()
            if backend == "hardware" and dispatched - (wall_start + t) > 0.25:
                raise RuntimeError("Control is over 250 ms behind schedule; stopping instead of replaying a backlog")
            target = JointTarget(target_q, timestamp=dispatched)
            effective_dt = dt if fast else min(0.25, max(1e-5, dispatched - last_dispatch))
            q, fresh = guard.apply(target, effective_dt, now=dispatched)
            last_dispatch = dispatched
            robot.write(q, target.timestamp)
            state = robot.read()
            frame = robot.render()
            recorder.append(target_q, q, state,
                            time_s=dispatched - wall_start if backend == "hardware" else t,
                            frame=frame, valid=fresh)
            count += 1
        return {"output": str(recorder.output), "samples": count, "backend": backend}
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
            recorder.close(error)


def demo(poses="open,fist,pinch,open", duration=2.5, rate=20, **kwargs):
    bank = gesture_poses()
    names = poses.split(",")
    if any(n not in bank for n in names):
        raise ValueError(f"Choose poses from {', '.join(bank)}")
    if duration <= 0:
        raise ValueError("Duration must be positive")

    def targets(initial):
        t, start = 0.0, initial
        for name in names:
            target = bank[name]
            seconds = move_duration(start, target, duration)
            steps = int(np.ceil(seconds * rate))
            for step in range(1, steps + 1):
                t += 1 / rate
                yield t, minimum_jerk(start, target, step / rate, seconds)
            for _ in range(rate):
                t += 1 / rate
                yield t, target
            start = target
    return run_targets(targets, rate=rate, **kwargs)


def replay(trajectory, rate=20, **kwargs):
    rows = load_trajectory(trajectory)
    times = np.array([r["time_s"] for r in rows])
    times -= times[0]
    commands = np.stack([r["command"] for r in rows])
    # A recording always begins with a fresh smooth transition from actual state.
    def targets(initial):
        seconds = move_duration(initial, commands[0])
        transition_steps = int(np.ceil(seconds * rate))
        for step in range(1, transition_steps + 1):
            yield step / rate, minimum_jerk(initial, commands[0], step / rate, seconds)
        for step in range(1, int(np.ceil(times[-1] * rate)) + 1):
            t = step / rate
            q = np.array([np.interp(t, times, commands[:, i]) for i in range(17)])
            yield (transition_steps + step) / rate, q
    return run_targets(targets, rate=rate, **kwargs)
