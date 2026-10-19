"""Read-only STS encoder/command logging and delay estimation for sim parameter checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np

from orca_feetech.hardware import STSBus, load_device


def record(args):
    config, cals, _, _ = load_device(args.device_config)
    bus = STSBus(config["port"], config.get("baudrate", 1000000))
    # No torque enable or writes. Use your existing debugger separately, with a
    # single bus owner; this logger is for manually moved, torque-off joints.
    started = time.monotonic()
    try:
        with args.output.open("x") as out:
            while time.monotonic()-started < args.seconds:
                rows = bus.read([c.motor_id for c in cals])
                q = [c.to_angle(r["ticks"]) for c, r in zip(cals, rows, strict=True)]
                out.write(json.dumps(dict(time_s=time.monotonic()-started, q=q, motors=rows))+"\n")
                out.flush()
                time.sleep(.05)
    finally:
        bus.close()


def fit(args):
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    if len(rows) < 50 or any(not row.get("execute") for row in rows):
        raise ValueError("Need >=50 executed (not dry-run) command/feedback samples")
    times = np.array([r["time_s"] for r in rows])
    q, commands = np.array([r["q"] for r in rows]), np.array([r["command"] for r in rows])
    if (not np.isfinite(times).all() or not np.isfinite([q, commands]).all()
            or q.shape[1:] != (17,) or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid timestamps or joint samples")
    fit_rows = []
    for joint in range(16):
        if np.ptp(commands[:, joint]) < .1:
            fit_rows.append(dict(joint=joint, fitted=False, reason="Insufficient excitation (<0.1 rad)"))
            continue
        candidates = []
        for delay in np.arange(0, .301, .01):
            valid = times >= times[0]+delay
            target = np.interp(times[valid]-delay, times, commands[:, joint])
            residual = q[valid, joint]-target
            bias = float(np.median(residual))
            rms = float(np.sqrt(np.mean((residual-bias)**2)))
            candidates.append((rms, float(delay), bias))
        rms, delay, bias = min(candidates)
        fit_rows.append(dict(joint=joint, fitted=True, delay_s=delay, bias_rad=bias, residual_rms_rad=rms))
    args.output.write_text(json.dumps(dict(joints=fit_rows, notes="Encoder response only; does not identify tendon "
                                          "to finger-angle error or contact friction. Use external angle measurements. "
                                          "Do not treat fitted delay as a full actuator dynamics model."), indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record-passive")
    r.add_argument("--device-config", type=Path, required=True)
    r.add_argument("--seconds", type=float, default=10)
    f = sub.add_parser("fit")
    f.add_argument("--log", type=Path, required=True)
    for cmd in (r, f):
        cmd.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "record-passive":
        if not 0 < args.seconds <= 300:
            p.error("Duration must be within 0..300 seconds")
        record(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
