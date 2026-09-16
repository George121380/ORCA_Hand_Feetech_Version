from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .hardware import JointCalibration, STSBus
from .model import JOINT_NAMES, MODEL_ID, ModelSpec


def calibrate(config, joint, output):
    """Interactive commissioning only; never called implicitly by runtime code."""
    if joint not in JOINT_NAMES:
        raise ValueError(f"Unknown joint {joint}")
    output = Path(output)
    source = output if output.exists() else Path(config)
    data = copy.deepcopy(yaml.safe_load(source.read_text()))
    if data.get("model_id") != MODEL_ID or not data.get("port"):
        raise ValueError("Set model_id and the actual port in the hardware YAML first")
    entry = data["joints"][joint]
    mid = int(entry["motor_id"])
    spec = ModelSpec.load()
    index = JOINT_NAMES.index(joint)
    neutral_deg = float(np.rad2deg(spec.neutral[index]))
    bus = STSBus(data["port"], data.get("baudrate", 1_000_000))
    stop = threading.Event()
    activity = [time.monotonic()]
    fault = []
    thread = None

    def idle_release():
        while not stop.wait(0.2):
            if time.monotonic() - activity[0] > 10 and mid in bus.armed:
                try:
                    bus.torque([mid], False)
                    print("\n[空闲 10 秒，已卸力；下一次点动从当前位置重新启用]", flush=True)
                except OSError as exc:
                    fault.append(str(exc))
                    stop.set()

    try:
        model_number = bus.ping(mid)
        if bus.read_register(mid, 33) != 0:
            raise RuntimeError("Use the vendor setup tool to select STS position mode (0), then retry")
        print(f"关节 {joint} → 候选 ID {mid}, 型号码 {model_number}。先核对实际走线。")
        if input("确认只调试这个关节，输入该舵机 ID：").strip() != str(mid):
            raise ValueError("Motor mapping not confirmed")
        print(f"模型中立角：{neutral_deg:.3f}°；模型范围："
              f"{np.rad2deg(spec.lower[index]):.2f}° .. {np.rad2deg(spec.upper[index]):.2f}°")
        print("命令 + / -：点动 5 ticks；read：读位置；lower / neutral / upper：记录锚点；save：保存；q：卸力退出。")
        print("在关节处核对角度，使用安全端点，禁止把机械堵转位置当作标定目标。空闲 10 秒自动卸力。")
        anchors = {}
        thread = threading.Thread(target=idle_release, daemon=True)
        thread.start()
        while True:
            command = input(f"{joint}> ").strip().lower()
            if fault:
                raise RuntimeError(f"Calibration communication fault: {fault[-1]}")
            activity[0] = time.monotonic()
            if command in ("q", "quit", "exit"):
                return {"saved": False}
            if command == "save":
                if set(anchors) != {"lower", "neutral", "upper"}:
                    print("需要 lower、neutral、upper 三个锚点。")
                    continue
                candidate = {"motor_id": mid, "verified": True, "model_number": model_number,
                             "model_angles_deg": [anchors[k][0] for k in ("lower", "neutral", "upper")],
                             "ticks": [anchors[k][1] for k in ("lower", "neutral", "upper")],
                             "calibrated_at": datetime.now(timezone.utc).isoformat()}
                JointCalibration.parse(joint, candidate)
                data["joints"][joint] = candidate
                data["calibrated"] = all(v.get("verified") is True for v in data["joints"].values())
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".yaml.tmp")
                temporary.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
                temporary.replace(output)
                return {"saved": str(output), "joint": joint, "hand_complete": data["calibrated"]}
            row = bus.read([mid])[0]
            if row["temperature_c"] >= data.get("max_temperature_c", 55):
                raise RuntimeError("Temperature threshold reached")
            if command in ("+", "-"):
                tick = row["ticks"] + (5 if command == "+" else -5)
                if not 10 <= tick <= 4085:
                    raise ValueError("Encoder boundary reached; inspect spool position")
                if mid not in bus.armed:
                    bus.write([mid], [row["ticks"]], speed=20, acceleration=5)
                    bus.torque([mid], True)
                bus.write([mid], [tick], speed=20, acceleration=5)
                time.sleep(0.1)
                print(bus.read([mid])[0])
            elif command in ("lower", "neutral", "upper"):
                degrees = neutral_deg if command == "neutral" else float(input("此位置对应的模型关节角（度）："))
                if not np.rad2deg(spec.lower[index]) <= degrees <= np.rad2deg(spec.upper[index]):
                    raise ValueError("Anchor outside model range")
                row = bus.read([mid])[0]  # Do not retain feedback captured before a blocking prompt.
                anchors[command] = (degrees, row["ticks"])
                print(f"已记录 {command}: {degrees:.3f}°, {row['ticks']} ticks")
            elif command == "read":
                print(row)
    finally:
        stop.set()
        if thread:
            thread.join(timeout=1)
        bus.close()
